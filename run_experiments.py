import os
import random
import argparse
from typing import Dict, List, Tuple
import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import pandas as pd
from torch import nn

from src.models.transformer import StandardTransformer
from src.models.htransformer import HMUTransformer, HMUTransformerAfterDecoder  # Mantengo una sola variante para claridad
from src.evaluation.metrics import TransformerEvaluator
from src.evaluation.visualization import ResultsVisualizer
from src.data.data_generator import DataGenerator
from src.data.commongen_loader import CommonGenLoader
from src.data.rocstories_loader import ROCStoriesLoader
from src.data.agnews_loader import AGNewsLoader
from src.data.sst2_loader import SST2Loader
from src.data.glue_loader import GLUELoader

# ------------------------------
# Utils
# ------------------------------

PAD_ID = 0  # Asumimos que 0 es el token <pad> en el DataGenerator

def set_seed(seed: int):
    """Set random seed for reproducibility on CPU & GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collate_fn(batch: List[Dict[str, torch.Tensor]], device: torch.device):
    """Stack examples and move to device."""
    src = torch.stack([b["src"] for b in batch], dim=0).to(device)
    tgt = torch.stack([b["tgt"] for b in batch], dim=0).to(device)
    return {"src": src, "tgt": tgt}

# ------------------------------
# Training
# ------------------------------

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    learning_rate: float = 3e-5,
    warmup_steps: int = 4_000,
    eval_every: int = 1,
    grad_accum_steps: int = 1,
) -> Tuple[torch.nn.Module, List[float], List[float], List[float], List[float], float]:
    """Train with mixed precision + scheduler. Returns model, train_losses, eval_losses, train_accs, eval_accs, train_time."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    train_losses, eval_losses = [], []
    train_accs, eval_accs = [], []
    global_step = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss, step_loss = 0.0, 0.0
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step, batch in enumerate(pbar, 1):
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                output = model(batch["src"], batch["tgt"])
                loss = criterion(output.view(-1, output.size(-1)), batch["tgt"].view(-1)) / grad_accum_steps
                preds = output.argmax(dim=-1)
                mask = batch["tgt"] != PAD_ID
                correct += ((preds == batch["tgt"]) & mask).sum().item()
                total += mask.sum().item()
            scaler.scale(loss).backward()
            step_loss += loss.item()
            if step % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            pbar.set_postfix({"loss": f"{step_loss:.4f}", "acc": f"{(correct/total):.4f}"})
            epoch_loss += loss.item()
            step_loss = 0 if step % grad_accum_steps == 0 else step_loss
        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)
        train_acc = correct / total if total > 0 else 0.0
        train_accs.append(train_acc)
        print(f"\nEpoch {epoch + 1}: avg train loss = {avg_train:.4f}, train acc = {train_acc:.4f}")
        # Eval -------------------------------------------------------
        if (epoch + 1) % eval_every == 0:
            model.eval()
            eval_loss = 0.0
            correct, total = 0, 0
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc="Evaluating"):
                    with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                        output = model(batch["src"], batch["tgt"])
                        loss = criterion(output.view(-1, output.size(-1)), batch["tgt"].view(-1))
                        preds = output.argmax(dim=-1)
                        mask = batch["tgt"] != PAD_ID
                        correct += ((preds == batch["tgt"]) & mask).sum().item()
                        total += mask.sum().item()
                    eval_loss += loss.item()
            avg_eval = eval_loss / len(eval_loader)
            eval_losses.append(avg_eval)
            eval_acc = correct / total if total > 0 else 0.0
            eval_accs.append(eval_acc)
            print(f"Eval loss = {avg_eval:.4f}, eval acc = {eval_acc:.4f}")
        else:
            avg_eval = float("nan")
            eval_acc = float("nan")
        # WandB ------------------------------------------------------
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                f"{model.__class__.__name__}_train_loss": avg_train,
                f"{model.__class__.__name__}_eval_loss": avg_eval,
                f"{model.__class__.__name__}_train_acc": train_acc,
                f"{model.__class__.__name__}_eval_acc": eval_acc,
            })
    train_time = time.time() - start_time
    return model, train_losses, eval_losses, train_accs, eval_accs, train_time

# ------------------------------
# Testing
# ------------------------------

def test_models(
    standard_transformer: torch.nn.Module,
    htransformer: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    tokenizer=None,
    eval_batch_size: int = 8,
    dataset_type: str = "synthetic"
):
    evaluator = TransformerEvaluator()

    # Recolectamos tensores para el evaluador
    all_src = []
    all_tgt = []
    for batch in test_loader:
        all_src.append(batch["src"].cpu())
        all_tgt.append(batch["tgt"].cpu())
    all_src = torch.cat(all_src, dim=0)
    all_tgt = torch.cat(all_tgt, dim=0)

    if dataset_type != "synthetic":
        # Evaluación normal: accuracy y loss
        std_results = hmu_results = None
        criterion = torch.nn.CrossEntropyLoss()
        total = 0
        if standard_transformer is not None:
            standard_transformer.eval()
            std_total_acc, std_total_loss = 0.0, 0.0
            std_total_exact, total = 0.0, 0
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Testing (Standard)"):
                    src = batch["src"]
                    tgt = batch["tgt"]
                    std_output = standard_transformer(src.to(device), tgt.to(device))
                    std_loss = criterion(std_output.view(-1, std_output.size(-1)), tgt.view(-1).to(device)).item()
                    batch_size = src.size(0)
                    std_total_loss += std_loss * batch_size
                    std_total_acc += evaluator.compute_sequence_accuracy(std_output.cpu(), tgt.cpu(), pad_id=PAD_ID) * batch_size
                    std_total_exact += evaluator.compute_exact_sequence_match(std_output.cpu(), tgt.cpu(), pad_id=PAD_ID) * batch_size
                    total += batch_size
            std_results = {"loss": std_total_loss / total, "accuracy": std_total_acc / total, "exact_match": std_total_exact / total}
        if htransformer is not None:
            htransformer.eval()
            hmu_total_acc, hmu_total_loss = 0.0, 0.0
            hmu_total_exact, total = 0.0, 0
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Testing (HTransformer)"):
                    src = batch["src"]
                    tgt = batch["tgt"]
                    hmu_output = htransformer(src.to(device), tgt.to(device))
                    hmu_loss = criterion(hmu_output.view(-1, hmu_output.size(-1)), tgt.view(-1).to(device)).item()
                    batch_size = src.size(0)
                    hmu_total_loss += hmu_loss * batch_size
                    hmu_total_acc += evaluator.compute_sequence_accuracy(hmu_output.cpu(), tgt.cpu(), pad_id=PAD_ID) * batch_size
                    hmu_total_exact += evaluator.compute_exact_sequence_match(hmu_output.cpu(), tgt.cpu(), pad_id=PAD_ID) * batch_size
                    total += batch_size
            hmu_results = {"loss": hmu_total_loss / total, "accuracy": hmu_total_acc / total, "exact_match": hmu_total_exact / total}
        return std_results, hmu_results, None, None

    # Si es synthetic, evaluación completa
    all_pairs = []
    for batch in test_loader:
        for i in range(batch["src"].size(0)):
            all_pairs.append((batch["src"][i].cpu(), batch["tgt"][i].cpu()))

    std_mem = hmu_mem = std_img = hmu_img = None
    if standard_transformer is not None:
        print("\nEvaluating long‑term memory (Standard Transformer)…")
        std_mem = evaluator.evaluate_long_term_memory(standard_transformer, all_src, all_pairs, batch_size=eval_batch_size)
        print("\nEvaluating imagination (Standard Transformer)…")
        std_img = evaluator.evaluate_imagination(standard_transformer, all_src, all_tgt, tokenizer=tokenizer, batch_size=eval_batch_size, dataset_type=dataset_type)
    if htransformer is not None:
        print("\nEvaluating long‑term memory (HTransformer)…")
        hmu_mem = evaluator.evaluate_long_term_memory(htransformer, all_src, all_pairs, batch_size=eval_batch_size)
        print("\nEvaluating imagination (HTransformer)…")
        hmu_img = evaluator.evaluate_imagination(htransformer, all_src, all_tgt, tokenizer=tokenizer, batch_size=eval_batch_size, dataset_type=dataset_type)

    return std_mem, hmu_mem, std_img, hmu_img

# ------------------------------
# Model Classes
# ------------------------------

class GLUEClassifier(nn.Module):
    def __init__(self, encoder, d_model, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, src):
        # Use the encoder's encode method to get the representation
        # This will include HMU for HTransformer and just encoder for StandardTransformer
        x = self.encoder.encode(src)
        
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.pool(x).squeeze(-1)  # (B, D)
        return self.classifier(x)

class AGNewsClassifier(nn.Module):
    def __init__(self, encoder, d_model, num_classes=4):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, src):
        # Use the encoder's encode method to get the representation
        # This will include HMU for HTransformer and just encoder for StandardTransformer
        x = self.encoder.encode(src)
        
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.pool(x).squeeze(-1)  # (B, D)
        return self.classifier(x)

# ------------------------------
# Model Creation Helpers
# ------------------------------

def create_model(model_type: str, args, vocab_size: int, dataset_type: str, num_classes: int = None):
    """Create a model based on type and dataset."""
    n_heads = args.d_model // 32  # Calculate number of heads consistently
    
    # Handle GLUE datasets
    if dataset_type in ["mrpc", "rte", "cola", "qnli", "sst2", "mnli"]:
        if num_classes is None:
            num_classes = 2  # Default for GLUE datasets
        
        if model_type == "transformer":
            return GLUEClassifier(
                StandardTransformer(
                    d_model=args.d_model,
                    nhead=n_heads,
                    num_encoder_layers=4,
                    num_decoder_layers=4,
                    dim_feedforward=args.d_model * 4,
                    dropout=0.1,
                    vocab_size=vocab_size,
                ),
                d_model=args.d_model,
                num_classes=num_classes
            )
        elif model_type == "htransformer":
            return GLUEClassifier(
                HMUTransformer(
                    d_model=args.d_model,
                    nhead=n_heads,
                    num_encoder_layers=4,
                    num_decoder_layers=4,
                    dim_feedforward=args.d_model * 4,
                    latent_dim=args.d_model // 2,
                    dropout=0.1,
                    vocab_size=vocab_size,
                ),
                d_model=args.d_model,
                num_classes=num_classes
            )
        elif model_type == "htransformerafterdecoder":
            return GLUEClassifier(
                HMUTransformerAfterDecoder(
                    d_model=args.d_model,
                    nhead=n_heads,
                    num_encoder_layers=4,
                    num_decoder_layers=4,
                    dim_feedforward=args.d_model * 4,
                    latent_dim=args.d_model // 2,
                    dropout=0.1,
                    vocab_size=vocab_size,
                ),
                d_model=args.d_model,
                num_classes=num_classes
            )
    elif dataset_type == "agnews":
        if model_type == "transformer":
            return AGNewsClassifier(
                StandardTransformer(
                    d_model=args.d_model,
                    nhead=n_heads,
                    num_encoder_layers=4,
                    num_decoder_layers=4,
                    dim_feedforward=args.d_model * 4,
                    dropout=0.1,
                    vocab_size=vocab_size,
                ),
                d_model=args.d_model,
                num_classes=4
            )
        elif model_type == "htransformer":
            return AGNewsClassifier(
                HMUTransformer(
                    d_model=args.d_model,
                    nhead=n_heads,
                    num_encoder_layers=4,
                    num_decoder_layers=4,
                    dim_feedforward=args.d_model * 4,
                    latent_dim=args.d_model // 2,
                    dropout=0.1,
                    vocab_size=vocab_size,
                ),
                d_model=args.d_model,
                num_classes=4
            )
        elif model_type == "htransformerafterdecoder":
            return AGNewsClassifier(
                HMUTransformerAfterDecoder(
                    d_model=args.d_model,
                    nhead=n_heads,
                    num_encoder_layers=4,
                    num_decoder_layers=4,
                    dim_feedforward=args.d_model * 4,
                    latent_dim=args.d_model // 2,
                    dropout=0.1,
                    vocab_size=vocab_size,
                ),
                d_model=args.d_model,
                num_classes=4
            )
    else:
        # Non-classification datasets
        if model_type == "transformer":
            return StandardTransformer(
                d_model=args.d_model,
                nhead=n_heads,
                num_encoder_layers=4,
                num_decoder_layers=4,
                dim_feedforward=args.d_model * 4,
                dropout=0.1,
                vocab_size=vocab_size,
            )
        elif model_type == "htransformer":
            return HMUTransformer(
                d_model=args.d_model,
                nhead=n_heads,
                num_encoder_layers=4,
                num_decoder_layers=4,
                dim_feedforward=args.d_model * 4,
                latent_dim=args.d_model // 2,
                dropout=0.1,
                vocab_size=vocab_size,
            )
        elif model_type == "htransformerafterdecoder":
            return HMUTransformerAfterDecoder(
                d_model=args.d_model,
                nhead=n_heads,
                num_encoder_layers=4,
                num_decoder_layers=4,
                dim_feedforward=args.d_model * 4,
                latent_dim=args.d_model // 2,
                dropout=0.1,
                vocab_size=vocab_size,
            )
    raise ValueError(f"Unknown model type: {model_type}")

def count_model_parameters(model, model_name: str):
    """Count and display detailed parameter breakdown for a model."""
    total_params = 0
    print(f"\n{model_name} Parameter Breakdown:")
    print("-" * 50)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        #print(f"{name:40s}: {param_count:>10,} params")
    
    print("-" * 50)
    print(f"Total {model_name} parameters: {total_params:,}")
    return total_params

def train_single_model(model, model_type: str, args, train_loader, eval_loader, device, dataset_type: str):
    """Train a single model and return results."""
    print(f"\nTraining {model_type}...")
    
    # Detailed parameter counting
    count_model_parameters(model, model_type)
    
    model = model.to(device)
    
    # Check if it's a classification task
    is_classification = dataset_type in ["agnews", "sst2", "mrpc", "rte", "cola", "qnli", "mnli"]
    
    if is_classification:
        # Classification training
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        train_losses, eval_losses, train_accs, eval_accs = [], [], [], []
        
        for epoch in range(args.epochs):
            model.train()
            epoch_loss, correct, total = 0.0, 0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for batch in pbar:
                output = model(batch["src"])
                loss = criterion(output, batch["tgt"].to(device))
                pred = output.argmax(dim=1)
                correct += (pred == batch["tgt"].to(device)).sum().item()
                total += batch["tgt"].size(0)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total):.4f}"})
            
            avg_train = epoch_loss / len(train_loader)
            train_losses.append(avg_train)
            train_accs.append(correct / total)
            
            # Evaluation
            model.eval()
            eval_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for batch in eval_loader:
                    output = model(batch["src"])
                    loss = criterion(output, batch["tgt"].to(device))
                    pred = output.argmax(dim=1)
                    correct += (pred == batch["tgt"].to(device)).sum().item()
                    total += batch["tgt"].size(0)
                    eval_loss += loss.item()
            
            avg_eval = eval_loss / len(eval_loader)
            eval_losses.append(avg_eval)
            eval_accs.append(correct / total)
            
            print(f"Epoch {epoch+1}: avg train loss = {avg_train:.4f}, train acc = {train_accs[-1]:.4f}")
            print(f"Eval loss = {avg_eval:.4f}, eval acc = {eval_accs[-1]:.4f}")
        
        return {
            'train_loss': train_losses,
            'eval_loss': eval_losses,
            'train_acc': train_accs,
            'eval_acc': eval_accs
        }
    else:
        # Standard sequence-to-sequence training
        model, tr_loss, ev_loss, tr_acc, ev_acc, train_time = train_model(
            model, train_loader, eval_loader, device,
            num_epochs=args.epochs, grad_accum_steps=args.grad_accum)
        
        return {
            'train_loss': tr_loss,
            'eval_loss': ev_loss,
            'train_acc': tr_acc,
            'eval_acc': ev_acc
        }

def evaluate_single_model(model, model_type: str, args, test_loader, device, dataset_type: str):
    """Evaluate a single model and return results."""
    model.eval()
    
    # Check if it's a classification task
    is_classification = dataset_type in ["agnews", "sst2", "mrpc", "rte", "cola", "qnli", "mnli"]
    
    if is_classification:
        criterion = nn.CrossEntropyLoss()
        correct, total, loss_total = 0, 0, 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                output = model(batch["src"])
                loss = criterion(output, batch["tgt"].to(device))
                pred = output.argmax(dim=1)
                correct += (pred == batch["tgt"].to(device)).sum().item()
                total += batch["tgt"].size(0)
                loss_total += loss.item() * batch["tgt"].size(0)
        
        return {
            "loss": loss_total / total,
            "accuracy": correct / total
        }
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
        test_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch in test_loader:
                output = model(batch["src"], batch["tgt"])
                loss = criterion(output.view(-1, output.size(-1)), batch["tgt"].view(-1))
                preds = output.argmax(dim=-1)
                mask = batch["tgt"] != PAD_ID
                correct += ((preds == batch["tgt"]) & mask).sum().item()
                total += mask.sum().item()
                test_loss += loss.item()
        
        return {
            "loss": test_loss / len(test_loader),
            "accuracy": correct / total if total > 0 else 0.0
        }

# ------------------------------
# Main
# ------------------------------

def main():
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser("Transformer experiment runner (revised)")
    parser.add_argument("--mode", choices=["all", "train", "benchmark"], default="all")
    parser.add_argument("--model", choices=["transformer", "htransformer", "htransformerafterdecoder"], default="htransformer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--dataset_size", type=int, default=20_000)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dataset", choices=["synthetic", "commongen", "rocstories", "agnews", "sst2", "mrpc", "rte", "cola", "qnli", "mnli", "glue"], default="synthetic")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use per dataset (useful for large datasets like QNLI)")
    #para experimentos rapidos max_samples = 100
    #para experimentos medianos max_samples = 5000
    #para experimentos completos max_samples = 20000
    parser.add_argument("--proof", action="store_true", default=False, help="Proof mode: use very small dataset sizes for quick testing")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size para evaluación (imagination y long-term memory)")
    parser.add_argument("--tokenizer_name", type=str, default="t5-small", help="Nombre del tokenizer/modelo de HuggingFace (por defecto t5-small)")
    args = parser.parse_args()

    # Apply proof mode settings
    if args.proof:
        print("🔬 PROOF MODE ENABLED: Using small dataset sizes for quick testing")
        # Set very small limits for all datasets
        if args.max_samples is None:  # Only override if not explicitly set
            args.max_samples = 100  # Very small size for proof mode
        args.epochs = min(args.epochs, 1)  # Max 1 epoch in proof mode
        args.batch_size = min(args.batch_size, 8)  # Small batch size
        print(f"  - Max samples per dataset: {args.max_samples}")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Batch size: {args.batch_size}")

    seed = 42
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(project="transformer-comparison-revised", config=vars(args))

    # ---------------------- Data ----------------------
    if args.dataset == "glue":
        # Run experiments on all GLUE datasets
        glue_datasets = ["mrpc", "rte", "cola", "qnli", "sst2", "mnli"]
        all_glue_results = {}
        
        print(f"\n{'='*60}")
        print("RUNNING EXPERIMENTS ON ALL GLUE DATASETS")
        print(f"{'='*60}")
        
        for glue_dataset in glue_datasets:
            print(f"\n{'='*50}")
            print(f"PROCESSING DATASET: {glue_dataset.upper()}")
            print(f"{'='*50}")
            
            # Load data for this GLUE dataset
            glue_train_loader = GLUELoader(dataset_name=glue_dataset, split="train", max_length=args.max_length, tokenizer_name=args.tokenizer_name, max_samples=args.max_samples)
            glue_dev_loader = GLUELoader(dataset_name=glue_dataset, split="validation", max_length=args.max_length, tokenizer_name=args.tokenizer_name, max_samples=args.max_samples)
            
            train_data = glue_train_loader.get_data()
            eval_data = glue_dev_loader.get_data()
            test_data = glue_dev_loader.get_data()  # Use dev as test since test labels are not available
            
            vocab_size = glue_train_loader.get_vocab_size()
            tokenizer = glue_train_loader.get_tokenizer()
            num_classes = glue_train_loader.get_num_classes()
            
            print(f"Dataset: {glue_dataset.upper()}")
            print(f"Train samples: {len(train_data)}")
            print(f"Eval samples: {len(eval_data)}")
            print(f"Number of classes: {num_classes}")
            
            # Create data loaders
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=lambda b: collate_fn(b, device))
            eval_loader  = DataLoader(eval_data,  batch_size=args.batch_size, shuffle=False,
                                      collate_fn=lambda b: collate_fn(b, device))
            test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False,
                                      collate_fn=lambda b: collate_fn(b, device))
            
            # Train and evaluate models
            all_models = ["transformer", "htransformer"]
            dataset_results = {}
            
            for model_type in all_models:
                print(f"\nTraining {model_type} on {glue_dataset.upper()}...")
                
                # Create model
                current_model = create_model(model_type, args, vocab_size, glue_dataset, num_classes)
                
                # Train model
                training_results = train_single_model(
                    current_model, model_type, args, train_loader, eval_loader, device, glue_dataset
                )
                
                # Save model
                torch.save(current_model.state_dict(), f"results/{model_type}_{glue_dataset}.pt")
                
                # Evaluate on test set
                eval_results = evaluate_single_model(
                    current_model, model_type, args, test_loader, device, glue_dataset
                )
                dataset_results[model_type] = eval_results
                
                # Clean up
                del current_model
                torch.cuda.empty_cache()
            
            # Store results for this dataset
            all_glue_results[glue_dataset] = dataset_results
            
            # Print summary for this dataset
            print(f"\nResults for {glue_dataset.upper()}:")
            summary_df = pd.DataFrame(dataset_results).T
            print(summary_df)
        
        # Create comprehensive GLUE results summary
        print(f"\n{'='*60}")
        print("COMPREHENSIVE GLUE RESULTS")
        print(f"{'='*60}")
        
        # Create a comprehensive DataFrame
        comprehensive_results = {}
        for dataset_name, dataset_results in all_glue_results.items():
            for model_name, model_results in dataset_results.items():
                comprehensive_results[f"{dataset_name}_{model_name}"] = model_results
        
        comprehensive_df = pd.DataFrame(comprehensive_results).T
        comprehensive_df.to_csv("results/glue_comprehensive_results.csv", index=True, index_label="Dataset_Model")
        print("\nComprehensive GLUE results saved to results/glue_comprehensive_results.csv")
        print(comprehensive_df)
        
        # Create dataset-wise summary
        dataset_summary = {}
        for dataset_name, dataset_results in all_glue_results.items():
            dataset_summary[dataset_name] = {
                'transformer_accuracy': dataset_results['transformer']['accuracy'],
                'htransformer_accuracy': dataset_results['htransformer']['accuracy'],
                'transformer_loss': dataset_results['transformer']['loss'],
                'htransformer_loss': dataset_results['htransformer']['loss']
            }
        
        dataset_summary_df = pd.DataFrame(dataset_summary).T
        dataset_summary_df.to_csv("results/glue_dataset_summary.csv", index=True, index_label="Dataset")
        print("\nDataset-wise summary saved to results/glue_dataset_summary.csv")
        print(dataset_summary_df)
        
        wandb.finish()
        return
        
    elif args.dataset in ["mrpc", "rte", "cola", "qnli", "mnli"]:
        # Use GLUE loader for these datasets
        glue_train_loader = GLUELoader(dataset_name=args.dataset, split="train", max_length=args.max_length, tokenizer_name=args.tokenizer_name, max_samples=args.max_samples)
        glue_dev_loader = GLUELoader(dataset_name=args.dataset, split="validation", max_length=args.max_length, tokenizer_name=args.tokenizer_name, max_samples=args.max_samples)
        
        train_data = glue_train_loader.get_data()
        eval_data = glue_dev_loader.get_data()
        test_data = glue_dev_loader.get_data()  # Use dev as test since test labels are not available
        
        vocab_size = glue_train_loader.get_vocab_size()
        tokenizer = glue_train_loader.get_tokenizer()
        num_classes = glue_train_loader.get_num_classes()
        
        print(f"GLUE Dataset: {args.dataset.upper()}")
        print(f"Train samples: {len(train_data)}")
        print(f"Eval samples: {len(eval_data)}")
        print(f"Number of classes: {num_classes}")
        
    elif args.dataset == "commongen":
        commongen_loader = CommonGenLoader(split="train", max_length=args.max_length, tokenizer_name=args.tokenizer_name)
        all_data = commongen_loader.get_data()
        vocab_size = commongen_loader.get_vocab_size()
        tokenizer = commongen_loader.get_tokenizer()
        n = len(all_data)
        tr, ev = int(0.8 * n), int(0.1 * n)
        train_data, eval_data, test_data = all_data[:tr], all_data[tr:tr+ev], all_data[tr+ev:]
        num_classes = None
    elif args.dataset == "rocstories":
        roc_loader = ROCStoriesLoader(split="train", max_length=args.max_length, tokenizer_name=args.tokenizer_name)
        all_data = roc_loader.get_data()
        vocab_size = roc_loader.get_vocab_size()
        tokenizer = roc_loader.get_tokenizer()
        n = len(all_data)
        tr, ev = int(0.8 * n), int(0.1 * n)
        train_data, eval_data, test_data = all_data[:tr], all_data[tr:tr+ev], all_data[tr+ev:]
        num_classes = None
    elif args.dataset == "agnews":
        agnews_loader = AGNewsLoader(split="train", max_length=args.max_length, tokenizer_name=args.tokenizer_name)
        all_data = agnews_loader.get_data()
        vocab_size = agnews_loader.get_vocab_size()
        tokenizer = agnews_loader.get_tokenizer()
        n = len(all_data)
        tr, ev = int(0.8 * n), int(0.1 * n)
        train_data, eval_data, test_data = all_data[:tr], all_data[tr:tr+ev], all_data[tr+ev:]
        num_classes = 4
    elif args.dataset == "sst2":
        # SST-2 has predefined train/dev splits, so we use them directly
        sst2_train_loader = SST2Loader(split="train", max_length=args.max_length, tokenizer_name=args.tokenizer_name)
        sst2_dev_loader = SST2Loader(split="validation", max_length=args.max_length, tokenizer_name=args.tokenizer_name)
        
        train_data = sst2_train_loader.get_data()
        eval_data = sst2_dev_loader.get_data()
        test_data = sst2_dev_loader.get_data()  # Use dev as test since test labels are not available
        
        vocab_size = sst2_train_loader.get_vocab_size()
        tokenizer = sst2_train_loader.get_tokenizer()
        num_classes = 2
    else:
        data_gen = DataGenerator(seed=seed)
        def gen_data(n: int):
            data = []
            patterns = ["random", "temporal", "hierarchical", "repetitive"]
            per = n // len(patterns)
            for p in patterns:
                data.extend(data_gen.generate_synthetic_data(per, args.seq_len, vocab_size=1000, pattern_type=p))
            return data
        all_data = gen_data(args.dataset_size)
        random.shuffle(all_data)
        tr, ev = int(0.7 * len(all_data)), int(0.2 * len(all_data))
        train_data, eval_data, test_data = all_data[:tr], all_data[tr:tr+ev], all_data[tr+ev:]
        vocab_size = 1000
        tokenizer = None
        num_classes = None
        
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, device))
    eval_loader  = DataLoader(eval_data,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, device))
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, device))

    # ---------------------- Models ----------------------
    # (No model creation needed here anymore - handled by helper functions)

    # ---------------------- Train / Eval ----------------------
    if args.mode in {"all", "train"}:
        if args.mode == "all":
            # Train and evaluate all models
            all_models = ["transformer", "htransformer"]
            all_training_results = {}
            all_eval_results = {}
            
            for model_type in all_models:
                print(f"\n{'='*50}")
                print(f"TRAINING AND EVALUATING {model_type.upper()}")
                print(f"{'='*50}")
                
                # Create model
                current_model = create_model(model_type, args, vocab_size, args.dataset, num_classes)
                
                # Train model
                training_results = train_single_model(
                    current_model, model_type, args, train_loader, eval_loader, device, args.dataset
                )
                all_training_results[model_type] = training_results
                
                # Save model
                torch.save(current_model.state_dict(), f"results/{model_type}.pt")
                
                # Evaluate on test set
                eval_results = evaluate_single_model(
                    current_model, model_type, args, test_loader, device, args.dataset
                )
                all_eval_results[model_type] = eval_results
                
                # Clean up
                del current_model
                torch.cuda.empty_cache()
            
            # Generate comparative results
            print(f"\n{'='*50}")
            print("COMPARATIVE RESULTS")
            print(f"{'='*50}")
            
            # Create summary DataFrame
            summary_df = pd.DataFrame(all_eval_results).T
            summary_df.to_csv("results/comparative_summary.csv", index=True, index_label="Model")
            print("\nComparative summary saved to results/comparative_summary.csv")
            print(summary_df)
            
            # Create training curves comparison
            vis = ResultsVisualizer(use_wandb=True)
            
            # Plot training curves for all models
            if args.dataset not in ["agnews", "sst2", "mrpc", "rte", "cola", "qnli"]:
                vis.plot_training_curves(
                    all_training_results["transformer"]["train_loss"],
                    all_training_results["transformer"]["eval_loss"],
                    all_training_results["htransformer"]["train_loss"],
                    all_training_results["htransformer"]["eval_loss"],
                    "results/training_curves_comparison.png"
                )
            
            # For synthetic dataset, run full evaluation
            if args.dataset == "synthetic":
                print(f"\n{'='*50}")
                print("RUNNING FULL EVALUATION (MEMORY & IMAGINATION)")
                print(f"{'='*50}")
                
                all_memory_results = {}
                all_imagination_results = {}
                
                for model_type in all_models:
                    print(f"\nEvaluating {model_type}...")
                    
                    # Recreate model for evaluation
                    eval_model = create_model(model_type, args, vocab_size, args.dataset, num_classes)
                    
                    # Load trained weights
                    eval_model.load_state_dict(torch.load(f"results/{model_type}.pt", map_location="cpu", weights_only=True), strict=False)
                    eval_model = eval_model.to(device)
                    
                    # Evaluate memory and imagination
                    evaluator = TransformerEvaluator()
                    
                    # Memory evaluation
                    all_pairs = []
                    for batch in test_loader:
                        for i in range(batch["src"].size(0)):
                            all_pairs.append((batch["src"][i].cpu(), batch["tgt"][i].cpu()))
                    
                    all_src = []
                    all_tgt = []
                    for batch in test_loader:
                        all_src.append(batch["src"].cpu())
                        all_tgt.append(batch["tgt"].cpu())
                    all_src = torch.cat(all_src, dim=0)
                    all_tgt = torch.cat(all_tgt, dim=0)
                    
                    print(f"Evaluating long-term memory for {model_type}...")
                    memory_results = evaluator.evaluate_long_term_memory(eval_model, all_src, all_pairs, batch_size=args.eval_batch_size)
                    all_memory_results[model_type] = memory_results
                    
                    print(f"Evaluating imagination for {model_type}...")
                    imagination_results = evaluator.evaluate_imagination(eval_model, all_src, all_tgt, tokenizer=tokenizer, batch_size=args.eval_batch_size, dataset_type=args.dataset)
                    all_imagination_results[model_type] = imagination_results
                    
                    del eval_model
                    torch.cuda.empty_cache()
                
                # Create comparative visualizations
                vis.plot_memory_retention(
                    all_memory_results["transformer"],
                    all_memory_results["htransformer"],
                    "results/memory_retention_comparison.png"
                )
                
                vis.plot_imagination_metrics(
                    all_imagination_results["transformer"],
                    all_imagination_results["htransformer"],
                    "results/imagination_comparison.png"
                )
                
                # Create comprehensive summary
                comprehensive_results = {}
                for model_type in all_models:
                    comprehensive_results[model_type] = {**all_memory_results[model_type], **all_imagination_results[model_type]}
                
                comprehensive_df = pd.DataFrame(comprehensive_results).T
                comprehensive_df.to_csv("results/comprehensive_evaluation.csv", index=True, index_label="Model")
                print("\nComprehensive evaluation saved to results/comprehensive_evaluation.csv")
                print(comprehensive_df)
            
            wandb.finish()
            return
            
        else:
            # Single model training
            current_model = create_model(args.model, args, vocab_size, args.dataset, num_classes)
            
            # Train model
            training_results = train_single_model(
                current_model, args.model, args, train_loader, eval_loader, device, args.dataset
            )
            
            # Save model
            torch.save(current_model.state_dict(), f"results/{args.model}.pt")
            
            del current_model
            torch.cuda.empty_cache()
            
            if args.mode == "train":
                print("Training completed.")
                return

    # ---------------------- Benchmark ----------------------
    if args.mode == "benchmark":
        # Create model for evaluation
        current_model = create_model(args.model, args, vocab_size, args.dataset, num_classes)
        
        # Load trained weights
        current_model.load_state_dict(torch.load(f"results/{args.model}.pt", map_location="cpu", weights_only=True), strict=False)
        current_model = current_model.to(device)
        current_model.eval()
        
        # Check if it's a classification task
        is_classification = args.dataset in ["agnews", "sst2", "mrpc", "rte", "cola", "qnli", "mnli"]
        
        if is_classification:
            # Classification evaluation
            criterion = nn.CrossEntropyLoss()
            correct, total, loss_total = 0, 0, 0.0
            
            with torch.no_grad():
                for batch in test_loader:
                    output = current_model(batch["src"])
                    loss = criterion(output, batch["tgt"].to(device))
                    pred = output.argmax(dim=1)
                    correct += (pred == batch["tgt"].to(device)).sum().item()
                    total += batch["tgt"].size(0)
                    loss_total += loss.item() * batch["tgt"].size(0)
            
            results = {
                "loss": loss_total / total,
                "accuracy": correct / total
            }
        else:
            # Non-classification evaluation
            if args.dataset == "synthetic":
                # Full evaluation for synthetic dataset
                evaluator = TransformerEvaluator()
                
                # Memory evaluation
                all_pairs = []
                for batch in test_loader:
                    for i in range(batch["src"].size(0)):
                        all_pairs.append((batch["src"][i].cpu(), batch["tgt"][i].cpu()))
                
                all_src = []
                all_tgt = []
                for batch in test_loader:
                    all_src.append(batch["src"].cpu())
                    all_tgt.append(batch["tgt"].cpu())
                all_src = torch.cat(all_src, dim=0)
                all_tgt = torch.cat(all_tgt, dim=0)
                
                print(f"Evaluating long-term memory for {args.model}...")
                memory_results = evaluator.evaluate_long_term_memory(current_model, all_src, all_pairs, batch_size=args.eval_batch_size)
                
                print(f"Evaluating imagination for {args.model}...")
                imagination_results = evaluator.evaluate_imagination(current_model, all_src, all_tgt, tokenizer=tokenizer, batch_size=args.eval_batch_size, dataset_type=args.dataset)
                
                results = {**memory_results, **imagination_results}
                
                # Create visualizations
                vis = ResultsVisualizer(use_wandb=True)
                vis.plot_memory_retention(memory_results, None, "results/memory_retention.png")
                vis.plot_imagination_metrics(imagination_results, None, "results/imagination.png")
            else:
                # Simple evaluation for other datasets
                results = evaluate_single_model(current_model, args.model, args, test_loader, device, args.dataset)
        
        # Create summary
        summary_df = pd.DataFrame({
            args.model: results
        })
        summary_df.to_csv("results/summary.csv", index=True, index_label="Metric")
        print("\nSummary saved to results/summary.csv")
        print(summary_df)
        
        del current_model
        torch.cuda.empty_cache()

    wandb.finish()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
