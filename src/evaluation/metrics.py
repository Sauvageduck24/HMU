import torch
import gc
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
import nltk
import torch
from tqdm import tqdm
import gc
from torch.utils.data import DataLoader, TensorDataset
from contextlib import nullcontext
nltk.download('punkt', quiet=True)

class TransformerEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def compute_sequence_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor, pad_id=None) -> float:
        """Compute accuracy for sequence predictions. Si pad_id no es None, ignora esos tokens. Si targets.shape[1]==1, compara solo el primer token (clasificación)."""
        if targets.dim() == 2 and targets.size(1) == 1:
            # Clasificación: solo primer token
            pred_flat = predictions.argmax(dim=-1)[:, 0]
            target_flat = targets[:, 0]
        else:
            pred_flat = predictions.argmax(dim=-1).flatten()
            target_flat = targets.flatten()
            min_len = min(len(pred_flat), len(target_flat))
            pred_flat = pred_flat[:min_len]
            target_flat = target_flat[:min_len]
            if pad_id is not None:
                mask = target_flat != pad_id
                pred_flat = pred_flat[mask]
                target_flat = target_flat[mask]
        if len(target_flat) == 0:
            return float('nan')
        return accuracy_score(target_flat.cpu(), pred_flat.cpu())
    
    def compute_exact_sequence_match(self, predictions: torch.Tensor, targets: torch.Tensor, pad_id=None) -> float:
        """Devuelve el % de secuencias donde toda la secuencia predicha coincide con el target. Si targets.shape[1]==1, compara solo el primer token."""
        if targets.dim() == 2 and targets.size(1) == 1:
            pred_tokens = predictions.argmax(dim=-1)[:, 0]
            tgt_tokens = targets[:, 0]
            matches = (pred_tokens == tgt_tokens).tolist()
        else:
            pred_tokens = predictions.argmax(dim=-1)  # (B, T)
            matches = []
            for pred_seq, tgt_seq in zip(pred_tokens, targets):
                if pad_id is not None:
                    tgt_mask = tgt_seq != pad_id
                    pred_seq = pred_seq[tgt_mask]
                    tgt_seq = tgt_seq[tgt_mask]
                matches.append(torch.equal(pred_seq, tgt_seq))
        return sum(matches) / len(matches) if matches else float("nan")
    
    def compute_memory_retention(
        self,
        model_outputs: torch.Tensor,      # (B, L, D)
        original_sequence: torch.Tensor,  # (B, L) o (B, L, D)
        time_steps: int
    ) -> torch.Tensor:
        """
        Vectorizada: calcula la retención para cada secuencia del batch.
        Si original_sequence es (B, L), se asume que son índices y no embeddings.
        Si es (B, L, D), se usa directamente.
        """
        # Si original_sequence es (B, L), conviértelo a float para similitud
        if original_sequence.dim() == 2:
            # No hay embedding aquí, así que solo convierte a float
            original_sequence = original_sequence.float().unsqueeze(-1)  # (B, L, 1)
            model_outputs = model_outputs.float()
            # Repite para igualar dimensiones
            original_sequence = original_sequence.expand_as(model_outputs)
        # Calcula la similitud coseno entre cada secuencia y su output
        cos_sim = torch.nn.functional.cosine_similarity(
            original_sequence, model_outputs, dim=-1
        )  # (B, L)
        return cos_sim.mean(dim=1)  # (B,)
    
    def compute_imagination_quality(
        self,
        generated_sequences: torch.Tensor,
        reference_sequences: torch.Tensor,
        tokenizer=None
    ) -> Dict[str, float]:
        """Compute various metrics for imagination quality, incluyendo ROUGE-2 y BLEU."""
        metrics = {}
        
        # Compute perplexity
        metrics['perplexity'] = self._compute_perplexity(generated_sequences, reference_sequences)
        
        # Compute sequence coherence
        metrics['coherence'] = self._compute_sequence_coherence(generated_sequences)
        
        # Compute diversity
        metrics['diversity'] = self._compute_sequence_diversity(generated_sequences)
        
        # ROUGE-2 y BLEU (si hay tokenizer)
        if tokenizer is not None:
            preds = generated_sequences.argmax(dim=-1).cpu().tolist()
            refs = reference_sequences.cpu().tolist()
            pred_texts = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
            ref_texts = [tokenizer.decode(r, skip_special_tokens=True) for r in refs]
            # BLEU
            bleu = nltk.translate.bleu_score.corpus_bleu([[ref.split()] for ref in ref_texts], [pred.split() for pred in pred_texts], weights=(0.5, 0.5, 0, 0))
            metrics['bleu'] = bleu
            # ROUGE-2
            scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
            rouge2_scores = [scorer.score(ref, pred)['rouge2'].fmeasure for ref, pred in zip(ref_texts, pred_texts)]
            metrics['rouge2'] = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
        
        return metrics
    
    def _compute_perplexity(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor
    ) -> float:
        """Compute perplexity between generated and reference sequences."""
        cross_entropy = torch.nn.functional.cross_entropy(
            generated.view(-1, generated.size(-1)),
            reference.view(-1),
            reduction='mean'
        )
        return torch.exp(cross_entropy).item()
    
    def _compute_sequence_coherence(self, sequence: torch.Tensor) -> float:
        """Compute how coherent the generated sequence is."""
        # Calculate average attention weights between consecutive tokens
        attention_weights = torch.matmul(sequence, sequence.transpose(-2, -1))
        return attention_weights.mean().item()
    
    def _compute_sequence_diversity(self, sequence: torch.Tensor) -> float:
        """Compute the diversity of the generated sequence."""
        # Calculate unique token ratio
        unique_tokens = torch.unique(sequence.argmax(dim=-1))
        return len(unique_tokens) / sequence.numel()
    
    def evaluate_long_term_memory(
        self,
        model,
        test_sequences: torch.Tensor,
        memory_tasks: List[Tuple[torch.Tensor, torch.Tensor]],
        batch_size=32
    ) -> Dict[str, float]:
        """Evaluate model performance on long-term memory tasks, ahora con batching y optimizaciones."""
        results = {}
        device = next(model.parameters()).device
        model.eval()
        # Test memory retention over different time steps
        for time_step in [10, 50, 100, 200]:
            retention_scores = []
            n = test_sequences.size(0)
            for i in tqdm(range(0, n, batch_size), desc=f"Retention t={time_step}"):
                batch = test_sequences[i:i+batch_size].to(device)
                try:
                    with torch.no_grad():
                        # For HMU models, we need to use encode to get the memory representation
                        # This will include the HMU processing for HMUTransformer
                        if hasattr(model, 'hmu') and hasattr(model, 'encode'):
                            output = model.encode(batch)
                        else:
                            # For standard transformer, use encode directly
                            output = model.encode(batch)
                    # Vectorizada: calcula retención para cada secuencia del batch
                    # Si compute_memory_retention no soporta batch, usa un bucle
                    if hasattr(self, 'compute_memory_retention_batch'):
                        retention = self.compute_memory_retention_batch(output, batch, time_step)
                        retention_scores.extend(retention.tolist())
                    else:
                        for j in range(batch.size(0)):
                            r = self.compute_memory_retention(output[j:j+1], batch[j:j+1], time_step)
                            retention_scores.append(r)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        import gc
                        torch.cuda.empty_cache()
                        gc.collect()
                        print("[WARNING] OOM detected, batch skipped.")
                    else:
                        raise
            # Convierte a numpy si es tensor
            if isinstance(retention_scores, torch.Tensor):
                retention_scores = retention_scores.cpu().numpy()
            elif isinstance(retention_scores, list) and len(retention_scores) > 0 and isinstance(retention_scores[0], torch.Tensor):
                retention_scores = [r.cpu().item() if r.numel() == 1 else r.cpu().numpy() for r in retention_scores]
            results[f'retention_{time_step}'] = np.mean(retention_scores)
        # Test memory retrieval accuracy
        retrieval_scores = []
        n = len(memory_tasks)
        for i in tqdm(range(0, n, batch_size), desc="Retrieval accuracy"):
            batch = memory_tasks[i:i+batch_size]
            queries = torch.stack([q for q, _ in batch]).to(device)
            targets = torch.stack([t for _, t in batch]).to(device)
            try:
                with torch.no_grad():
                    # Use full forward pass for HMU models to ensure correct processing
                    prediction = model(queries, targets)
                accs = self.compute_sequence_accuracy(prediction, targets)
                if isinstance(accs, float):
                    retrieval_scores.append(accs)
                else:
                    retrieval_scores.extend(accs.tolist())
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("[WARNING] OOM detected, batch skipped.")
                else:
                    raise
        results['retrieval_accuracy'] = np.mean(retrieval_scores)
        return results
    
    def evaluate_imagination(
        self,
        model,
        test_sequences: torch.Tensor,
        reference_sequences: torch.Tensor,
        tokenizer=None,
        batch_size=32,
        dataset_type="synthetic"
    ) -> Dict[str, float]:
        """Efficient GPU evaluation for imagination tasks. Si dataset_type es commongen, solo accuracy."""
        results = {}
        device = next(model.parameters()).device
        model.eval()

        # Dataset & Loader
        dataset = TensorDataset(test_sequences, reference_sequences)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # AMP context (automatic mixed precision)
        autocast_ctx = torch.amp.autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()

        generated_all = []
        ref_all = []

        for src_batch, tgt_batch in tqdm(loader, desc="Evaluating imagination (batches)"):
            src_batch = src_batch.to(device, non_blocking=True)
            tgt_batch = tgt_batch.to(device, non_blocking=True)

            try:
                with autocast_ctx:
                    with torch.no_grad():
                        # Use full forward pass instead of separate encode/decode for HMU models
                        # This ensures the HMU is applied correctly in both architectures
                        generated = model(src_batch, tgt_batch)

                generated_all.append(generated.cpu())
                ref_all.append(tgt_batch.cpu())

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("[WARNING] OOM detected, batch skipped.")
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise

            del src_batch, tgt_batch, generated
            torch.cuda.empty_cache()

        # Aggregate results
        generated_sequences = torch.cat(generated_all, dim=0)
        reference_sequences = torch.cat(ref_all, dim=0)

        if dataset_type == "commongen":
            # Solo accuracy como en train
            acc = self.compute_sequence_accuracy(generated_sequences, reference_sequences)
            results['accuracy'] = acc
            return results
        else:
            # Compute metrics
            quality_metrics = self.compute_imagination_quality(
                generated_sequences, reference_sequences, tokenizer=tokenizer
            )
            results.update(quality_metrics)
            return results
