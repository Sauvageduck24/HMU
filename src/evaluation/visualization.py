import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import wandb
import torch

class ResultsVisualizer:
    def __init__(self, use_wandb: bool = True):
        """Initialize the visualizer.
        
        Args:
            use_wandb: Whether to log results to Weights & Biases
        """
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="transformer-comparison")
    
    def plot_memory_retention(
        self,
        standard_results: Dict[str, float],
        htransformer_results: Dict[str, float],
        save_path: str = None
    ):
        """Plot memory retention comparison."""
        time_steps = [int(k.split('_')[1]) for k in standard_results.keys() if k.startswith('retention_')]
        standard_retentions = [standard_results[f'retention_{t}'] for t in time_steps]
        htransformer_retentions = [htransformer_results[f'retention_{t}'] for t in time_steps]
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, standard_retentions, 'b-', label='Standard Transformer')
        plt.plot(time_steps, htransformer_retentions, 'r-', label='HTransformer')
        plt.xlabel('Time Steps')
        plt.ylabel('Memory Retention')
        plt.title('Memory Retention Over Time')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        if self.use_wandb:
            wandb.log({"memory_retention": wandb.Image(plt.gcf())})
        
        plt.close()
    
    def plot_imagination_metrics(
        self,
        standard_results: Dict[str, float],
        htransformer_results: Dict[str, float],
        save_path: str = None
    ):
        """Plot imagination quality metrics comparison."""
        metrics = ['coherence', 'diversity', 'perplexity']
        standard_values = [standard_results[m] for m in metrics]
        htransformer_values = [htransformer_results[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, standard_values, width, label='Standard Transformer')
        plt.bar(x + width/2, htransformer_values, width, label='HTransformer')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Imagination Quality Metrics Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        
        if self.use_wandb:
            wandb.log({"imagination_metrics": wandb.Image(plt.gcf())})
        
        plt.close()
    
    def plot_sequence_length_impact(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: str = None
    ):
        """Plot the impact of sequence length on model performance."""
        seq_lengths = sorted(results.keys())
        metrics = ['retention_100', 'coherence', 'diversity']
        
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            standard_values = [results[seq_len]['standard'][metric] for seq_len in seq_lengths]
            htransformer_values = [results[seq_len]['htransformer'][metric] for seq_len in seq_lengths]
            
            plt.plot(seq_lengths, standard_values, 'b--', label=f'Standard {metric}')
            plt.plot(seq_lengths, htransformer_values, 'r-', label=f'HTransformer {metric}')
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Score')
        plt.title('Impact of Sequence Length on Model Performance')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        if self.use_wandb:
            wandb.log({"sequence_length_impact": wandb.Image(plt.gcf())})
        
        plt.close()
    
    def create_summary_table(
        self,
        standard_results: Dict[str, float],
        htransformer_results: Dict[str, float]
    ) -> pd.DataFrame:
        """Create a summary table of all metrics."""
        metrics = {
            'Memory Retention (100 steps)': (
                standard_results['retention_100'],
                htransformer_results['retention_100']
            ),
            'Memory Retrieval Accuracy': (
                standard_results['retrieval_accuracy'],
                htransformer_results['retrieval_accuracy']
            ),
            'Sequence Coherence': (
                standard_results['coherence'],
                htransformer_results['coherence']
            ),
            'Sequence Diversity': (
                standard_results['diversity'],
                htransformer_results['diversity']
            ),
            'Perplexity': (
                standard_results['perplexity'],
                htransformer_results['perplexity']
            )
        }
        
        df = pd.DataFrame(
            metrics,
            index=['Standard Transformer', 'HTransformer']
        ).T
        
        if self.use_wandb:
            wandb.log({"summary_table": wandb.Table(dataframe=df)})
        
        return df
    
    def plot_attention_patterns(
        self,
        standard_attention: torch.Tensor,
        htransformer_attention: torch.Tensor,
        save_path: str = None
    ):
        """Plot attention patterns for both models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.heatmap(
            standard_attention.cpu().numpy(),
            ax=ax1,
            cmap='viridis',
            title='Standard Transformer Attention'
        )
        
        sns.heatmap(
            htransformer_attention.cpu().numpy(),
            ax=ax2,
            cmap='viridis',
            title='HTransformer Attention'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if self.use_wandb:
            wandb.log({"attention_patterns": wandb.Image(plt.gcf())})
        
        plt.close()
    
    def log_metrics(
        self,
        standard_results: Dict[str, float],
        htransformer_results: Dict[str, float],
        step: int = None
    ):
        """Log all metrics to Weights & Biases."""
        if not self.use_wandb:
            return
        
        metrics = {}
        for key in standard_results:
            metrics[f'standard_{key}'] = standard_results[key]
            metrics[f'htransformer_{key}'] = htransformer_results[key]
        
        if step is not None:
            metrics['step'] = step
        
        wandb.log(metrics)
    
    def plot_training_curves(
        self,
        standard_train_losses: List[float],
        standard_eval_losses: List[float],
        htransformer_train_losses: List[float],
        htransformer_eval_losses: List[float],
        save_path: str = None
    ):
        """Plot training and evaluation curves for both models.
        
        Args:
            standard_train_losses: Training losses for Standard Transformer
            standard_eval_losses: Evaluation losses for Standard Transformer
            htransformer_train_losses: Training losses for HTransformer
            htransformer_eval_losses: Evaluation losses for HTransformer
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot training losses
        epochs = range(1, len(standard_train_losses) + 1)
        plt.plot(epochs, standard_train_losses, 'b-', label='Standard Transformer (Train)')
        plt.plot(epochs, htransformer_train_losses, 'r-', label='HTransformer (Train)')
        
        # Plot evaluation losses
        eval_epochs = range(1, len(standard_eval_losses) + 1)
        plt.plot(eval_epochs, standard_eval_losses, 'b--', label='Standard Transformer (Eval)')
        plt.plot(eval_epochs, htransformer_eval_losses, 'r--', label='HTransformer (Eval)')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Curves')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        if self.use_wandb:
            wandb.log({"training_curves": wandb.Image(plt.gcf())})
        
        plt.close() 