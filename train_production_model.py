#!/usr/bin/env python3
"""
Production ML Model Training Script

Train production-ready ML models on the generated training data.
Focused on simplicity and reliability without complex dependencies.

Usage:
    python train_production_model.py --data-path ./production_data/production_training_data_*.json
    python train_production_model.py --evaluate-model ./ml_models/production_model_v1
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

from open_memory_suite.dispatcher import ML_AVAILABLE

if ML_AVAILABLE:
    from open_memory_suite.dispatcher import MLTrainer, MLPolicy

console = Console()

class ProductionModelTrainer:
    """Simple trainer for production ML models."""
    
    def __init__(self, output_dir: Path):
        """Initialize the trainer."""
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
    async def train_from_data_file(
        self,
        data_path: Path,
        model_name: str = "production_model",
        **training_config
    ) -> Dict[str, Any]:
        """Train a model from a data file."""
        
        rprint(f"🚀 [bold blue]Training production ML model from {data_path}[/bold blue]")
        
        # Load training data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        rprint(f"📊 Loaded {len(raw_data)} training examples")
        
        # Convert to the format expected by MLTrainer
        training_data = []
        for example in raw_data:
            # Handle both dictionary and object formats
            if isinstance(example, dict):
                training_example = {
                    "text": example["text"],
                    "label": example["label"],
                    "action": example["action"],
                    "context_features": example["context_features"],
                }
            else:
                # Assume it's already in the right format
                training_example = {
                    "text": example.text,
                    "label": example.label,
                    "action": example.action,
                    "context_features": example.context_features,
                }
            
            training_data.append(training_example)
        
        # Create trainer with production settings
        model_config = {
            "num_actions": 4,  # store, summarize, drop, defer
            "lora_r": 16,
            "lora_alpha": 32,
            **training_config.get("model_config", {})
        }
        
        training_settings = {
            "num_epochs": 3,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "max_length": 512,
            "weight_decay": 0.01,
            "eval_steps": 50,
            "save_steps": 100,
            "logging_steps": 10,
            "warmup_steps": 50,
            "early_stopping_patience": 2,
            "gradient_accumulation_steps": 1,
            **training_config.get("training_config", {})
        }
        
        trainer = MLTrainer(
            output_dir=self.output_dir,
            use_wandb=False,  # Disable for production
            model_config=model_config,
            training_config=training_settings,
        )
        
        # Train the model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Training model...", total=100)
            
            try:
                results = await trainer.train(
                    training_data=training_data,
                    validation_split=0.2,
                    experiment_name=model_name,
                )
                progress.update(task, completed=100)
                
            except Exception as e:
                rprint(f"❌ Training failed: {e}")
                raise
        
        # Display results
        self._display_training_results(results)
        
        return results
    
    def _display_training_results(self, results: Dict[str, Any]) -> None:
        """Display training results in a nice table."""
        
        results_table = Table(title="Training Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="magenta")
        
        # Core metrics
        results_table.add_row("Final Train Loss", f"{results.get('train_loss', 'N/A'):.4f}")
        
        # Evaluation results
        eval_results = results.get('eval_results', {})
        if eval_results:
            results_table.add_row("Validation Loss", f"{eval_results.get('eval_loss', 'N/A'):.4f}")
            results_table.add_row("Validation Accuracy", f"{eval_results.get('eval_accuracy', 'N/A'):.4f}")
            results_table.add_row("Validation F1", f"{eval_results.get('eval_f1', 'N/A'):.4f}")
        
        # Model info
        results_table.add_row("Model Path", str(results.get('model_path', 'N/A')))
        results_table.add_row("Training Time", f"{results.get('training_time_minutes', 'N/A'):.2f} min")
        
        console.print(results_table)
        
        rprint(f"✅ [bold green]Model saved to: {results.get('model_path')}[/bold green]")
    
    async def evaluate_model(self, model_path: Path) -> Dict[str, Any]:
        """Evaluate a trained model."""
        
        rprint(f"🔍 [bold blue]Evaluating model at {model_path}[/bold blue]")
        
        # Load the model
        ml_policy = MLPolicy(
            model_path=model_path,
            confidence_threshold=0.5,
            fallback_to_heuristic=False,
        )
        
        await ml_policy.initialize()
        
        # Get model statistics
        stats = ml_policy.get_stats()
        
        # Display stats
        self._display_model_stats(stats)
        
        return stats
    
    def _display_model_stats(self, stats: Dict[str, Any]) -> None:
        """Display model statistics."""
        
        stats_table = Table(title="Model Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")
        
        stats_table.add_row("Policy Name", stats.get("policy_name", "N/A"))
        stats_table.add_row("Policy Version", stats.get("policy_version", "N/A"))
        stats_table.add_row("Model Loaded", "✅" if stats.get("model_loaded") else "❌")
        stats_table.add_row("Device", stats.get("device", "N/A"))
        stats_table.add_row("Total Decisions", str(stats.get("decisions_made", 0)))
        stats_table.add_row("ML Decisions", str(stats.get("ml_decisions", 0)))
        stats_table.add_row("Heuristic Fallbacks", str(stats.get("heuristic_fallbacks", 0)))
        
        if stats.get("avg_inference_time_ms"):
            stats_table.add_row("Avg Inference Time", f"{stats['avg_inference_time_ms']:.1f}ms")
        
        console.print(stats_table)

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train production ML models")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to training data JSON file")
    parser.add_argument("--output-dir", type=str, default="./ml_models",
                       help="Output directory for trained models")
    parser.add_argument("--model-name", type=str, default="production_model_v1",
                       help="Name for the trained model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--evaluate-model", type=str,
                       help="Path to model to evaluate (instead of training)")
    
    args = parser.parse_args()
    
    if not ML_AVAILABLE:
        console.print("❌ [bold red]ML components not available[/bold red]")
        console.print("Install with: poetry install")
        return
    
    trainer = ProductionModelTrainer(Path(args.output_dir))
    
    try:
        if args.evaluate_model:
            # Evaluate existing model
            await trainer.evaluate_model(Path(args.evaluate_model))
        else:
            # Train new model
            training_config = {
                "training_config": {
                    "num_epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                }
            }
            
            results = await trainer.train_from_data_file(
                data_path=Path(args.data_path),
                model_name=args.model_name,
                **training_config
            )
            
            rprint("\n🎉 [bold green]Training completed successfully![/bold green]")
            
    except Exception as e:
        rprint(f"❌ [bold red]Error: {e}[/bold red]")
        raise

if __name__ == "__main__":
    asyncio.run(main())