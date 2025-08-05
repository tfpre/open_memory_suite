#!/usr/bin/env python3
"""
ML Policy Training Script

This script provides a command-line interface for training the ML-enhanced
memory routing policy using DistilBERT + LoRA fine-tuning.

Usage:
    python train_ml_policy.py --collect-data
    python train_ml_policy.py --train --data-path ./training_data.json
    python train_ml_policy.py --evaluate --model-path ./ml_models/latest
    python train_ml_policy.py --full-pipeline  # Collect data + train + evaluate

Features:
- Automated data collection from heuristic policy decisions
- DistilBERT + LoRA fine-tuning with experiment tracking
- Comprehensive evaluation and model comparison
- Production-ready model export and deployment
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

# Import project modules
from open_memory_suite.adapters import InMemoryAdapter, FileStoreAdapter, FAISStoreAdapter
from open_memory_suite.benchmark import BenchmarkHarness, CostModel
from open_memory_suite.dispatcher import (
    HeuristicPolicy,
    MLPolicy,
    MLTrainer,
    DataCollector,
    FrugalDispatcher,
    PolicyRegistry,
    ML_AVAILABLE,
)


console = Console()


class MLTrainingPipeline:
    """Complete ML training pipeline for memory routing policy."""
    
    def __init__(self, output_dir: Path = Path("./ml_training_output")):
        """Initialize the training pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.heuristic_policy = None
        self.harness = None
        self.data_collector = None
        self.trainer = None
        
        # Results
        self.training_data = None
        self.trained_model_path = None
        self.evaluation_results = None
    
    async def initialize_components(self) -> None:
        """Initialize the pipeline components."""
        rprint("🔧 [bold blue]Initializing ML training pipeline...[/bold blue]")
        
        # Create adapters for data collection
        adapters = [
            InMemoryAdapter("memory_store"),
            FileStoreAdapter("file_store", self.output_dir / "file_storage"),
        ]
        
        # Try to add FAISS adapter if available
        try:
            faiss_adapter = FAISStoreAdapter("faiss_store", embedding_dim=384)
            adapters.append(faiss_adapter)
        except Exception as e:
            console.print(f"⚠️ FAISS adapter not available: {e}")
        
        # Initialize adapters
        for adapter in adapters:
            await adapter.initialize()
        
        # Create components
        self.heuristic_policy = HeuristicPolicy()
        
        cost_model = CostModel()
        self.harness = BenchmarkHarness(
            adapters=adapters,
            cost_model=cost_model,
            trace_file=self.output_dir / "training_traces.jsonl"
        )
        
        self.data_collector = DataCollector(
            heuristic_policy=self.heuristic_policy,
            harness=self.harness,
            output_path=self.output_dir / "training_data"
        )
        
        self.trainer = MLTrainer(
            output_dir=self.output_dir / "models",
            use_wandb=True,
        )
        
        rprint("✅ [bold green]Pipeline initialized successfully![/bold green]")
    
    async def collect_training_data(
        self,
        num_conversations: int = 100,
        min_examples_per_action: int = 50,
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Collect training data from heuristic policy decisions.
        
        Args:
            num_conversations: Number of conversations to simulate
            min_examples_per_action: Minimum examples per action type
            save_path: Where to save the collected data
            
        Returns:
            Path to the saved training data
        """
        rprint("📊 [bold blue]Collecting training data...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Collecting training examples...", total=None)
            
            training_data, class_distribution = await self.data_collector.collect_training_data(
                num_conversations=num_conversations,
                min_examples_per_action=min_examples_per_action,
            )
            
            progress.update(task, completed=True)
        
        # Save data
        if save_path is None:
            save_path = self.output_dir / "collected_training_data.json"
        
        with open(save_path, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        # Display statistics
        self._display_data_statistics(training_data, class_distribution)
        
        self.training_data = training_data
        rprint(f"💾 [bold green]Training data saved to {save_path}[/bold green]")
        
        return save_path
    
    def _display_data_statistics(
        self,
        training_data: list,
        class_distribution: Dict[str, int]
    ) -> None:
        """Display training data statistics."""
        table = Table(title="Training Data Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Examples", str(len(training_data)))
        table.add_row("Classes", str(len(class_distribution)))
        
        for action, count in class_distribution.items():
            percentage = (count / len(training_data)) * 100
            table.add_row(f"  {action}", f"{count} ({percentage:.1f}%)")
        
        console.print(table)
    
    async def train_model(
        self,
        training_data_path: Path,
        experiment_name: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Train the ML model on collected data.
        
        Args:
            training_data_path: Path to training data JSON file
            experiment_name: Name for the training experiment
            model_config: Model configuration overrides
            training_config: Training configuration overrides
            
        Returns:
            Path to the trained model
        """
        rprint("🚀 [bold blue]Training ML model...[/bold blue]")
        
        # Load training data
        with open(training_data_path) as f:
            training_data = json.load(f)
        
        rprint(f"📖 Loaded {len(training_data)} training examples")
        
        # Update trainer configuration if provided
        if model_config:
            self.trainer.model_config.update(model_config)
        if training_config:
            self.trainer.training_config.update(training_config)
        
        # Train model
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training model...", total=None)
            
            training_results = await self.trainer.train(
                training_data=training_data,
                experiment_name=experiment_name,
            )
            
            progress.update(task, completed=True)
        
        training_time = time.time() - start_time
        
        # Display training results
        self._display_training_results(training_results, training_time)
        
        self.trained_model_path = Path(training_results["model_path"])
        rprint(f"🎯 [bold green]Model training completed![/bold green]")
        
        return self.trained_model_path
    
    def _display_training_results(
        self,
        results: Dict[str, Any],
        training_time: float
    ) -> None:
        """Display training results."""
        table = Table(title="Training Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Training Time", f"{training_time:.1f} seconds")
        table.add_row("Final Train Loss", f"{results['train_loss']:.4f}")
        
        eval_results = results.get("eval_results", {})
        if eval_results:
            table.add_row("Validation Accuracy", f"{eval_results.get('accuracy', 0):.3f}")
            table.add_row("Validation F1", f"{eval_results.get('f1', 0):.3f}")
        
        table.add_row("Model Path", str(results["model_path"]))
        
        console.print(table)
    
    async def evaluate_model(
        self,
        model_path: Path,
        test_conversations: int = 50,
        comparison_baseline: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate the trained ML model against baselines.
        
        Args:
            model_path: Path to the trained model
            test_conversations: Number of test conversations
            comparison_baseline: Whether to compare with heuristic baseline
            
        Returns:
            Evaluation results
        """
        rprint("📊 [bold blue]Evaluating trained model...[/bold blue]")
        
        # Initialize ML policy with trained model
        ml_policy = MLPolicy(model_path=model_path)
        await ml_policy.initialize()
        
        # Create dispatcher with ML policy
        ml_dispatcher = FrugalDispatcher(
            adapters=self.harness.adapters,
            cost_model=self.harness.cost_model,
        )
        
        # Register ML policy
        policy_registry = PolicyRegistry()
        policy_registry.register(ml_policy, set_as_default=True)
        ml_dispatcher.policy_registry = policy_registry
        
        # Run evaluation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running evaluation...", total=None)
            
            # Evaluate ML policy
            ml_results = await self._run_policy_evaluation(
                dispatcher=ml_dispatcher,
                policy_name="ml_enhanced",
                num_conversations=test_conversations,
            )
            
            # Compare with heuristic baseline if requested
            baseline_results = None
            if comparison_baseline:
                heuristic_dispatcher = FrugalDispatcher(
                    adapters=self.harness.adapters,
                    cost_model=self.harness.cost_model,
                )
                baseline_registry = PolicyRegistry()
                baseline_registry.register(self.heuristic_policy, set_as_default=True)
                heuristic_dispatcher.policy_registry = baseline_registry
                
                baseline_results = await self._run_policy_evaluation(
                    dispatcher=heuristic_dispatcher,
                    policy_name="heuristic_v1",
                    num_conversations=test_conversations,
                )
            
            progress.update(task, completed=True)
        
        # Compile evaluation results
        evaluation_results = {
            "ml_policy": ml_results,
            "baseline_policy": baseline_results,
            "comparison": self._compare_policies(ml_results, baseline_results) if baseline_results else None,
        }
        
        # Display results
        self._display_evaluation_results(evaluation_results)
        
        # Save results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        self.evaluation_results = evaluation_results
        rprint(f"📋 [bold green]Evaluation completed! Results saved to {results_path}[/bold green]")
        
        return evaluation_results
    
    async def _run_policy_evaluation(
        self,
        dispatcher: FrugalDispatcher,
        policy_name: str,
        num_conversations: int,
    ) -> Dict[str, Any]:
        """Run evaluation for a specific policy."""
        # Generate test conversations
        test_data = await self.data_collector._collect_examples(num_conversations)
        
        # Track metrics
        total_cost = 0.0
        total_decisions = 0
        decision_distribution = {}
        
        for example in test_data:
            # Simulate routing decision
            decision = await dispatcher.route_memory(
                item=example.get("item"),  # This would need proper conversion
                session_id=example.get("session_id", "test_session"),
                policy_name=policy_name,
            )
            
            total_decisions += 1
            if decision.estimated_cost:
                total_cost += decision.estimated_cost.total_cost
            
            action = decision.action.value
            decision_distribution[action] = decision_distribution.get(action, 0) + 1
        
        return {
            "policy_name": policy_name,
            "total_conversations": num_conversations,
            "total_decisions": total_decisions,
            "total_cost": total_cost,
            "avg_cost_per_decision": total_cost / total_decisions if total_decisions > 0 else 0,
            "decision_distribution": decision_distribution,
        }
    
    def _compare_policies(
        self,
        ml_results: Dict[str, Any],
        baseline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare ML policy with baseline."""
        ml_cost = ml_results["avg_cost_per_decision"]
        baseline_cost = baseline_results["avg_cost_per_decision"]
        
        cost_reduction = ((baseline_cost - ml_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        
        return {
            "cost_reduction_percent": cost_reduction,
            "ml_avg_cost": ml_cost,
            "baseline_avg_cost": baseline_cost,
            "ml_decisions": ml_results["total_decisions"],
            "baseline_decisions": baseline_results["total_decisions"],
        }
    
    def _display_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Display evaluation results."""
        # ML Policy Results
        ml_table = Table(title="ML Policy Evaluation")
        ml_table.add_column("Metric", style="cyan")
        ml_table.add_column("Value", style="magenta")
        
        ml_results = results["ml_policy"]
        ml_table.add_row("Policy", ml_results["policy_name"])
        ml_table.add_row("Total Decisions", str(ml_results["total_decisions"]))
        ml_table.add_row("Total Cost", f"${ml_results['total_cost']:.4f}")
        ml_table.add_row("Avg Cost/Decision", f"${ml_results['avg_cost_per_decision']:.6f}")
        
        console.print(ml_table)
        
        # Comparison results
        if results["comparison"]:
            comp_table = Table(title="Policy Comparison")
            comp_table.add_column("Metric", style="cyan")
            comp_table.add_column("Value", style="magenta")
            
            comparison = results["comparison"]
            comp_table.add_row("Cost Reduction", f"{comparison['cost_reduction_percent']:.1f}%")
            comp_table.add_row("ML Avg Cost", f"${comparison['ml_avg_cost']:.6f}")
            comp_table.add_row("Baseline Avg Cost", f"${comparison['baseline_avg_cost']:.6f}")
            
            console.print(comp_table)
    
    async def run_full_pipeline(
        self,
        num_conversations: int = 100,
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline: data collection → training → evaluation.
        
        Args:
            num_conversations: Number of conversations for data collection
            experiment_name: Name for the experiment
            
        Returns:
            Complete pipeline results
        """
        rprint("🎯 [bold blue]Running complete ML training pipeline...[/bold blue]")
        
        # Step 1: Initialize
        await self.initialize_components()
        
        # Step 2: Collect data
        data_path = await self.collect_training_data(num_conversations=num_conversations)
        
        # Step 3: Train model
        model_path = await self.train_model(
            training_data_path=data_path,
            experiment_name=experiment_name,
        )
        
        # Step 4: Evaluate model
        eval_results = await self.evaluate_model(
            model_path=model_path,
            test_conversations=max(20, num_conversations // 5),
        )
        
        # Compile final results
        pipeline_results = {
            "data_collection": {
                "data_path": str(data_path),
                "num_examples": len(self.training_data) if self.training_data else 0,
            },
            "model_training": {
                "model_path": str(model_path),
            },
            "evaluation": eval_results,
            "pipeline_completed": True,
        }
        
        # Save pipeline results
        results_path = self.output_dir / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        rprint(f"🎉 [bold green]Complete pipeline finished! Results: {results_path}[/bold green]")
        
        return pipeline_results


async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Train ML-enhanced memory routing policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ml_policy.py --full-pipeline --conversations 150
  python train_ml_policy.py --collect-data --conversations 200 --output ./my_training
  python train_ml_policy.py --train --data-path ./training_data.json --experiment my_experiment
  python train_ml_policy.py --evaluate --model-path ./ml_models/latest
        """
    )
    
    # Action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--collect-data", action="store_true",
                            help="Collect training data from heuristic policy")
    action_group.add_argument("--train", action="store_true",
                            help="Train ML model on collected data")
    action_group.add_argument("--evaluate", action="store_true",
                            help="Evaluate trained model")
    action_group.add_argument("--full-pipeline", action="store_true",
                            help="Run complete pipeline: collect + train + evaluate")
    
    # Configuration arguments
    parser.add_argument("--output", type=Path, default="./ml_training_output",
                       help="Output directory for training artifacts")
    parser.add_argument("--conversations", type=int, default=100,
                       help="Number of conversations for data collection")
    parser.add_argument("--data-path", type=Path,
                       help="Path to training data (for --train)")
    parser.add_argument("--model-path", type=Path,
                       help="Path to trained model (for --evaluate)")
    parser.add_argument("--experiment", type=str,
                       help="Experiment name for tracking")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Check ML availability
    if not ML_AVAILABLE:
        console.print("❌ [bold red]ML components not available. Please install required dependencies.[/bold red]")
        console.print("Run: poetry install --with optional")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline(output_dir=args.output)
    
    try:
        if args.full_pipeline:
            # Run complete pipeline
            await pipeline.run_full_pipeline(
                num_conversations=args.conversations,
                experiment_name=args.experiment,
            )
        
        elif args.collect_data:
            # Data collection only
            await pipeline.initialize_components()
            await pipeline.collect_training_data(num_conversations=args.conversations)
        
        elif args.train:
            # Training only
            if not args.data_path:
                console.print("❌ [bold red]--data-path required for training[/bold red]")
                sys.exit(1)
            
            await pipeline.initialize_components()
            
            # Update training config from CLI args
            training_config = {
                "num_epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
            }
            
            await pipeline.train_model(
                training_data_path=args.data_path,
                experiment_name=args.experiment,
                training_config=training_config,
            )
        
        elif args.evaluate:
            # Evaluation only
            if not args.model_path:
                console.print("❌ [bold red]--model-path required for evaluation[/bold red]")
                sys.exit(1)
            
            await pipeline.initialize_components()
            await pipeline.evaluate_model(model_path=args.model_path)
        
    except KeyboardInterrupt:
        console.print("\n⏹️ [bold yellow]Training interrupted by user[/bold yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ [bold red]Error: {e}[/bold red]")
        raise


if __name__ == "__main__":
    # Check if running in Jupyter/IPython
    try:
        import IPython
        IPython.get_ipython()
        # If we're in Jupyter, run with nest_asyncio
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())
    except (ImportError, AttributeError):
        # Normal execution
        asyncio.run(main())