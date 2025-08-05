#!/usr/bin/env python3
"""
ML-Enhanced FrugalDispatcher Demo

This demo showcases the M3 milestone: ML-enhanced intelligent memory routing
that learns better decisions than rule-based heuristics through fine-tuned
DistilBERT + LoRA models.

Features demonstrated:
- ML policy vs. heuristic policy comparison
- Real-time decision explanation and confidence scoring
- Cost-aware routing with budget constraints
- Performance monitoring and statistics
- Graceful fallback when ML confidence is low

Usage:
    python demo_ml_enhanced_dispatcher.py
    python demo_ml_enhanced_dispatcher.py --train-model
    python demo_ml_enhanced_dispatcher.py --compare-policies
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

from open_memory_suite.adapters import MemoryItem, InMemoryAdapter, FileStoreAdapter
from open_memory_suite.benchmark import CostModel, BudgetType
from open_memory_suite.dispatcher import (
    FrugalDispatcher,
    HeuristicPolicy,
    PolicyRegistry,
    ConversationContext,
    MemoryAction,
    ML_AVAILABLE,
)

if ML_AVAILABLE:
    from open_memory_suite.dispatcher import MLPolicy, MLTrainer, DataCollector

console = Console()


class MLEnhancedDemo:
    """Interactive demo of ML-enhanced memory routing."""
    
    def __init__(self):
        """Initialize the demo."""
        self.adapters = []
        self.cost_model = None
        self.heuristic_policy = None
        self.ml_policy = None
        self.demo_conversations = []
        
    async def initialize(self) -> None:
        """Initialize demo components."""
        rprint("🔧 [bold blue]Initializing ML-Enhanced Dispatcher Demo...[/bold blue]")
        
        # Create adapters
        self.adapters = [
            InMemoryAdapter("memory_store"),
            FileStoreAdapter("file_store", Path("./demo_storage")),
        ]
        
        # Initialize adapters
        for adapter in self.adapters:
            await adapter.initialize()
        
        # Create cost model
        self.cost_model = CostModel()
        
        # Create policies
        self.heuristic_policy = HeuristicPolicy()
        
        if ML_AVAILABLE:
            # Check if trained model exists
            model_path = Path("./ml_models")
            if any(model_path.glob("ml_policy_*")):
                latest_model = max(model_path.glob("ml_policy_*"), key=lambda p: p.stat().st_mtime)
                self.ml_policy = MLPolicy(
                    model_path=latest_model,
                    confidence_threshold=0.7,
                    fallback_to_heuristic=True,
                )
                await self.ml_policy.initialize()
                rprint(f"✅ Loaded ML policy from {latest_model}")
            else:
                rprint("⚠️ No trained ML model found. Use --train-model to create one.")
        
        # Create demo conversations
        self._create_demo_conversations()
        
        rprint("✅ [bold green]Demo initialization complete![/bold green]")
    
    def _create_demo_conversations(self) -> None:
        """Create diverse demo conversations."""
        self.demo_conversations = [
            {
                "name": "Personal Information Sharing",
                "budget": BudgetType.STANDARD,
                "turns": [
                    ("user", "Hi! My name is Sarah Chen and I'm a software engineer."),
                    ("assistant", "Nice to meet you, Sarah! What kind of software do you work on?"),
                    ("user", "I work on machine learning systems at TechCorp downtown."),
                    ("assistant", "That sounds fascinating! What specific ML areas interest you?"),
                    ("user", "I'm particularly interested in NLP and computer vision."),
                    ("assistant", "Those are exciting fields! Any current projects you're working on?"),
                    ("user", "Yes, we're building a document analysis system using transformers."),
                    ("assistant", "Transformers are powerful for document understanding. What challenges are you facing?"),
                    ("user", "Thanks for the insights!"),
                    ("assistant", "You're welcome! Feel free to ask if you have more questions."),
                ]
            },
            {
                "name": "Technical Discussion",
                "budget": BudgetType.PREMIUM,
                "turns": [
                    ("user", "Can you explain how attention mechanisms work in transformers?"),
                    ("assistant", "Attention mechanisms allow models to focus on different parts of the input sequence when processing each token. The key innovation is the self-attention mechanism where each position can attend to all positions in the previous layer."),
                    ("user", "What's the difference between self-attention and cross-attention?"),
                    ("assistant", "Self-attention computes relationships within the same sequence, while cross-attention computes relationships between different sequences, like in encoder-decoder architectures."),
                    ("user", "How does the multi-head attention work?"),
                    ("assistant", "Multi-head attention runs multiple attention functions in parallel, each with different learned projections. This allows the model to attend to information from different representation subspaces."),
                    ("user", "That makes sense, thank you!"),
                    ("assistant", "You're welcome! Transformers are a rich topic with lots of interesting details."),
                ]
            },
            {
                "name": "Budget-Conscious Chat",
                "budget": BudgetType.MINIMAL,
                "turns": [
                    ("user", "Hello there!"),
                    ("assistant", "Hi! How can I help you today?"),
                    ("user", "I'm looking for information about machine learning."),
                    ("assistant", "I'd be happy to help! What specific aspect of ML interests you?"),
                    ("user", "Just general concepts for now."),
                    ("assistant", "Machine learning is about training algorithms to find patterns in data and make predictions or decisions without being explicitly programmed for each task."),
                    ("user", "Got it, thanks."),
                    ("assistant", "You're welcome! Let me know if you need more details."),
                    ("user", "Will do."),
                    ("assistant", "Great! I'm here when you need me."),
                ]
            },
        ]
    
    async def run_conversation_comparison(self) -> None:
        """Run conversations comparing heuristic vs ML policies."""
        if not ML_AVAILABLE or not self.ml_policy:
            rprint("❌ ML policy not available for comparison")
            return
        
        rprint("\n🆚 [bold blue]Policy Comparison Demo[/bold blue]")
        
        for conversation in self.demo_conversations:
            await self._run_single_conversation_comparison(conversation)
    
    async def _run_single_conversation_comparison(self, conversation: Dict[str, Any]) -> None:
        """Run a single conversation with both policies."""
        conv_name = conversation["name"]
        budget = conversation["budget"]
        turns = conversation["turns"]
        
        rprint(f"\n📝 [bold cyan]{conv_name}[/bold cyan] (Budget: {budget.value})")
        
        # Create sessions for both policies
        session_id_heuristic = f"heuristic_{conv_name.replace(' ', '_').lower()}"
        session_id_ml = f"ml_{conv_name.replace(' ', '_').lower()}"
        
        # Create dispatchers
        heuristic_dispatcher = await self._create_dispatcher(self.heuristic_policy, "heuristic")
        ml_dispatcher = await self._create_dispatcher(self.ml_policy, "ml_enhanced")
        
        # Track results
        heuristic_results = {"total_cost": 0.0, "decisions": []}
        ml_results = {"total_cost": 0.0, "decisions": []}
        
        # Process each turn
        for turn_idx, (speaker, content) in enumerate(turns):
            item = MemoryItem(
                content=content,
                speaker=speaker,
                session_id=f"demo_{conv_name}",
                metadata={"turn": turn_idx},
            )
            
            # Get decisions from both policies
            heuristic_decision = await heuristic_dispatcher.route_memory(
                item, session_id_heuristic
            )
            ml_decision = await ml_dispatcher.route_memory(
                item, session_id_ml
            )
            
            # Execute decisions
            await heuristic_dispatcher.execute_decision(heuristic_decision, item, session_id_heuristic)
            await ml_dispatcher.execute_decision(ml_decision, item, session_id_ml)
            
            # Track costs
            if heuristic_decision.estimated_cost:
                heuristic_results["total_cost"] += heuristic_decision.estimated_cost.total_cost
            if ml_decision.estimated_cost:
                ml_results["total_cost"] += ml_decision.estimated_cost.total_cost
            
            # Store decisions for analysis
            heuristic_results["decisions"].append({
                "turn": turn_idx,
                "content": content[:50] + "..." if len(content) > 50 else content,
                "action": heuristic_decision.action.value,
                "adapter": heuristic_decision.selected_adapter,
                "reasoning": heuristic_decision.reasoning[:80] + "..." if len(heuristic_decision.reasoning) > 80 else heuristic_decision.reasoning,
            })
            
            ml_results["decisions"].append({
                "turn": turn_idx,
                "content": content[:50] + "..." if len(content) > 50 else content,
                "action": ml_decision.action.value,
                "adapter": ml_decision.selected_adapter,
                "reasoning": ml_decision.reasoning[:80] + "..." if len(ml_decision.reasoning) > 80 else ml_decision.reasoning,
                "confidence": getattr(ml_decision, 'confidence', 0.0),
            })
        
        # Display comparison
        self._display_conversation_results(conv_name, heuristic_results, ml_results)
    
    async def _create_dispatcher(self, policy, policy_name: str) -> FrugalDispatcher:
        """Create a dispatcher with the given policy."""
        dispatcher = FrugalDispatcher(
            adapters=self.adapters,
            cost_model=self.cost_model,
        )
        
        registry = PolicyRegistry()
        registry.register(policy, set_as_default=True)
        dispatcher.policy_registry = registry
        
        await dispatcher.initialize()
        return dispatcher
    
    def _display_conversation_results(
        self,
        conversation_name: str,
        heuristic_results: Dict[str, Any],
        ml_results: Dict[str, Any],
    ) -> None:
        """Display comparison results for a conversation."""
        
        # Cost comparison
        heuristic_cost = heuristic_results["total_cost"]
        ml_cost = ml_results["total_cost"]
        cost_reduction = ((heuristic_cost - ml_cost) / heuristic_cost * 100) if heuristic_cost > 0 else 0
        
        cost_table = Table(title=f"Cost Comparison - {conversation_name}")
        cost_table.add_column("Policy", style="cyan")
        cost_table.add_column("Total Cost", style="magenta")
        cost_table.add_column("Avg Cost/Turn", style="yellow")
        
        num_turns = len(heuristic_results["decisions"])
        cost_table.add_row(
            "Heuristic",
            f"${heuristic_cost:.6f}",
            f"${heuristic_cost/num_turns:.6f}" if num_turns > 0 else "$0.000000"
        )
        cost_table.add_row(
            "ML Enhanced",
            f"${ml_cost:.6f}",
            f"${ml_cost/num_turns:.6f}" if num_turns > 0 else "$0.000000"
        )
        cost_table.add_row(
            "Savings",
            f"{cost_reduction:+.1f}%",
            f"${(heuristic_cost-ml_cost)/num_turns:.6f}" if num_turns > 0 else "$0.000000"
        )
        
        console.print(cost_table)
        
        # Decision comparison for key turns
        decision_table = Table(title="Key Decision Differences")
        decision_table.add_column("Turn", style="dim")
        decision_table.add_column("Content", style="white")
        decision_table.add_column("Heuristic", style="blue")
        decision_table.add_column("ML Enhanced", style="green")
        decision_table.add_column("ML Confidence", style="yellow")
        
        for i, (h_decision, ml_decision) in enumerate(zip(
            heuristic_results["decisions"][:5],  # Show first 5 turns
            ml_results["decisions"][:5]
        )):
            if h_decision["action"] != ml_decision["action"]:
                decision_table.add_row(
                    str(h_decision["turn"]),
                    h_decision["content"],
                    h_decision["action"],
                    ml_decision["action"],
                    f"{ml_decision.get('confidence', 0.0):.2f}"
                )
        
        if decision_table.rows:
            console.print(decision_table)
        else:
            console.print("✅ [green]Policies made identical decisions for all turns[/green]")
    
    async def demonstrate_ml_features(self) -> None:
        """Demonstrate specific ML policy features."""
        if not ML_AVAILABLE or not self.ml_policy:
            rprint("❌ ML policy not available")
            return
        
        rprint("\n🧠 [bold blue]ML Policy Features Demo[/bold blue]")
        
        # Feature 1: Confidence-based decisions
        await self._demo_confidence_based_routing()
        
        # Feature 2: Context-aware reasoning
        await self._demo_context_aware_reasoning()
        
        # Feature 3: Performance statistics
        await self._demo_performance_statistics()
    
    async def _demo_confidence_based_routing(self) -> None:
        """Demo confidence-based routing with fallback."""
        rprint("\n🎯 [bold yellow]Confidence-Based Routing[/bold yellow]")
        
        dispatcher = await self._create_dispatcher(self.ml_policy, "ml_enhanced")
        
        test_cases = [
            ("What is machine learning?", "Clear technical question"),
            ("asdf jkl;", "Unclear/noisy input"),
            ("ok", "Simple acknowledgment"),
            ("I was born on January 15, 1990 in San Francisco", "Factual information"),
        ]
        
        confidence_table = Table()
        confidence_table.add_column("Input", style="white")
        confidence_table.add_column("Action", style="green")
        confidence_table.add_column("Confidence", style="yellow")
        confidence_table.add_column("Source", style="cyan")
        
        for content, description in test_cases:
            item = MemoryItem(
                content=content,
                speaker="user",
                session_id="confidence_demo",
            )
            
            decision = await dispatcher.route_memory(item, "confidence_demo")
            
            # Determine if ML or heuristic was used
            source = "ML" if hasattr(decision, 'confidence') and decision.confidence >= 0.7 else "Heuristic"
            confidence = getattr(decision, 'confidence', 0.0)
            
            confidence_table.add_row(
                content[:40] + "..." if len(content) > 40 else content,
                decision.action.value,
                f"{confidence:.3f}",
                source
            )
        
        console.print(confidence_table)
    
    async def _demo_context_aware_reasoning(self) -> None:
        """Demo context-aware reasoning capabilities."""
        rprint("\n🧩 [bold yellow]Context-Aware Reasoning[/bold yellow]")
        
        dispatcher = await self._create_dispatcher(self.ml_policy, "ml_enhanced")
        
        # Build up context gradually
        context_demo = [
            ("user", "Hi, I'm working on a project"),
            ("assistant", "I'd be happy to help! What kind of project?"),
            ("user", "It's a machine learning project for my company TechCorp"),
            ("assistant", "Interesting! What specific ML problem are you solving?"),
            ("user", "We need to classify customer feedback into categories"),
            ("assistant", "Text classification is a great use case. What approach are you considering?"),
            ("user", "thanks"),  # This should be dropped due to context
        ]
        
        context_table = Table(title="Context Evolution")
        context_table.add_column("Turn", style="dim")
        context_table.add_column("Speaker", style="cyan")
        context_table.add_column("Content", style="white")
        context_table.add_column("Action", style="green")
        context_table.add_column("Reasoning", style="yellow")
        
        for turn, (speaker, content) in enumerate(context_demo):
            item = MemoryItem(
                content=content,
                speaker=speaker,
                session_id="context_demo",
            )
            
            decision = await dispatcher.route_memory(item, "context_demo")
            
            context_table.add_row(
                str(turn + 1),
                speaker,
                content[:30] + "..." if len(content) > 30 else content,
                decision.action.value,
                decision.reasoning[:50] + "..." if len(decision.reasoning) > 50 else decision.reasoning
            )
        
        console.print(context_table)
    
    async def _demo_performance_statistics(self) -> None:
        """Demo performance monitoring and statistics."""
        rprint("\n📊 [bold yellow]Performance Statistics[/bold yellow]")
        
        # Get stats from ML policy
        ml_stats = self.ml_policy.get_stats()
        
        stats_table = Table(title="ML Policy Performance")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")
        
        stats_table.add_row("Policy Name", ml_stats["policy_name"])
        stats_table.add_row("Policy Version", ml_stats["policy_version"])
        stats_table.add_row("Total Decisions", str(ml_stats["decisions_made"]))
        stats_table.add_row("ML Decisions", str(ml_stats["ml_decisions"]))
        stats_table.add_row("Heuristic Fallbacks", str(ml_stats["heuristic_fallbacks"]))
        stats_table.add_row("ML Decision Rate", f"{ml_stats['ml_decision_rate']:.1%}")
        stats_table.add_row("Avg Inference Time", f"{ml_stats['avg_inference_time_ms']:.1f}ms")
        stats_table.add_row("Device", ml_stats["device"])
        stats_table.add_row("Model Loaded", "✅" if ml_stats["model_loaded"] else "❌")
        
        console.print(stats_table)
    
    async def cleanup(self) -> None:
        """Clean up demo resources."""
        for adapter in self.adapters:
            await adapter.cleanup()


async def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML-Enhanced Dispatcher Demo")
    parser.add_argument("--train-model", action="store_true",
                       help="Train a new ML model for demonstration")
    parser.add_argument("--compare-policies", action="store_true",
                       help="Run policy comparison demo")
    parser.add_argument("--ml-features", action="store_true",
                       help="Demonstrate ML-specific features")
    parser.add_argument("--full-demo", action="store_true",
                       help="Run complete demonstration")
    
    args = parser.parse_args()
    
    if not ML_AVAILABLE:
        console.print("❌ [bold red]ML components not available[/bold red]")
        console.print("Install with: poetry install")
        return
    
    # Initialize demo
    demo = MLEnhancedDemo()
    
    try:
        if args.train_model:
            # Train a model for demonstration
            rprint("🚀 [bold blue]Training ML model for demo...[/bold blue]")
            
            from open_memory_suite.dispatcher import MLTrainer, DataCollector
            from open_memory_suite.benchmark import BenchmarkHarness
            
            # Quick training pipeline
            trainer = MLTrainer(
                output_dir=Path("./ml_models"),
                use_wandb=False,
                training_config={
                    "num_epochs": 2,
                    "batch_size": 8,
                    "learning_rate": 2e-5,
                }
            )
            
            # Generate minimal training data
            training_data = []
            for i in range(50):
                training_data.append({
                    "text": f"Turn {i}, Budget: standard | Current: [user]: Example message {i}",
                    "label": i % 5,
                    "action": ["store", "store_faiss", "store_file", "summarize", "drop"][i % 5],
                    "context_features": [float(i), 20.0, 4.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0],
                })
            
            await trainer.train(training_data, experiment_name="demo_model")
            rprint("✅ [bold green]Demo model trained![/bold green]")
        
        # Initialize demo components
        await demo.initialize()
        
        if args.compare_policies or args.full_demo:
            await demo.run_conversation_comparison()
        
        if args.ml_features or args.full_demo:
            await demo.demonstrate_ml_features()
        
        if not any([args.train_model, args.compare_policies, args.ml_features, args.full_demo]):
            # Default: run basic demo
            rprint("\n🎯 [bold blue]Running ML-Enhanced Dispatcher Demo[/bold blue]")
            await demo.run_conversation_comparison()
            await demo.demonstrate_ml_features()
        
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())