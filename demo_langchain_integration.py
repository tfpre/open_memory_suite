#!/usr/bin/env python3
"""
LangChain Integration Demo

Demonstrates the FrugalMemory integration with LangChain, showing:
- Drop-in replacement for LangChain memory
- Cost optimization in conversational agents
- Comparison with standard LangChain memory
- Production usage patterns

Usage:
    python demo_langchain_integration.py
    python demo_langchain_integration.py --compare-memory-types
    python demo_langchain_integration.py --ml-enhanced
"""

import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Import LangChain components (with fallback)
try:
    from langchain.llms.base import LLM
    from langchain.agents import AgentType, initialize_agent, create_conversational_agent
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import LLMResult, Generation
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create fallback classes
    class LLM:
        pass
    class CallbackManagerForLLMRun:
        pass
    rprint("⚠️ LangChain not available. Install with: pip install langchain")

from open_memory_suite.integrations import FrugalMemory, LANGCHAIN_AVAILABLE as FRUGAL_LC_AVAILABLE

console = Console()

# Mock LLM for demonstration
class MockLLM(LLM):
    """Mock LLM for demonstration purposes."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.responses = [
            "That's an interesting question! Let me think about it...",
            "Based on our previous conversation, I can help you with that.",
            "I understand what you're looking for. Here's my response...",
            "Thanks for the clarification. Let me provide more details...",
            "That's a great follow-up question!",
        ]
        self.response_index = 0
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _call(
        self,
        prompt: str,
        stop: List[str] = None,
        run_manager: CallbackManagerForLLMRun = None,
        **kwargs: Any,
    ) -> str:
        """Generate a mock response."""
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return f"{response} (Responding to: {prompt[-50:]+'...' if len(prompt) > 50 else prompt})"


class LangChainIntegrationDemo:
    """Demo of LangChain integration with FrugalMemory."""
    
    def __init__(self):
        """Initialize the demo."""
        self.conversations = self._create_demo_conversations()
        
    def _create_demo_conversations(self) -> List[Dict[str, Any]]:
        """Create demo conversations for testing."""
        return [
            {
                "name": "Trip Planning Assistant",
                "description": "Multi-turn conversation about planning a vacation",
                "turns": [
                    ("I'm planning a trip to Japan for 2 weeks in spring. What should I know?", "I'd love to help you plan your Japan trip! Spring is an excellent time to visit..."),
                    ("What cities should I visit?", "For a 2-week spring trip, I recommend Tokyo, Kyoto, and Osaka as your main bases..."),
                    ("How much should I budget for this trip?", "Based on our discussion about your 2-week Japan trip in spring..."),
                    ("What about cherry blossom viewing?", "Since you're visiting in spring, cherry blossom season is perfect timing..."),
                    ("Should I book accommodations in advance?", "Given that you're planning to visit Tokyo, Kyoto, and Osaka during cherry blossom season..."),
                ]
            },
            {
                "name": "Technical Support Session",
                "description": "Troubleshooting a software issue with context",
                "turns": [
                    ("My Python application keeps crashing with a memory error", "I can help you troubleshoot this memory error. Let me gather some information..."),
                    ("It happens when I process large datasets", "Based on your previous message about the memory error, large datasets can indeed cause issues..."),
                    ("I'm using pandas to read CSV files", "Now I understand the context better. Pandas with large CSV files often causes memory issues..."),
                    ("What's the best way to optimize this?", "Considering your pandas CSV processing issue, here are several optimization strategies..."),
                ]
            },
            {
                "name": "Learning Session",
                "description": "Educational conversation with cumulative knowledge",
                "turns": [
                    ("I want to learn about machine learning", "Great! Machine learning is a fascinating field. Let's start with the basics..."),
                    ("What's the difference between supervised and unsupervised learning?", "Excellent question! Building on our ML introduction..."),
                    ("Can you give me examples of supervised learning?", "Of course! Remember we discussed supervised vs unsupervised learning..."),
                    ("What about neural networks?", "Neural networks are a key part of machine learning, which we've been discussing..."),
                ]
            },
        ]
    
    async def run_basic_demo(self) -> None:
        """Run basic FrugalMemory demonstration."""
        rprint("\n🚀 [bold blue]Basic FrugalMemory Demo[/bold blue]")
        
        if not LANGCHAIN_AVAILABLE or not FRUGAL_LC_AVAILABLE:
            rprint("❌ Required components not available")
            return
        
        # Create FrugalMemory instance
        memory = FrugalMemory(
            cost_budget=1.0,
            budget_type="standard",
            session_id="basic_demo"
        )
        
        # Wait for initialization
        await memory.initialize()
        
        # Simulate conversation
        conversation = self.conversations[0]
        rprint(f"\n📝 [bold cyan]{conversation['name']}[/bold cyan]")
        rprint(f"   {conversation['description']}")
        
        for i, (user_input, ai_response) in enumerate(conversation['turns']):
            rprint(f"\n[dim]Turn {i+1}:[/dim]")
            rprint(f"[blue]User:[/blue] {user_input}")
            rprint(f"[green]AI:[/green] {ai_response}")
            
            # Save to memory
            memory.save_context(
                {"input": user_input},
                {"output": ai_response}
            )
            
            # Show memory stats
            if i % 2 == 1:  # Every other turn
                stats = memory.get_memory_stats()
                rprint(f"[dim]Memory: {stats['message_count']} messages, Budget: ${stats['cost_budget']}[/dim]")
        
        # Show final stats
        final_stats = memory.get_memory_stats()
        self._display_memory_stats("FrugalMemory Results", final_stats)
        
        await memory.cleanup()
    
    async def run_comparison_demo(self) -> None:
        """Compare FrugalMemory with standard LangChain memory."""
        rprint("\n🆚 [bold blue]Memory Comparison Demo[/bold blue]")
        
        if not LANGCHAIN_AVAILABLE or not FRUGAL_LC_AVAILABLE:
            rprint("❌ Required components not available")
            return
        
        # Create both memory types
        frugal_memory = FrugalMemory(
            cost_budget=1.5,
            budget_type="standard",
            session_id="comparison_frugal"
        )
        
        standard_memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
        
        await frugal_memory.initialize()
        
        # Run same conversation through both
        conversation = self.conversations[1]  # Technical support
        rprint(f"\n📝 [bold cyan]{conversation['name']}[/bold cyan]")
        
        for i, (user_input, ai_response) in enumerate(conversation['turns']):
            # Save to both memories
            inputs = {"input": user_input}
            outputs = {"output": ai_response}
            
            frugal_memory.save_context(inputs, outputs)
            standard_memory.save_context(inputs, outputs)
        
        # Compare results
        frugal_stats = frugal_memory.get_memory_stats()
        standard_vars = standard_memory.load_memory_variables({})
        
        # Display comparison
        comparison_table = Table(title="Memory Comparison")
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column("FrugalMemory", style="green")
        comparison_table.add_column("Standard Memory", style="blue")
        
        comparison_table.add_row(
            "Messages Stored",
            str(frugal_stats['message_count']),
            str(len(standard_vars.get('history', [])))
        )
        comparison_table.add_row(
            "Cost Optimization",
            "✅ Enabled",
            "❌ None"
        )
        comparison_table.add_row(
            "Intelligent Routing",
            "✅ Yes",
            "❌ Simple Buffer"
        )
        comparison_table.add_row(
            "Budget Control",
            f"${frugal_stats['cost_budget']}",
            "❌ Unlimited"
        )
        comparison_table.add_row(
            "Adapters",
            ", ".join(frugal_stats['adapters']),
            "In-Memory Only"
        )
        
        console.print(comparison_table)
        
        await frugal_memory.cleanup()
    
    async def run_ml_enhanced_demo(self) -> None:
        """Demonstrate ML-enhanced memory routing."""
        rprint("\n🧠 [bold blue]ML-Enhanced Memory Demo[/bold blue]")
        
        # Check for ML model
        model_path = Path("./ml_models")
        available_models = list(model_path.glob("ml_policy_*")) if model_path.exists() else []
        
        if not available_models:
            rprint("⚠️ No trained ML models found. Demonstrating with heuristic policy.")
            enable_ml = False
            ml_model_path = None
        else:
            enable_ml = True
            ml_model_path = available_models[0]
            rprint(f"✅ Using ML model: {ml_model_path}")
        
        # Create ML-enhanced memory
        memory = FrugalMemory(
            cost_budget=2.0,
            budget_type="premium",
            enable_ml_policy=enable_ml,
            ml_model_path=ml_model_path,
            session_id="ml_demo"
        )
        
        await memory.initialize()
        
        # Run learning conversation
        conversation = self.conversations[2]  # Learning session
        rprint(f"\n📝 [bold cyan]{conversation['name']}[/bold cyan]")
        
        for i, (user_input, ai_response) in enumerate(conversation['turns']):
            rprint(f"\n[dim]Turn {i+1}:[/dim]")
            rprint(f"[blue]User:[/blue] {user_input[:80]}{'...' if len(user_input) > 80 else ''}")
            
            # Save to memory
            memory.save_context(
                {"input": user_input},
                {"output": ai_response}
            )
            
            # Show routing decision (if available)
            rprint(f"[dim]Memory routing: {'ML-enhanced' if enable_ml else 'Heuristic'} policy[/dim]")
        
        # Display final stats
        stats = memory.get_memory_stats()
        self._display_memory_stats("ML-Enhanced Memory Results", stats)
        
        await memory.cleanup()
    
    async def run_advanced_retrieval_demo(self) -> None:
        """Demonstrate advanced memory retrieval capabilities."""
        rprint("\n🔍 [bold blue]Advanced Memory Retrieval Demo[/bold blue]")
        
        memory = FrugalMemory(
            cost_budget=1.0,
            adapters=["memory", "file"],
            session_id="retrieval_demo"
        )
        
        await memory.initialize()
        
        # Build up some conversation history
        conversation_history = [
            ("What's the weather like today?", "I don't have access to real-time weather data."),
            ("I'm planning a picnic this weekend", "That sounds lovely! What kind of food are you planning?"),
            ("I was thinking sandwiches and fruit", "Great choices! Don't forget drinks and maybe some snacks."),
            ("Should I check the weather forecast?", "Definitely! You mentioned planning a picnic, so weather is important."),
        ]
        
        # Save conversation
        for user_msg, ai_msg in conversation_history:
            memory.save_context({"input": user_msg}, {"output": ai_msg})
        
        # Demonstrate retrieval
        queries = [
            "picnic planning",
            "weather information",
            "food suggestions"
        ]
        
        rprint("\n🔍 [bold yellow]Memory Retrieval Results[/bold yellow]")
        
        for query in queries:
            rprint(f"\n[cyan]Query:[/cyan] {query}")
            memories = await memory.retrieve_relevant_memories(query, max_memories=3)
            
            if memories:
                for i, mem in enumerate(memories, 1):
                    content_preview = mem['content'][:60] + "..." if len(mem['content']) > 60 else mem['content']
                    rprint(f"  {i}. [{mem['speaker']}] {content_preview}")
            else:
                rprint("  No relevant memories found")
        
        await memory.cleanup()
    
    def _display_memory_stats(self, title: str, stats: Dict[str, Any]) -> None:
        """Display memory statistics in a table."""
        
        stats_table = Table(title=title)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")
        
        stats_table.add_row("Session ID", stats.get('session_id', 'N/A'))
        stats_table.add_row("Message Count", str(stats.get('message_count', 0)))
        stats_table.add_row("Cost Budget", f"${stats.get('cost_budget', 0)}")
        stats_table.add_row("Budget Type", stats.get('budget_type', 'N/A'))
        stats_table.add_row("Adapters", ", ".join(stats.get('adapters', [])))
        stats_table.add_row("Initialized", "✅" if stats.get('initialized') else "❌")
        
        console.print(stats_table)
    
    async def run_production_example(self) -> None:
        """Show production usage example."""
        rprint("\n🏭 [bold blue]Production Usage Example[/bold blue]")
        
        code_example = """
# Production LangChain Integration Example

from langchain.llms import OpenAI
from langchain.agents import create_conversational_agent
from open_memory_suite.integrations import FrugalMemory

# Create cost-optimized memory
memory = FrugalMemory(
    cost_budget=5.0,
    budget_type="premium",
    enable_ml_policy=True,
    ml_model_path="./models/production_model.pt"
)

# Initialize memory (important!)
await memory.initialize()

# Use with LangChain agent
llm = OpenAI(temperature=0.7)
agent = create_conversational_agent(
    llm=llm,
    tools=your_tools,
    memory=memory,  # Drop-in replacement!
    verbose=True
)

# Run conversations with automatic cost optimization
result = agent.run("Help me plan a complex project")

# Monitor memory usage
stats = memory.get_memory_stats()
print(f"Memory usage: {stats['message_count']} messages, "
      f"Cost: ${stats.get('total_cost', 0):.4f}")

# Clean up
await memory.cleanup()
"""
        
        console.print(Panel(
            code_example.strip(),
            title="Production Usage Example",
            border_style="green",
            expand=False
        ))


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="LangChain Integration Demo")
    parser.add_argument("--compare-memory-types", action="store_true",
                       help="Compare FrugalMemory with standard LangChain memory")
    parser.add_argument("--ml-enhanced", action="store_true",
                       help="Demonstrate ML-enhanced memory routing")
    parser.add_argument("--advanced-retrieval", action="store_true",
                       help="Show advanced memory retrieval capabilities")
    parser.add_argument("--production-example", action="store_true",
                       help="Show production usage example")
    parser.add_argument("--all", action="store_true",
                       help="Run all demonstrations")
    
    args = parser.parse_args()
    
    if not LANGCHAIN_AVAILABLE:
        console.print("❌ [bold red]LangChain not available[/bold red]")
        console.print("Install with: pip install langchain")
        return
    
    if not FRUGAL_LC_AVAILABLE:
        console.print("❌ [bold red]FrugalMemory integration not available[/bold red]")
        return
    
    demo = LangChainIntegrationDemo()
    
    try:
        rprint("🎯 [bold blue]LangChain Integration Demo[/bold blue]")
        rprint("Demonstrating cost-optimized memory for LangChain applications\n")
        
        if args.all or not any([args.compare_memory_types, args.ml_enhanced, 
                               args.advanced_retrieval, args.production_example]):
            # Default: run basic demo
            await demo.run_basic_demo()
        
        if args.compare_memory_types or args.all:
            await demo.run_comparison_demo()
        
        if args.ml_enhanced or args.all:
            await demo.run_ml_enhanced_demo()
        
        if args.advanced_retrieval or args.all:
            await demo.run_advanced_retrieval_demo()
        
        if args.production_example or args.all:
            await demo.run_production_example()
        
        rprint("\n✅ [bold green]Demo completed successfully![/bold green]")
        
    except Exception as e:
        rprint(f"❌ [bold red]Demo failed: {e}[/bold red]")
        raise

if __name__ == "__main__":
    asyncio.run(main())