#!/usr/bin/env python3
"""
FrugalDispatcher Comprehensive Demo

Showcases the intelligent cost-aware memory routing system with:
- Rule-based content analysis and routing decisions
- Cost-aware adapter selection with budget constraints
- Real-time decision explanation and tracing
- Multi-session conversation management
- Performance monitoring and statistics

This demonstrates the core value proposition: intelligent memory management
that reduces costs while maintaining high recall through smart routing.
"""

import asyncio
from pathlib import Path
from typing import List

from open_memory_suite.adapters.base import MemoryItem
from open_memory_suite.adapters.memory_store import InMemoryAdapter
from open_memory_suite.adapters.file_store import FileStoreAdapter
from open_memory_suite.benchmark.cost_model import BudgetType, CostModel
from open_memory_suite.dispatcher import (
    FrugalDispatcher,
    HeuristicPolicy,
    PolicyRegistry,
    MemoryAction,
    Priority,
)


class FrugalDispatcherDemo:
    """Interactive demo of the FrugalDispatcher system."""
    
    def __init__(self):
        self.dispatcher = None
        self.session_scenarios = {
            "budget_conscious": {
                "budget": BudgetType.MINIMAL,
                "description": "Ultra-frugal mode - maximum cost savings"
            },
            "balanced": {
                "budget": BudgetType.STANDARD,
                "description": "Balanced mode - quality/cost trade-off"
            },
            "premium": {
                "budget": BudgetType.PREMIUM,
                "description": "Premium mode - prioritize quality"
            }
        }
    
    async def setup_system(self):
        """Initialize the FrugalDispatcher with adapters and policies."""
        print("🔧 Setting up FrugalDispatcher system...")
        
        # Create adapters with different cost/quality profiles
        adapters = [
            InMemoryAdapter("memory_store"),  # Fast, free, ephemeral
            FileStoreAdapter("file_store", storage_path=Path("./demo_storage"))  # Cheap, persistent, slow
        ]
        
        # Initialize adapters
        for adapter in adapters:
            await adapter.initialize()
        
        # Create cost model and policy registry
        cost_model = CostModel()
        policy_registry = PolicyRegistry()
        
        # Register heuristic policy
        heuristic_policy = HeuristicPolicy("demo_heuristic", "1.0")
        policy_registry.register(heuristic_policy, set_as_default=True)
        
        # Create dispatcher
        self.dispatcher = FrugalDispatcher(
            adapters=adapters,
            cost_model=cost_model,
            policy_registry=policy_registry,
            default_budget=BudgetType.STANDARD
        )
        
        await self.dispatcher.initialize()
        print("✅ System initialized successfully!")
        
        return adapters
    
    def create_conversation_scenarios(self) -> dict:
        """Create diverse conversation scenarios for testing."""
        return {
            "personal_info": [
                MemoryItem(content="Hi, my name is Sarah Chen", speaker="user"),
                MemoryItem(content="I'm 28 years old and work as a data scientist", speaker="user"),
                MemoryItem(content="My phone number is 555-0199", speaker="user"),
                MemoryItem(content="I live in Seattle, Washington", speaker="user"),
            ],
            
            "technical_questions": [
                MemoryItem(content="What is the difference between supervised and unsupervised learning?", speaker="user"),
                MemoryItem(content="How do I implement a convolutional neural network?", speaker="user"),
                MemoryItem(content="Can you explain gradient descent?", speaker="user"),
            ],
            
            "conversational_flow": [
                MemoryItem(content="That's really helpful, thank you!", speaker="user"),
                MemoryItem(content="ok", speaker="user"),
                MemoryItem(content="I see what you mean", speaker="user"),
                MemoryItem(content="got it", speaker="assistant"),
                MemoryItem(content="Perfect!", speaker="user"),
            ],
            
            "long_content": [
                MemoryItem(content=" ".join([
                    "I'm working on a machine learning project that involves analyzing customer behavior data.",
                    "The dataset contains information about purchase history, browsing patterns, and demographic data.",
                    "I need to build a recommendation system that can predict what products customers might be interested in.",
                    "The main challenges are handling sparse data, dealing with cold start problems for new customers,",
                    "and ensuring the recommendations are both accurate and diverse.",
                    "I'm considering using collaborative filtering, content-based filtering, or a hybrid approach.",
                    "The system needs to scale to millions of users and provide real-time recommendations.",
                ]), speaker="user"),
            ],
            
            "mixed_content": [
                MemoryItem(content="What's the weather like today?", speaker="user"),
                MemoryItem(content="My favorite programming language is Python", speaker="user"),
                MemoryItem(content="thanks", speaker="user"),
                MemoryItem(content="I graduated from MIT in 2018", speaker="user"),
                MemoryItem(content="How do I debug memory leaks?", speaker="user"),
                MemoryItem(content="ok got it", speaker="assistant"),
            ]
        }
    
    async def demonstrate_scenario(self, scenario_name: str, items: List[MemoryItem], budget_type: BudgetType):
        """Demonstrate routing decisions for a specific scenario."""
        print(f"\n{'='*20} {scenario_name.upper()} SCENARIO {'='*20}")
        print(f"Budget Type: {budget_type.value.upper()}")
        print(f"Items to process: {len(items)}")
        
        session_id = f"demo_{scenario_name}_{budget_type.value}"
        context = await self.dispatcher.get_or_create_context(
            session_id=session_id,
            budget_type=budget_type
        )
        
        total_cost = 0.0
        decisions_summary = {"store": 0, "drop": 0, "summarize": 0}
        
        for i, item in enumerate(items, 1):
            decision, success = await self.dispatcher.route_and_execute(item, session_id)
            
            print(f"\n📝 Item {i}: \"{item.content[:60]}{'...' if len(item.content) > 60 else ''}\"")
            print(f"   Speaker: {item.speaker}")
            print(f"   🧠 Decision: {decision.action.value.upper()}")
            
            if decision.selected_adapter:
                print(f"   📦 Adapter: {decision.selected_adapter}")
            
            if decision.estimated_cost:
                cost = decision.estimated_cost.total_cost
                total_cost += cost
                print(f"   💰 Cost: ${cost:.6f}")
            
            print(f"   🎯 Priority: {decision.detected_priority.value}")
            print(f"   📊 Confidence: {decision.confidence:.2f}")
            print(f"   💭 Reasoning: {decision.reasoning}")
            print(f"   ✅ Executed: {'Yes' if success else 'No'}")
            
            decisions_summary[decision.action.value] += 1
        
        print(f"\n📈 SCENARIO SUMMARY:")
        print(f"   Total Cost: ${total_cost:.6f}")
        print(f"   Decisions: {decisions_summary}")
        
        # Show context state
        session_summary = await self.dispatcher.get_session_summary(session_id)
        if session_summary:
            print(f"   Session Duration: {session_summary['session_duration_minutes']:.1f} min")
            print(f"   Total Stored Items: {session_summary['total_stored_items']}")
    
    async def demonstrate_retrieval(self):
        """Demonstrate memory retrieval across different scenarios."""
        print(f"\n{'='*20} MEMORY RETRIEVAL DEMO {'='*20}")
        
        # Test queries that should find relevant information
        test_queries = [
            "What is the user's name?",
            "Tell me about machine learning questions",
            "What programming language does the user prefer?",
            "What is the user's educational background?",
            "How can I implement neural networks?",
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: \"{query}\"")
            
            # Try retrieval from different sessions
            for session_type in ["demo_personal_info_standard", "demo_technical_questions_standard", "demo_mixed_content_standard"]:
                try:
                    result = await self.dispatcher.retrieve_memories(
                        query=query,
                        session_id=session_type,
                        k=3
                    )
                    
                    if result.items:
                        print(f"   📚 Found {len(result.items)} items from {session_type}:")
                        for item in result.items[:2]:  # Show top 2
                            content = item.content[:80] + "..." if len(item.content) > 80 else item.content
                            print(f"      • \"{content}\" ({item.speaker})")
                    else:
                        print(f"   ❌ No relevant items found in {session_type}")
                        
                except Exception as e:
                    print(f"   ⚠️  Error retrieving from {session_type}: {e}")
    
    async def demonstrate_performance_monitoring(self):
        """Show performance statistics and monitoring capabilities."""
        print(f"\n{'='*20} PERFORMANCE MONITORING {'='*20}")
        
        stats = self.dispatcher.get_stats()
        
        print("📊 DISPATCHER STATISTICS:")
        dispatcher_stats = stats["dispatcher_stats"]
        print(f"   Total Routing Decisions: {dispatcher_stats['total_routing_decisions']}")
        print(f"   Items Stored: {dispatcher_stats['items_stored']}")
        print(f"   Items Dropped: {dispatcher_stats['items_dropped']}")
        print(f"   Items Summarized: {dispatcher_stats['items_summarized']}")
        print(f"   Average Decision Time: {dispatcher_stats['avg_decision_time_ms']:.2f}ms")
        print(f"   Active Sessions: {dispatcher_stats['active_sessions']}")
        
        print(f"\n🏥 ADAPTER HEALTH:")
        adapter_health = stats["adapter_health"]
        for adapter_name, is_healthy in adapter_health.items():
            status = "✅ Healthy" if is_healthy else "❌ Unhealthy"
            print(f"   {adapter_name}: {status}")
        
        print(f"\n📈 POLICY PERFORMANCE:")
        policy_stats = stats["policy_stats"]
        for policy_name, policy_data in policy_stats.items():
            print(f"   Policy: {policy_name}")
            print(f"   - Decisions Made: {policy_data['decisions_made']}")
            print(f"   - Items Stored: {policy_data['items_stored']}")
            print(f"   - Items Dropped: {policy_data['items_dropped']}")
            print(f"   - Cost Saved: ${policy_data['total_cost_saved']:.6f}")
    
    async def demonstrate_budget_constraints(self):
        """Show how different budget types affect routing decisions."""
        print(f"\n{'='*20} BUDGET CONSTRAINT DEMO {'='*20}")
        
        # Same content, different budgets
        test_content = [
            MemoryItem(content="My name is Alex Rodriguez", speaker="user"),
            MemoryItem(content="What is deep learning?", speaker="user"),  
            MemoryItem(content="I work at Microsoft", speaker="user"),
            MemoryItem(content="thanks for the help", speaker="user"),
            MemoryItem(content="How do I optimize neural networks?", speaker="user"),
        ]
        
        for budget_name, budget_info in self.session_scenarios.items():
            print(f"\n💰 Budget Type: {budget_name.upper()}")
            print(f"   Description: {budget_info['description']}")
            
            session_id = f"budget_demo_{budget_name}"
            decisions = []
            
            for item in test_content:
                decision = await self.dispatcher.route_memory(
                    item, 
                    session_id, 
                    policy_name="demo_heuristic"
                )
                decisions.append(decision.action.value)
            
            print(f"   Routing Pattern: {' → '.join(decisions)}")
            
            # Show how budget affects adapter selection
            context = await self.dispatcher.get_or_create_context(session_id, budget_info["budget"])
            print(f"   Budget Critical: {context.is_budget_critical()}")
    
    async def run_comprehensive_demo(self):
        """Run the complete FrugalDispatcher demonstration."""
        print("🎭 FrugalDispatcher Comprehensive Demo")
        print("=" * 60)
        print("Showcasing intelligent cost-aware memory routing")
        print("=" * 60)
        
        try:
            # Setup system
            adapters = await self.setup_system()
            
            # Create conversation scenarios
            scenarios = self.create_conversation_scenarios()
            
            # Demonstrate different scenarios with different budgets
            budget_types = [BudgetType.STANDARD, BudgetType.MINIMAL, BudgetType.PREMIUM]
            
            for scenario_name, items in scenarios.items():
                for budget_type in budget_types[:1]:  # Just standard for brevity
                    await self.demonstrate_scenario(scenario_name, items, budget_type)
            
            # Demonstrate retrieval capabilities
            await self.demonstrate_retrieval()
            
            # Show budget constraint effects
            await self.demonstrate_budget_constraints()
            
            # Display performance monitoring
            await self.demonstrate_performance_monitoring()
            
            print(f"\n{'='*60}")
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("Key Features Demonstrated:")
            print("✅ Intelligent content analysis and routing")
            print("✅ Cost-aware adapter selection")
            print("✅ Budget constraint handling")  
            print("✅ Cross-session memory retrieval")
            print("✅ Real-time performance monitoring")
            print("✅ Detailed decision reasoning")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if self.dispatcher:
                await self.dispatcher.cleanup()
            for adapter in adapters:
                await adapter.cleanup()
            
            # Clean up demo files
            import shutil
            demo_path = Path("./demo_storage")
            if demo_path.exists():
                shutil.rmtree(demo_path)
            
            trace_file = Path("./memory_traces.jsonl")
            if trace_file.exists():
                trace_file.unlink()


async def main():
    """Run the FrugalDispatcher demo."""
    demo = FrugalDispatcherDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())