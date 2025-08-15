#!/usr/bin/env python3
"""
Simple Benchmark Validation Script

Quick validation of the Open Memory Suite's core capabilities:
- Cost reduction measurement
- Routing accuracy testing  
- Performance benchmarking
- System health verification

Usage:
    poetry run python simple_benchmark.py
    poetry run python simple_benchmark.py --quick
    poetry run python simple_benchmark.py --detailed
"""

import asyncio
import time
from typing import Dict, Any, List, Tuple
import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Import Open Memory Suite components
from open_memory_suite.adapters import MemoryItem, InMemoryAdapter, FileStoreAdapter
from open_memory_suite.benchmark.cost_model import CostModel, BudgetType
from open_memory_suite.dispatcher import (
    FrugalDispatcher, 
    HeuristicPolicy,
    PolicyRegistry,
    ConversationContext
)

console = Console()

class SimpleBenchmark:
    """Simple benchmark for core system validation."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    async def setup_system(self) -> FrugalDispatcher:
        """Setup a basic dispatcher for testing."""
        console.print("🔧 Setting up test system...")
        
        # Create adapters
        adapters = [
            InMemoryAdapter("test_memory"),
            FileStoreAdapter("test_file", Path("./benchmark_temp"))
        ]
        
        # Initialize adapters
        for adapter in adapters:
            await adapter.initialize()
        
        # Create cost model
        cost_model = CostModel()
        
        # Create dispatcher
        dispatcher = FrugalDispatcher(adapters=adapters, cost_model=cost_model)
        
        # Setup policy
        policy_registry = PolicyRegistry()
        policy_registry.register(HeuristicPolicy(), set_as_default=True)
        dispatcher.policy_registry = policy_registry
        
        await dispatcher.initialize()
        
        console.print("✅ System setup complete")
        return dispatcher
    
    async def test_cost_reduction(self, dispatcher: FrugalDispatcher) -> Dict[str, Any]:
        """Test cost reduction vs naive storage."""
        console.print("\n💰 Testing Cost Reduction...")
        
        test_messages = [
            "ok",  # Should be discarded
            "got it",  # Should be discarded  
            "My name is Alice Chen",  # Should be stored
            "What is machine learning?",  # Should be stored
            "thanks",  # Should be discarded
            "I live in San Francisco",  # Should be stored
            "sounds good",  # Should be discarded
            "My phone number is 555-1234",  # Should be stored
        ]
        
        # Test with intelligent routing
        intelligent_cost = 0.0
        stored_count = 0
        
        for i, content in enumerate(test_messages):
            item = MemoryItem(
                content=content,
                speaker="user", 
                session_id=f"test_{i}",
                metadata={}
            )
            
            decision = await dispatcher.route_memory(item, f"test_{i}")
            success = await dispatcher.execute_decision(decision, item, f"test_{i}")
            
            if decision.estimated_cost:
                intelligent_cost += decision.estimated_cost.total_cost / 100
            
            if success:
                stored_count += 1
        
        # Calculate naive cost (store everything)
        naive_cost = len(test_messages) * 0.001  # Assume $0.001 per item
        
        cost_reduction = ((naive_cost - intelligent_cost) / naive_cost) * 100
        
        results = {
            "intelligent_cost": intelligent_cost,
            "naive_cost": naive_cost, 
            "cost_reduction_percent": cost_reduction,
            "messages_stored": stored_count,
            "messages_total": len(test_messages),
            "storage_rate": (stored_count / len(test_messages)) * 100
        }
        
        console.print(f"📊 Cost Results:")
        console.print(f"   Intelligent: ${intelligent_cost:.6f}")
        console.print(f"   Naive: ${naive_cost:.6f}")  
        console.print(f"   🎉 Reduction: {cost_reduction:.1f}%")
        console.print(f"   📝 Stored: {stored_count}/{len(test_messages)} ({results['storage_rate']:.1f}%)")
        
        return results
    
    async def test_routing_accuracy(self, dispatcher: FrugalDispatcher) -> Dict[str, Any]:
        """Test routing decision accuracy."""
        console.print("\n🧠 Testing Routing Accuracy...")
        
        test_cases = [
            ("thanks", "discard"),
            ("My name is John", "store"), 
            ("What is the weather?", "store"),
            ("ok", "discard"),
            ("I work at Google", "store"),
            ("got it", "discard")
        ]
        
        correct_decisions = 0
        total_decisions = len(test_cases)
        
        for i, (content, expected) in enumerate(test_cases):
            item = MemoryItem(
                content=content,
                speaker="user",
                session_id=f"accuracy_test_{i}",
                metadata={}
            )
            
            decision = await dispatcher.route_memory(item, f"accuracy_test_{i}")
            
            actual = "store" if decision.action.value != "drop" else "discard"
            
            if actual == expected:
                correct_decisions += 1
                status = "✅"
            else:
                status = "❌"
            
            console.print(f"   {status} '{content}' -> {actual} (expected {expected})")
        
        accuracy = (correct_decisions / total_decisions) * 100
        
        results = {
            "correct_decisions": correct_decisions,
            "total_decisions": total_decisions,
            "accuracy_percent": accuracy
        }
        
        console.print(f"🎯 Accuracy: {correct_decisions}/{total_decisions} ({accuracy:.1f}%)")
        
        return results
    
    async def test_performance(self, dispatcher: FrugalDispatcher) -> Dict[str, Any]:
        """Test system performance."""
        console.print("\n⚡ Testing Performance...")
        
        test_message = "This is a test message for performance evaluation"
        num_requests = 50
        
        latencies = []
        
        for i in range(num_requests):
            item = MemoryItem(
                content=f"{test_message} {i}",
                speaker="user",
                session_id=f"perf_test_{i}",
                metadata={}
            )
            
            start = time.time()
            decision = await dispatcher.route_memory(item, f"perf_test_{i}")
            await dispatcher.execute_decision(decision, item, f"perf_test_{i}")
            end = time.time()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        
        results = {
            "num_requests": num_requests,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "throughput_rps": 1000 / avg_latency if avg_latency > 0 else 0
        }
        
        console.print(f"📈 Performance:")
        console.print(f"   Average: {avg_latency:.1f}ms")
        console.print(f"   P95: {p95_latency:.1f}ms")
        console.print(f"   Throughput: {results['throughput_rps']:.1f} req/sec")
        
        return results
    
    async def test_system_health(self, dispatcher: FrugalDispatcher) -> Dict[str, Any]:
        """Test basic system health."""
        console.print("\n🏥 Testing System Health...")
        
        health_checks = {
            "dispatcher_initialized": dispatcher is not None,
            "adapters_available": len(dispatcher.adapters) > 0,
            "policy_registered": hasattr(dispatcher.policy_registry, '_default_policy') and dispatcher.policy_registry._default_policy is not None,
            "cost_model_loaded": dispatcher.cost_model is not None
        }
        
        all_healthy = all(health_checks.values())
        
        for check, status in health_checks.items():
            icon = "✅" if status else "❌"
            console.print(f"   {icon} {check.replace('_', ' ').title()}")
        
        results = {
            "all_healthy": all_healthy,
            "health_checks": health_checks,
            "num_adapters": len(dispatcher.adapters),
            "adapter_names": [getattr(adapter, 'name', str(adapter)) for adapter in dispatcher.adapters]
        }
        
        if all_healthy:
            console.print("🎉 System is healthy!")
        else:
            console.print("⚠️  System has issues")
        
        return results
    
    async def cleanup(self, dispatcher: FrugalDispatcher):
        """Cleanup test resources."""
        console.print("\n🧹 Cleaning up...")
        
        try:
            if hasattr(dispatcher, 'cleanup'):
                await dispatcher.cleanup()
            
            for adapter in dispatcher.adapters:
                if hasattr(adapter, 'cleanup'):
                    await adapter.cleanup()
            
            # Remove temp files
            temp_dir = Path("./benchmark_temp")
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
            console.print("✅ Cleanup complete")
        except Exception as e:
            console.print(f"⚠️  Cleanup warning: {e}")
    
    async def run_benchmark(self, quick: bool = False) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        console.print(Panel.fit(
            "🧠 [bold]Open Memory Suite - Simple Benchmark[/bold]\n"
            "Testing core system capabilities...",
            border_style="blue"
        ))
        
        dispatcher = await self.setup_system()
        
        try:
            # Run tests
            self.results["system_health"] = await self.test_system_health(dispatcher)
            self.results["cost_reduction"] = await self.test_cost_reduction(dispatcher)
            self.results["routing_accuracy"] = await self.test_routing_accuracy(dispatcher)
            
            if not quick:
                self.results["performance"] = await self.test_performance(dispatcher)
            
            # Summary
            await self.show_summary()
            
        finally:
            await self.cleanup(dispatcher)
        
        return self.results
    
    async def show_summary(self):
        """Show benchmark summary."""
        console.print("\n" + "="*60)
        console.print("🏆 [bold]BENCHMARK SUMMARY[/bold]")
        console.print("="*60)
        
        # Create summary table
        table = Table(title="Key Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Result", style="green")
        table.add_column("Status", style="yellow")
        
        # Cost reduction
        cost_result = self.results.get("cost_reduction", {})
        cost_reduction = cost_result.get("cost_reduction_percent", 0)
        cost_status = "✅ Excellent" if cost_reduction > 50 else "⚠️  Needs improvement"
        table.add_row(
            "Cost Reduction",
            f"{cost_reduction:.1f}%",
            cost_status
        )
        
        # Routing accuracy
        accuracy_result = self.results.get("routing_accuracy", {})
        accuracy = accuracy_result.get("accuracy_percent", 0)
        accuracy_status = "✅ Good" if accuracy > 80 else "⚠️  Needs improvement"
        table.add_row(
            "Routing Accuracy", 
            f"{accuracy:.1f}%",
            accuracy_status
        )
        
        # System health
        health_result = self.results.get("system_health", {})
        health = health_result.get("all_healthy", False)
        health_status = "✅ Healthy" if health else "❌ Issues detected"
        table.add_row(
            "System Health",
            "Pass" if health else "Fail",
            health_status
        )
        
        # Performance (if available)
        if "performance" in self.results:
            perf_result = self.results["performance"]
            avg_latency = perf_result.get("avg_latency_ms", 0)
            perf_status = "✅ Fast" if avg_latency < 100 else "⚠️  Slow" if avg_latency < 500 else "❌ Very slow"
            table.add_row(
                "Avg Latency",
                f"{avg_latency:.1f}ms",
                perf_status
            )
        
        console.print(table)
        
        # Overall assessment
        overall_score = 0
        if cost_reduction > 50:
            overall_score += 25
        if accuracy > 80:
            overall_score += 25
        if health:
            overall_score += 25
        if "performance" in self.results and self.results["performance"]["avg_latency_ms"] < 200:
            overall_score += 25
        
        if overall_score >= 75:
            console.print("\n🎉 [bold green]SYSTEM PERFORMANCE: EXCELLENT[/bold green]")
        elif overall_score >= 50:
            console.print("\n⚠️  [bold yellow]SYSTEM PERFORMANCE: GOOD[/bold yellow]") 
        else:
            console.print("\n❌ [bold red]SYSTEM PERFORMANCE: NEEDS WORK[/bold red]")
        
        # Execution time
        execution_time = time.time() - self.start_time
        console.print(f"\n⏱️  Benchmark completed in {execution_time:.1f} seconds")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Open Memory Suite Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (skip performance tests)")
    parser.add_argument("--detailed", action="store_true", help="Run detailed benchmark with extra metrics")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = SimpleBenchmark()
    
    try:
        results = await benchmark.run_benchmark(quick=args.quick)
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"\n💾 Results saved to {output_path}")
        
        # Exit code based on results
        if results.get("system_health", {}).get("all_healthy", False):
            return 0
        else:
            return 1
        
    except Exception as e:
        console.print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))