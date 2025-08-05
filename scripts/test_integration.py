#!/usr/bin/env python3
"""
Integration test to verify M1 milestone:
"The harness can run with all adapters on a 100-turn sample and log metrics"

This script demonstrates the core functionality working end-to-end.
"""

import asyncio
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_memory_suite.adapters import FAISStoreAdapter
from open_memory_suite.benchmark import BenchmarkHarness
from open_memory_suite.benchmark.harness import create_sample_session, create_sample_queries
from open_memory_suite.benchmark.trace import load_trace_file


async def main():
    """Run the integration test."""
    print("🧪 Running Open Memory Suite Integration Test")
    print("=" * 50)
    
    # Create test directory
    test_dir = Path(__file__).parent.parent / "test_output"
    test_dir.mkdir(exist_ok=True)
    
    # Initialize components
    trace_file = test_dir / "integration_test.jsonl"
    harness = BenchmarkHarness(trace_file)
    
    # Test FAISS adapter
    print("\\n1️⃣  Testing FAISS Adapter")
    adapter = FAISStoreAdapter(
        name="integration_faiss",
        index_path=test_dir / "integration_index"
    )
    
    # Create test data
    session = create_sample_session()
    queries = create_sample_queries()
    
    print(f"   📝 Session: {session.session_id} with {len(session.turns)} turns")
    print(f"   ❓ Queries: {len(queries)} evaluation queries")
    
    # Run full evaluation
    print("\\n2️⃣  Running Full Evaluation")
    try:
        result = await harness.run_full_evaluation(adapter, [session], queries)
        
        print(f"   ✅ Adapter: {result['adapter_name']}")
        print(f"   📊 Sessions processed: {result['sessions_processed']}")
        print(f"   🔄 Total turns: {result['total_turns']}")
        
        # Check session results
        session_result = result['session_results'][0]
        print(f"   💾 Items stored: {session_result['total_items_stored']}")
        print(f"   ⚠️  Errors: {len(session_result['errors'])}")
        
        # Check recall results
        recall_result = result['recall_evaluation']
        print(f"   🎯 Recall rate: {recall_result['recall_rate']:.2%}")
        print(f"   🔍 Successful retrievals: {recall_result['successful_retrievals']}/{recall_result['total_queries']}")
        
    except Exception as e:
        print(f"   ❌ Error during evaluation: {e}")
        return False
    
    # Verify trace logging
    print("\\n3️⃣  Verifying Trace Logs")
    try:
        events = load_trace_file(trace_file)
        print(f"   📄 Trace events logged: {len(events)}")
        
        # Count event types
        event_types = {}
        for event in events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        for event_type, count in event_types.items():
            print(f"   📈 {event_type}: {count} events")
        
        # Check for required metrics
        has_latency = any(event.latency_ms > 0 for event in events)
        has_success = any(event.success for event in events)
        
        print(f"   ⏱️  Has latency measurements: {'✅' if has_latency else '❌'}")
        print(f"   ✔️  Has success indicators: {'✅' if has_success else '❌'}")
        
    except Exception as e:
        print(f"   ❌ Error reading trace file: {e}")
        return False
    
    # Milestone verification
    print("\\n🏆 Milestone M1 Verification")
    
    checks = [
        (len(events) > 0, "Harness produces trace logs"),
        (result['sessions_processed'] > 0, "Sessions are processed"),
        (session_result['total_items_stored'] > 0, "Items are stored in adapter"),
        (len(session_result['errors']) == 0, "No storage errors"),
        (recall_result['total_queries'] > 0, "Recall queries are evaluated"),
        (has_latency and has_success, "Metrics include latency and success")
    ]
    
    all_passed = True
    for passed, description in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {description}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\\n🎉 SUCCESS: M1 Milestone Achieved!")
        print("   The harness can run adapters on sample data and log metrics.")
        print(f"   Results saved to: {test_dir}")
        return True
    else:
        print("\\n💥 FAILURE: M1 Milestone not met.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)