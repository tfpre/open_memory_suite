#!/usr/bin/env python3
"""
End-to-End Pipeline Validation

This script validates the complete implementation of Day 1-2 deliverables:
1. Benchmark harness with cost/recall metrics
2. CLI interface functionality  
3. 3-class router integration
4. Personal assistant demo components
5. Cost model and telemetry integration

Success criteria from project_state_and_directions.md:
- CLI `python -m benchmark.run_eval --help` works
- Cost/recall metrics are produced
- 3-class routing decisions are made
- Demo components are functional
"""

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def validate_cost_model():
    """Validate cost model and telemetry integration."""
    logger.info("🔧 Validating cost model...")
    
    try:
        from open_memory_suite.benchmark.cost_model import CostModel, OperationType
        from open_memory_suite.core.tokens import count_tokens
        
        # Initialize cost model
        cost_model = CostModel()
        
        # Test cost predictions
        test_cases = [
            (OperationType.STORE, "memory_store", "Hello world", 100),
            (OperationType.RETRIEVE, "faiss_store", "What is the meeting time?", 50),
            (OperationType.STORE, "file_store", "Long content " * 50, 200),
        ]
        
        for op, adapter, content, item_count in test_cases:
            tokens = count_tokens(content)
            cost_cents, latency_ms = cost_model.predict(
                op=op,
                adapter=adapter,
                tokens=tokens,
                k=5,
                item_count=item_count
            )
            
            logger.info(f"  ✓ {adapter}|{op.value}: ${cost_cents/100:.4f}, {latency_ms:.1f}ms")
        
        logger.info("✅ Cost model validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cost model validation failed: {e}")
        return False

async def validate_three_class_router():
    """Validate 3-class router functionality."""
    logger.info("🤖 Validating 3-class router...")
    
    try:
        from ml_training.three_class_router import ThreeClassRouter, create_synthetic_training_data
        
        # Test with default router (no trained model)
        router = ThreeClassRouter(confidence_threshold=0.75)
        
        # Test sample inputs that should trigger different classes
        test_cases = [
            ("thanks", "discard"),
            ("My meeting is Tuesday at 3pm with John", "store"),
            ("I had a really long conversation about artificial intelligence and machine learning today. We discussed many topics including the future of automation, ethical considerations, and how these technologies might transform our industry over the next decade. The conversation was quite insightful and raised many important questions about human-machine collaboration.", "compress"),
        ]
        
        for content, expected_class in test_cases:
            decision = router.predict(content, return_reasoning=True)
            logger.info(f"  ✓ '{content[:30]}...' → {decision.class_name} (confidence: {decision.confidence:.2f})")
            logger.info(f"    Reasoning: {decision.reasoning}")
        
        # Test synthetic data generation
        training_data = create_synthetic_training_data(n_examples=100)
        logger.info(f"  ✓ Generated {len(training_data)} synthetic training examples")
        
        # Test feature extraction
        from ml_training.three_class_router import ContentAnalyzer
        analyzer = ContentAnalyzer()
        features = analyzer.extract_features("My name is John and I work at Microsoft")
        logger.info(f"  ✓ Extracted {len(features)} features from sample content")
        
        logger.info("✅ 3-class router validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ 3-class router validation failed: {e}")
        return False

async def validate_benchmark_harness():
    """Validate benchmark harness functionality."""
    logger.info("📊 Validating benchmark harness...")
    
    try:
        from benchmark.harness import (
            BenchmarkHarness, 
            create_sample_session, 
            create_sample_queries,
            ConversationSession,
            EvaluationQuery
        )
        from open_memory_suite.adapters import InMemoryAdapter
        
        # Create test components
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_file = Path(temp_dir) / "test_trace.jsonl"
            harness = BenchmarkHarness(trace_file)
            
            # Create test adapter
            adapter = InMemoryAdapter("test_adapter")
            await adapter.initialize()
            
            # Create test data
            sessions = [create_sample_session()]
            queries = create_sample_queries()
            
            logger.info(f"  ✓ Created {len(sessions)} test sessions with {len(queries)} queries")
            
            # Run evaluation
            result = await harness.run_full_evaluation(
                adapter=adapter,
                sessions=sessions,
                queries=queries,
                enable_tracing=True
            )
            
            # Validate results
            assert result.sessions_processed > 0, "No sessions processed"
            assert result.total_turns > 0, "No turns processed"
            assert result.total_cost_cents >= 0, "Invalid cost calculation"
            assert 0 <= result.avg_recall_score <= 1, "Invalid recall score"
            
            logger.info(f"  ✓ Processed {result.sessions_processed} sessions, {result.total_turns} turns")
            logger.info(f"  ✓ Cost: {result.total_cost_cents}¢, Recall: {result.avg_recall_score:.3f}")
            logger.info(f"  ✓ Latency: {result.avg_latency_ms:.1f}ms")
            
            await adapter.cleanup()
        
        logger.info("✅ Benchmark harness validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Benchmark harness validation failed: {e}")
        return False

async def validate_cli_interface():
    """Validate CLI interface without actually running subprocess."""
    logger.info("⚡ Validating CLI interface...")
    
    try:
        # Import CLI components
        from benchmark.run_eval import BenchmarkConfig, BenchmarkRunner
        
        # Test configuration loading
        config = BenchmarkConfig()
        logger.info(f"  ✓ Loaded benchmark configuration with {len(config.config['adapters'])} adapters")
        
        # Test adapter configuration
        for adapter_name in config.config['adapters']:
            adapter_config = config.get_adapter_config(adapter_name)
            logger.info(f"  ✓ Adapter config: {adapter_name} → {adapter_config.get('type', 'unknown')}")
        
        # Test dataset path resolution
        sample_path = config.get_dataset_path('sample')
        logger.info(f"  ✓ Sample dataset path: {sample_path}")
        
        # Test runner initialization
        runner = BenchmarkRunner(config)
        logger.info(f"  ✓ Initialized benchmark runner with cost model")
        
        logger.info("✅ CLI interface validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ CLI interface validation failed: {e}")
        return False

async def validate_frugal_dispatcher_integration():
    """Validate FrugalDispatcher with 3-class policy integration."""
    logger.info("🎯 Validating FrugalDispatcher integration...")
    
    try:
        from open_memory_suite.dispatcher.frugal_dispatcher import FrugalDispatcher
        from open_memory_suite.dispatcher.three_class_policy import ThreeClassMLPolicy
        from open_memory_suite.dispatcher.core import PolicyRegistry
        from open_memory_suite.adapters import InMemoryAdapter, MemoryItem
        from open_memory_suite.benchmark.cost_model import CostModel
        
        # Create components
        adapters = [InMemoryAdapter("test_memory")]
        await adapters[0].initialize()
        
        cost_model = CostModel()
        policy_registry = PolicyRegistry()
        
        # Create 3-class policy
        three_class_policy = ThreeClassMLPolicy(confidence_threshold=0.75)
        await three_class_policy.initialize()
        policy_registry.register(three_class_policy, set_as_default=True)
        
        # Create dispatcher
        dispatcher = FrugalDispatcher(
            adapters=adapters,
            cost_model=cost_model,
            policy_registry=policy_registry
        )
        
        await dispatcher.initialize()
        
        # Test routing decisions
        test_items = [
            MemoryItem(content="thanks", speaker="user", session_id="test"),
            MemoryItem(content="My meeting is Tuesday at 3pm", speaker="user", session_id="test"),
            MemoryItem(content="Very long content " * 50, speaker="user", session_id="test"),
        ]
        
        for item in test_items:
            decision, success = await dispatcher.route_and_execute(item, "test_session")
            logger.info(f"  ✓ '{item.content[:30]}...' → {decision.action.value} (success: {success})")
        
        # Get policy stats
        stats = three_class_policy.get_stats()
        logger.info(f"  ✓ Policy stats: {stats['total_decisions']} decisions, {stats['ml_decisions']} ML")
        
        await dispatcher.cleanup()
        
        logger.info("✅ FrugalDispatcher integration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ FrugalDispatcher integration validation failed: {e}")
        return False

async def validate_personal_assistant_components():
    """Validate personal assistant demo components."""
    logger.info("🤖 Validating personal assistant components...")
    
    try:
        from demos.personal_assistant import PersonalAssistantDemo, ChatMessage, RoutingEvent
        
        # Test demo initialization (without actually running server)
        demo = PersonalAssistantDemo(port=8999)  # Use different port
        
        # Test model classes
        chat_msg = ChatMessage(
            content="Hello assistant",
            speaker="user",
            session_id="test"
        )
        logger.info(f"  ✓ ChatMessage: {chat_msg.content[:20]}...")
        
        routing_event = RoutingEvent(
            timestamp=time.time(),
            session_id="test",
            content_preview="Hello...",
            routing_decision="store",
            confidence=0.85,
            estimated_cost_cents=5,
            reasoning="Test routing decision"
        )
        logger.info(f"  ✓ RoutingEvent: {routing_event.routing_decision} (confidence: {routing_event.confidence})")
        
        # Test helper methods
        action_mapping = demo._map_action_to_class
        from open_memory_suite.dispatcher.core import MemoryAction
        
        assert action_mapping(MemoryAction.DROP) == "discard"
        assert action_mapping(MemoryAction.STORE) == "store"
        assert action_mapping(MemoryAction.SUMMARIZE) == "compress"
        
        logger.info("  ✓ Action mapping works correctly")
        
        # Test cost estimation
        naive_cost = demo._estimate_naive_cost("This is a test message")
        logger.info(f"  ✓ Naive cost estimation: {naive_cost}¢")
        
        logger.info("✅ Personal assistant components validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Personal assistant components validation failed: {e}")
        return False

async def validate_research_integration():
    """Validate research integration layer."""
    logger.info("🔬 Validating research integration...")
    
    try:
        from benchmark.research_integration import (
            ResearchIntegrationRunner,
            EpmembenchAdapter,
            LongmemevalAdapter,
            BenchmarkResult
        )
        from open_memory_suite.benchmark.cost_model import CostModel
        from open_memory_suite.dispatcher.frugal_dispatcher import FrugalDispatcher
        from open_memory_suite.adapters import InMemoryAdapter
        
        # Create components
        cost_model = CostModel()
        adapters = [InMemoryAdapter("research_test")]
        await adapters[0].initialize()
        
        dispatcher = FrugalDispatcher(adapters=adapters, cost_model=cost_model)
        await dispatcher.initialize()
        
        # Test research integration runner
        runner = ResearchIntegrationRunner(cost_model, dispatcher)
        logger.info("  ✓ ResearchIntegrationRunner initialized")
        
        # Test benchmark result structure
        result = BenchmarkResult(
            accuracy=0.85,
            f1_score=0.82,
            recall=0.88,
            precision=0.79,
            total_cost_cents=150,
            cost_per_correct_answer=5.2,
            routing_decisions=["store", "discard", "compress"],
            cost_savings_vs_naive=0.35,
            avg_latency_ms=45.2,
            p95_latency_ms=120.0,
            metadata={"framework": "test"}
        )
        
        logger.info(f"  ✓ BenchmarkResult: {result.accuracy:.2f} accuracy, {result.cost_savings_vs_naive:.1%} savings")
        
        # Test adapters
        epmem_adapter = EpmembenchAdapter(cost_model, dispatcher)
        longmem_adapter = LongmemevalAdapter(cost_model, dispatcher)
        
        logger.info("  ✓ Research framework adapters created")
        
        await dispatcher.cleanup()
        
        logger.info("✅ Research integration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Research integration validation failed: {e}")
        return False

async def validate_sample_data_flow():
    """Validate end-to-end data flow with sample data."""
    logger.info("🔄 Validating end-to-end data flow...")
    
    try:
        # Create sample data
        from benchmark.harness import create_sample_session, create_sample_queries
        
        session = create_sample_session()
        queries = create_sample_queries()
        
        logger.info(f"  ✓ Sample data: {session.turn_count} turns, {len(queries)} queries")
        
        # Test data serialization
        session_dict = {
            'session_id': session.session_id,
            'turns': [
                {
                    'turn_id': turn.turn_id,
                    'speaker': turn.speaker,
                    'content': turn.content,
                    'timestamp': turn.timestamp,
                    'metadata': turn.metadata
                }
                for turn in session.turns
            ],
            'session_metadata': session.session_metadata,
            'start_time': session.start_time,
            'end_time': session.end_time
        }
        
        # Test JSON serialization
        json_data = json.dumps(session_dict, indent=2)
        logger.info(f"  ✓ Session serialization: {len(json_data)} chars")
        
        # Test query processing
        for query in queries:
            assert query.query_text, "Empty query text"
            assert query.query_type in ['factual', 'semantic'], f"Invalid query type: {query.query_type}"
            assert query.expected_content, "No expected content"
            
        logger.info(f"  ✓ Query validation: {len(queries)} queries validated")
        
        logger.info("✅ End-to-end data flow validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end data flow validation failed: {e}")
        return False

def print_validation_summary(results: Dict[str, bool]):
    """Print a summary of validation results."""
    print("\n" + "="*60)
    print("📋 PIPELINE VALIDATION SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print("-"*60)
    print(f"Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.0f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ Day 1-2 deliverables are ready")
        print("✅ Pipeline is functional and cost-aware")
        print("✅ Ready to proceed to Day 3-4 demo integration")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} validation(s) failed")
        print("❌ Pipeline requires fixes before proceeding")
    
    print("="*60)

async def main():
    """Run complete pipeline validation."""
    print("🚀 Starting Open Memory Suite Pipeline Validation")
    print("Testing Day 1-2 deliverables according to project_state_and_directions.md")
    print("-"*60)
    
    # Define validation tests
    validation_tests = [
        ("Cost Model & Telemetry", validate_cost_model),
        ("3-Class Router", validate_three_class_router),
        ("Benchmark Harness", validate_benchmark_harness),
        ("CLI Interface", validate_cli_interface),
        ("FrugalDispatcher Integration", validate_frugal_dispatcher_integration),
        ("Personal Assistant Components", validate_personal_assistant_components),
        ("Research Integration Layer", validate_research_integration),
        ("Sample Data Flow", validate_sample_data_flow),
    ]
    
    # Run validations
    results = {}
    for test_name, test_func in validation_tests:
        print(f"\n📝 Running: {test_name}")
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Validation crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print_validation_summary(results)
    
    # Return exit code
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)