"""Tests for ML-enhanced memory routing components."""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import List

import pytest
import torch

from open_memory_suite.adapters import MemoryItem, InMemoryAdapter
from open_memory_suite.benchmark import CostModel, BenchmarkHarness
from open_memory_suite.benchmark.cost_model import BudgetType
from open_memory_suite.dispatcher import (
    ConversationContext,
    MemoryAction,
    HeuristicPolicy,
    ML_AVAILABLE,
)

# Skip all ML tests if components not available
pytestmark = pytest.mark.skipif(not ML_AVAILABLE, reason="ML components not available")

if ML_AVAILABLE:
    from open_memory_suite.dispatcher import (
        MLPolicy,
        TriageClassifier,
        MLTrainer,
        DataCollector,
    )


class TestTriageClassifier:
    """Test the DistilBERT + LoRA triage classifier."""
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        model = TriageClassifier(
            num_actions=5,
            lora_r=8,  # Smaller for testing
        )
        
        assert model.num_actions == 5
        assert model.action_to_idx[MemoryAction.STORE] == 0
        assert model.action_to_idx[MemoryAction.DROP] == 4
        
        # Test model components exist
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'classifier')
        assert hasattr(model, 'context_embedding')
    
    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        model = TriageClassifier(num_actions=5, lora_r=8)
        model.eval()
        
        # Create dummy inputs
        batch_size = 2
        seq_length = 32
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        context_features = torch.randn(batch_size, 10)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, context_features)
        
        # Check output shapes
        assert outputs["logits"].shape == (batch_size, 5)
        assert outputs["hidden_states"].shape == (batch_size, model.config.hidden_size)
        assert outputs["context_embedding"].shape == (batch_size, 64)
    
    def test_prediction(self):
        """Test prediction functionality."""
        model = TriageClassifier(num_actions=5, lora_r=8)
        model.eval()
        
        # Create dummy inputs
        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones(1, 32)
        context_features = torch.randn(1, 10)
        
        # Make prediction
        action, confidence = model.predict(input_ids, attention_mask, context_features)
        
        # Check outputs
        assert action in model.action_to_idx or action in model.action_to_idx.values()
        assert 0.0 <= confidence <= 1.0


class TestMLPolicy:
    """Test the ML-enhanced memory policy."""
    
    @pytest.fixture
    async def ml_policy(self):
        """Create an ML policy for testing."""
        policy = MLPolicy(
            model_path=None,  # Use pre-trained model
            confidence_threshold=0.5,
            fallback_to_heuristic=True,
        )
        await policy.initialize()
        return policy
    
    async def test_policy_initialization(self, ml_policy):
        """Test ML policy initialization."""
        assert ml_policy.name == "ml_enhanced"
        assert ml_policy.model is not None
        assert ml_policy.tokenizer is not None
        assert ml_policy.heuristic_policy is not None
    
    async def test_decide_action(self, ml_policy):
        """Test action decision making."""
        # Create test item and context
        item = MemoryItem(
            content="What is machine learning?",
            speaker="user",
            session_id="test_session",
        )
        
        context = ConversationContext(
            session_id="test_session",
            budget_type=BudgetType.STANDARD,
        )
        
        # Make decision
        action = await ml_policy.decide_action(item, context)
        
        # Should return a valid action
        assert isinstance(action, MemoryAction)
    
    async def test_confidence_fallback(self):
        """Test fallback to heuristic when confidence is low."""
        policy = MLPolicy(
            confidence_threshold=0.99,  # Very high threshold
            fallback_to_heuristic=True,
        )
        await policy.initialize()
        
        item = MemoryItem(
            content="test",
            speaker="user",
            session_id="test_session",
        )
        
        context = ConversationContext(
            session_id="test_session",
            budget_type=BudgetType.STANDARD,
        )
        
        # Should fall back to heuristic
        action = await policy.decide_action(item, context)
        assert isinstance(action, MemoryAction)
        
        # Check that heuristic fallback was used
        stats = policy.get_stats()
        assert stats["heuristic_fallbacks"] > 0
    
    async def test_stats_tracking(self, ml_policy):
        """Test performance statistics tracking."""
        initial_stats = ml_policy.get_stats()
        
        # Make some decisions
        for i in range(3):
            item = MemoryItem(
                content=f"Test message {i}",
                speaker="user",
                session_id="test_session",
            )
            
            context = ConversationContext(
                session_id="test_session",
                budget_type=BudgetType.STANDARD,
            )
            
            await ml_policy.decide_action(item, context)
        
        # Check stats updated
        final_stats = ml_policy.get_stats()
        total_decisions = final_stats["ml_decisions"] + final_stats["heuristic_fallbacks"]
        assert total_decisions >= initial_stats["decisions_made"]


class TestDataCollector:
    """Test the training data collection system."""
    
    @pytest.fixture
    async def data_collector(self):
        """Create a data collector for testing."""
        # Create minimal components
        heuristic_policy = HeuristicPolicy()
        
        adapters = [InMemoryAdapter("memory_store")]
        for adapter in adapters:
            await adapter.initialize()
        
        cost_model = CostModel()
        harness = BenchmarkHarness(
            adapters=adapters,
            cost_model=cost_model,
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = DataCollector(
                heuristic_policy=heuristic_policy,
                harness=harness,
                output_path=Path(temp_dir),
            )
            yield collector
            
            # Cleanup
            for adapter in adapters:
                await adapter.cleanup()
    
    async def test_conversation_scenario_generation(self, data_collector):
        """Test generation of diverse conversation scenarios."""
        scenarios = data_collector._create_conversation_scenarios(5)
        
        assert len(scenarios) == 5
        for scenario in scenarios:
            assert "session_id" in scenario
            assert "budget_type" in scenario
            assert "conversation_type" in scenario
            assert "num_turns" in scenario
            assert "user_profile" in scenario
            
            # Check budget type is valid
            assert scenario["budget_type"] in [
                BudgetType.MINIMAL,
                BudgetType.STANDARD, 
                BudgetType.PREMIUM,
            ]
    
    async def test_training_data_collection(self, data_collector):
        """Test training data collection process."""
        # Collect a small amount of data
        examples, class_distribution = await data_collector.collect_training_data(
            num_conversations=5,
            min_examples_per_action=5,
            max_retries=1,
        )
        
        assert len(examples) > 0
        assert len(class_distribution) > 0
        
        # Check example structure
        for example in examples[:3]:  # Check first few
            assert "text" in example
            assert "label" in example
            assert "action" in example
            assert "context_features" in example
            assert "session_id" in example
            
            # Check label is valid
            assert 0 <= example["label"] < 5
            
            # Check context features
            assert isinstance(example["context_features"], list)
            assert len(example["context_features"]) == 10


class TestMLTrainer:
    """Test the ML training pipeline."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        return [
            {
                "text": "Context: [user]: Hello | Turn 1, Budget: standard | Current: [user]: What is AI?",
                "label": 0,  # STORE
                "action": "store",
                "context_features": [1.0, 15.0, 3.0, 0.0, 0.1, 1.0, 1.0, 0.0, 0.0, 0.0],
            },
            {
                "text": "Context: [user]: Thanks | Turn 2, Budget: standard | Current: [assistant]: AI is...",
                "label": 3,  # SUMMARIZE
                "action": "summarize",
                "context_features": [2.0, 150.0, 25.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            {
                "text": "Context: [assistant]: AI is... | Turn 3, Budget: standard | Current: [user]: ok",
                "label": 4,  # DROP
                "action": "drop",
                "context_features": [3.0, 2.0, 1.0, 1.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0],
            },
        ] * 20  # Replicate to have enough samples
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = MLTrainer(
                output_dir=Path(temp_dir),
                use_wandb=False,  # Disable for testing
            )
            
            assert trainer.model_config["num_actions"] == 5
            assert trainer.training_config["num_epochs"] == 3
            assert trainer.tokenizer is not None
    
    def test_dataset_preparation(self, sample_training_data):
        """Test dataset preparation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = MLTrainer(
                output_dir=Path(temp_dir),
                use_wandb=False,
            )
            
            train_data, val_data = trainer._prepare_datasets(
                training_data=sample_training_data,
                validation_split=0.2,
            )
            
            assert len(train_data) > 0
            assert len(val_data) > 0
            assert len(train_data) + len(val_data) == len(sample_training_data)
            
            # Check datasets were created
            assert trainer.train_dataset is not None
            assert trainer.val_dataset is not None
    
    @pytest.mark.slow
    async def test_quick_training(self, sample_training_data):
        """Test a quick training run (minimal epochs)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create trainer with minimal configuration
            trainer = MLTrainer(
                output_dir=Path(temp_dir),
                use_wandb=False,
                model_config={
                    "num_actions": 5,
                    "lora_r": 4,  # Very small for fast training
                    "lora_alpha": 8,
                },
                training_config={
                    "num_epochs": 1,  # Single epoch
                    "batch_size": 8,
                    "learning_rate": 1e-4,
                    "eval_steps": 5,
                    "save_steps": 10,
                    "logging_steps": 2,
                    "warmup_steps": 0,
                },
            )
            
            # Train model
            results = await trainer.train(
                training_data=sample_training_data,
                validation_split=0.2,
                experiment_name="test_training",
            )
            
            # Check results
            assert "train_loss" in results
            assert "eval_results" in results
            assert "model_path" in results
            
            # Check model was saved
            model_path = Path(results["model_path"])
            assert model_path.exists()
            assert (model_path / "model.pt").exists()
            assert (model_path / "tokenizer").exists()


class TestMLIntegration:
    """Integration tests for ML components working together."""
    
    @pytest.mark.slow
    async def test_end_to_end_ml_pipeline(self):
        """Test complete ML pipeline end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Create minimal training data
            training_data = [
                {
                    "text": f"Turn {i}, Budget: standard | Current: [user]: Test message {i}",
                    "label": i % 5,  # Cycle through actions
                    "action": ["store", "store_faiss", "store_file", "summarize", "drop"][i % 5],
                    "context_features": [float(i), 10.0, 3.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0],
                }
                for i in range(25)  # Minimal dataset
            ]
            
            # Save training data
            data_path = temp_path / "training_data.json"
            with open(data_path, 'w') as f:
                json.dump(training_data, f)
            
            # Step 2: Train model
            trainer = MLTrainer(
                output_dir=temp_path / "models",
                use_wandb=False,
                model_config={"lora_r": 4, "lora_alpha": 8},
                training_config={
                    "num_epochs": 1,
                    "batch_size": 4,
                    "eval_steps": 5,
                    "save_steps": 10,
                },
            )
            
            results = await trainer.train(
                training_data=training_data,
                experiment_name="integration_test",
            )
            
            model_path = Path(results["model_path"])
            
            # Step 3: Test trained model
            ml_policy = MLPolicy(
                model_path=model_path,
                confidence_threshold=0.0,  # Always use ML
                fallback_to_heuristic=False,
            )
            await ml_policy.initialize()
            
            # Step 4: Make predictions
            test_item = MemoryItem(
                content="What is the weather today?",
                speaker="user",
                session_id="integration_test",
            )
            
            context = ConversationContext(
                session_id="integration_test",
                budget_type=BudgetType.STANDARD,
            )
            
            action = await ml_policy.decide_action(test_item, context)
            
            # Verify the pipeline worked
            assert isinstance(action, MemoryAction)
            
            # Check stats
            stats = ml_policy.get_stats()
            assert stats["ml_decisions"] > 0
            assert stats["model_loaded"] is True


# Mark slow tests
@pytest.mark.slow
class TestMLPerformance:
    """Performance tests for ML components."""
    
    async def test_inference_speed(self):
        """Test that inference is fast enough for real-time use."""
        policy = MLPolicy(confidence_threshold=0.0)
        await policy.initialize()
        
        # Time multiple inferences
        import time
        
        test_items = [
            MemoryItem(
                content=f"Test message {i} with some content to analyze",
                speaker="user" if i % 2 == 0 else "assistant",
                session_id="perf_test",
            )
            for i in range(10)
        ]
        
        context = ConversationContext(
            session_id="perf_test",
            budget_type=BudgetType.STANDARD,
        )
        
        start_time = time.time()
        
        for item in test_items:
            await policy.decide_action(item, context)
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / len(test_items)) * 1000
        
        # Should be under 500ms per decision for real-time use
        assert avg_time_ms < 500, f"Average inference time {avg_time_ms:.1f}ms too slow"
        
        print(f"✅ Average inference time: {avg_time_ms:.1f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])