"""
Training pipeline for ML-enhanced memory routing policy.

This module provides the infrastructure to train DistilBERT + LoRA models
for intelligent memory routing decisions, including data collection,
training loop, evaluation, and model persistence.
"""

import asyncio
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset
import wandb
from tqdm import tqdm

from ..adapters.base import MemoryItem
from ..benchmark.harness import BenchmarkHarness
from ..benchmark.cost_model import BudgetType
from .core import ConversationContext, MemoryAction
from .ml_policy import TriageClassifier, MLPolicy
from .heuristic_policy import HeuristicPolicy


class RoutingDataset(Dataset):
    """PyTorch dataset for memory routing decisions."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        context_features: Optional[List[List[float]]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of input texts
            labels: List of target labels (action indices)
            context_features: Optional context feature vectors
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.context_features = context_features
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        assert len(texts) == len(labels), "Texts and labels must have same length"
        if context_features:
            assert len(texts) == len(context_features), "Context features length mismatch"
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            item = {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(label, dtype=torch.long),
            }
        else:
            # Simple tokenization for testing
            item = {
                "text": text,
                "labels": torch.tensor(label, dtype=torch.long),
            }
        
        # Add context features if available
        if self.context_features:
            context = self.context_features[idx]
            item["context_features"] = torch.tensor(context, dtype=torch.float32)
        
        return item


class DataCollector:
    """
    Collects training data by running conversations through the heuristic policy
    and capturing routing decisions with their context.
    """
    
    def __init__(
        self,
        heuristic_policy: HeuristicPolicy,
        harness: BenchmarkHarness,
        output_path: Path,
    ):
        """
        Initialize the data collector.
        
        Args:
            heuristic_policy: Policy to collect decisions from
            harness: Benchmark harness for running conversations
            output_path: Where to save collected data
        """
        self.heuristic_policy = heuristic_policy
        self.harness = harness
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Action mapping for labels
        self.action_to_idx = {
            MemoryAction.STORE: 0,
            "store_faiss": 1,  # Will be inferred from adapter selection
            "store_file": 2,
            MemoryAction.SUMMARIZE: 3,
            MemoryAction.DROP: 4,
        }
    
    async def collect_training_data(
        self,
        num_conversations: int = 100,
        min_examples_per_action: int = 50,
        max_retries: int = 3,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Collect training data from conversations.
        
        Args:
            num_conversations: Number of conversations to simulate
            min_examples_per_action: Minimum examples per action type
            max_retries: Maximum retries if data is unbalanced
            
        Returns:
            Tuple of (training_examples, class_distribution)
        """
        print(f"🎯 Collecting training data from {num_conversations} conversations...")
        
        all_examples = []
        
        for retry in range(max_retries):
            print(f"📝 Collection attempt {retry + 1}/{max_retries}")
            
            examples = await self._collect_examples(num_conversations)
            all_examples.extend(examples)
            
            # Check class balance
            class_counts = self._get_class_distribution(all_examples)
            print(f"📊 Class distribution: {class_counts}")
            
            # Check if we have enough examples for each class
            if all(count >= min_examples_per_action for count in class_counts.values()):
                print("✅ Sufficient examples collected for all classes")
                break
            
            # Increase conversation count for next retry
            num_conversations = int(num_conversations * 1.5)
        
        # Save collected data
        output_file = self.output_path / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_examples, f, indent=2, default=str)
        
        print(f"💾 Saved {len(all_examples)} examples to {output_file}")
        
        return all_examples, self._get_class_distribution(all_examples)
    
    async def _collect_examples(self, num_conversations: int) -> List[Dict[str, Any]]:
        """Collect examples from simulated conversations."""
        examples = []
        
        # Create diverse conversation scenarios
        scenarios = self._create_conversation_scenarios(num_conversations)
        
        for i, scenario in enumerate(tqdm(scenarios, desc="Collecting data")):
            try:
                session_examples = await self._simulate_conversation(scenario, f"session_{i}")
                examples.extend(session_examples)
            except Exception as e:
                print(f"❌ Error in conversation {i}: {e}")
                continue
        
        return examples
    
    def _create_conversation_scenarios(self, num_conversations: int) -> List[Dict[str, Any]]:
        """Create diverse conversation scenarios for data collection."""
        scenarios = []
        
        budget_types = [BudgetType.MINIMAL, BudgetType.STANDARD, BudgetType.PREMIUM]
        conversation_types = [
            "casual_chat",
            "technical_discussion", 
            "information_sharing",
            "question_answering",
            "task_planning",
        ]
        
        for i in range(num_conversations):
            scenario = {
                "session_id": f"training_session_{i}",
                "budget_type": random.choice(budget_types),
                "conversation_type": random.choice(conversation_types),
                "num_turns": random.randint(10, 50),
                "user_profile": self._generate_user_profile(),
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_user_profile(self) -> Dict[str, Any]:
        """Generate a diverse user profile for conversation simulation."""
        profiles = [
            {"name": "Alice Chen", "occupation": "Software Engineer", "interests": ["technology", "AI", "programming"]},
            {"name": "Bob Johnson", "occupation": "Teacher", "interests": ["education", "books", "travel"]},
            {"name": "Carol Williams", "occupation": "Doctor", "interests": ["medicine", "research", "fitness"]},
            {"name": "David Miller", "occupation": "Student", "interests": ["learning", "games", "music"]},
            {"name": "Emma Davis", "occupation": "Designer", "interests": ["art", "creativity", "fashion"]},
        ]
        return random.choice(profiles)
    
    async def _simulate_conversation(
        self,
        scenario: Dict[str, Any],
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Simulate a conversation and collect routing decisions."""
        examples = []
        
        # Create conversation context
        context = ConversationContext(
            session_id=session_id,
            budget_type=scenario["budget_type"],
            conversation_type=scenario["conversation_type"],
        )
        
        # Generate conversation turns
        conversation_turns = self._generate_conversation_turns(scenario)
        
        for turn_idx, (speaker, content) in enumerate(conversation_turns):
            # Create memory item
            item = MemoryItem(
                content=content,
                speaker=speaker,
                session_id=session_id,
                metadata={"turn": turn_idx, "scenario": scenario["conversation_type"]},
            )
            
            # Get routing decision from heuristic policy
            action = await self.heuristic_policy.decide_action(item, context)
            
            # Choose adapter to get specific routing decision
            if action == MemoryAction.STORE:
                # Simulate adapter selection
                mock_adapters = [
                    type('MockAdapter', (), {'name': 'memory_store'})(),
                    type('MockAdapter', (), {'name': 'faiss_store'})(),
                    type('MockAdapter', (), {'name': 'file_store'})(),
                ]
                selected_adapter = await self.heuristic_policy.choose_adapter(
                    item, mock_adapters, context
                )
                specific_action = f"store_{selected_adapter.name.replace('_store', '')}" if selected_adapter else "store"
            else:
                specific_action = action.value
            
            # Prepare input text for model
            input_text = self._prepare_input_text(item, context)
            
            # Extract context features
            context_features = self._extract_context_features(item, context)
            
            # Create training example
            example = {
                "text": input_text,
                "label": self.action_to_idx.get(specific_action, 0),
                "action": specific_action,
                "context_features": context_features,
                "item_content": content,
                "speaker": speaker,
                "turn": turn_idx,
                "session_id": session_id,
                "scenario": scenario,
            }
            
            examples.append(example)
            
            # Update context
            context.turn_count += 1
            context.recent_turns.append(item)
            if len(context.recent_turns) > 10:
                context.recent_turns = context.recent_turns[-10:]
        
        return examples
    
    def _generate_conversation_turns(self, scenario: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Generate realistic conversation turns based on scenario."""
        turns = []
        user_profile = scenario["user_profile"]
        conversation_type = scenario["conversation_type"]
        num_turns = scenario["num_turns"]
        
        # Base conversation starters by type
        starters = {
            "casual_chat": [
                ("user", f"Hi! My name is {user_profile['name']}."),
                ("assistant", f"Nice to meet you, {user_profile['name']}! How are you doing today?"),
            ],
            "technical_discussion": [
                ("user", "I'm working on a machine learning project and need some advice."),
                ("assistant", "I'd be happy to help! What kind of ML project are you working on?"),
            ],
            "information_sharing": [
                ("user", f"I work as a {user_profile['occupation']} and I'm interested in {user_profile['interests'][0]}."),
                ("assistant", f"That's interesting! Tell me more about your work as a {user_profile['occupation']}."),
            ],
            "question_answering": [
                ("user", "I have some questions about artificial intelligence."),
                ("assistant", "Great! I'm here to help. What would you like to know about AI?"),
            ],
            "task_planning": [
                ("user", "I need help planning a project related to my work."),
                ("assistant", "I'd be glad to help you plan your project. What's the project about?"),
            ],
        }
        
        # Add conversation starter
        turns.extend(starters.get(conversation_type, starters["casual_chat"]))
        
        # Generate additional turns with variety
        templates = self._get_conversation_templates(conversation_type, user_profile)
        
        while len(turns) < num_turns:
            template = random.choice(templates)
            speaker = "user" if len(turns) % 2 == 0 else "assistant"
            content = template.format(**user_profile)
            turns.append((speaker, content))
        
        return turns[:num_turns]
    
    def _get_conversation_templates(self, conv_type: str, profile: Dict[str, Any]) -> List[str]:
        """Get conversation templates for different types."""
        templates = {
            "casual_chat": [
                "I really enjoy {interests[0]} in my free time.",
                "What do you think about {interests[1]}?",
                "As a {occupation}, I often think about these topics.",
                "That's a great point! I hadn't considered that before.",
                "Thanks for the explanation!",
                "Could you tell me more about that?",
                "I see what you mean.",
                "That makes sense to me.",
            ],
            "technical_discussion": [
                "I'm using Python for this project.",
                "The model accuracy is around 85% currently.",
                "I'm thinking about using transformer architecture.",
                "What about data preprocessing steps?",
                "How would you handle overfitting in this case?",
                "The dataset has about 10,000 samples.",
                "I'm particularly interested in NLP applications.",
                "What metrics would you recommend?",
            ],
            "information_sharing": [
                "In my experience as a {occupation}, I've learned that...",
                "One interesting aspect of {interests[0]} is...",
                "I recently discovered something fascinating about {interests[1]}.",
                "My phone number is 555-0123 in case you need to reach me.",
                "I work at TechCorp downtown on Main Street.",
                "We usually meet every Tuesday at 3 PM.",
                "The project deadline is March 15th, 2024.",
                "I prefer email communication for work matters.",
            ],
        }
        
        return templates.get(conv_type, templates["casual_chat"])
    
    def _prepare_input_text(self, item: MemoryItem, context: ConversationContext) -> str:
        """Prepare input text for the model."""
        speaker_prefix = f"[{item.speaker or 'unknown'}]: " if item.speaker else ""
        
        recent_context = ""
        if context.recent_turns:
            recent_turns = context.recent_turns[-2:]
            recent_context = " | ".join([
                f"[{turn.speaker or 'unknown'}]: {turn.content[:100]}"
                for turn in recent_turns
            ])
            recent_context = f"Context: {recent_context} | "
        
        metadata = f"Turn {context.turn_count}, Budget: {context.budget_type.value}"
        input_text = f"{recent_context}{metadata} | Current: {speaker_prefix}{item.content}"
        
        return input_text
    
    def _extract_context_features(self, item: MemoryItem, context: ConversationContext) -> List[float]:
        """Extract context features for the model."""
        return [
            float(context.turn_count),
            float(len(item.content)),
            float(len(item.content.split())),
            float(context.total_stored_items),
            float(context.get_session_duration_minutes()),
            1.0 if item.speaker == "user" else 0.0,
            1.0 if "?" in item.content else 0.0,
            1.0 if context.is_budget_critical() else 0.0,
            float(context.budget_type.value == "minimal"),
            float(context.budget_type.value == "premium"),
        ]
    
    def _get_class_distribution(self, examples: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of classes in the dataset."""
        distribution = {}
        for example in examples:
            action = example["action"]
            distribution[action] = distribution.get(action, 0) + 1
        return distribution


class MLTrainer:
    """
    Trainer for the ML-enhanced memory routing policy.
    
    Handles the complete training pipeline including data loading,
    model training with LoRA, evaluation, and model persistence.
    """
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        output_dir: Path = Path("./ml_models"),
        use_wandb: bool = True,
    ):
        """
        Initialize the trainer.
        
        Args:
            model_config: Configuration for the model
            training_config: Configuration for training
            output_dir: Directory to save trained models
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.model_config = model_config or self._get_default_model_config()
        self.training_config = training_config or self._get_default_training_config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Training state
        self.model: Optional[TriageClassifier] = None
        self.train_dataset: Optional[RoutingDataset] = None
        self.val_dataset: Optional[RoutingDataset] = None
        self.trainer: Optional[Trainer] = None
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """Get default model configuration."""
        return {
            "model_name": "distilbert-base-uncased",
            "num_actions": 5,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "hidden_dropout": 0.1,
        }
    
    def _get_default_training_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            "num_epochs": 3,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "max_length": 512,
            "early_stopping_patience": 2,
            "eval_steps": 100,
            "save_steps": 500,
            "logging_steps": 50,
        }
    
    async def train(
        self,
        training_data: List[Dict[str, Any]],
        validation_split: float = 0.2,
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the ML model on collected data.
        
        Args:
            training_data: List of training examples
            validation_split: Fraction of data to use for validation
            experiment_name: Name for the experiment (for logging)
            
        Returns:
            Training results and metrics
        """
        print(f"🚀 Starting ML model training with {len(training_data)} examples...")
        
        # Initialize experiment tracking
        if self.use_wandb:
            wandb.init(
                project="open-memory-suite",
                name=experiment_name or f"ml_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={**self.model_config, **self.training_config},
            )
        
        try:
            # Prepare datasets
            train_data, val_data = self._prepare_datasets(training_data, validation_split)
            
            # Initialize model
            self.model = TriageClassifier(**self.model_config)
            
            # Setup training
            self._setup_training()
            
            # Train model
            train_result = self.trainer.train()
            
            # Evaluate model
            eval_results = await self._evaluate_model()
            
            # Save model
            model_path = await self._save_model()
            
            results = {
                "train_loss": train_result.training_loss,
                "eval_results": eval_results,
                "model_path": str(model_path),
                "training_examples": len(training_data),
                "validation_examples": len(val_data),
            }
            
            print(f"✅ Training completed! Model saved to {model_path}")
            print(f"📊 Final results: {eval_results}")
            
            return results
            
        finally:
            if self.use_wandb:
                wandb.finish()
    
    def _prepare_datasets(
        self,
        training_data: List[Dict[str, Any]],
        validation_split: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare training and validation datasets."""
        # Split data
        train_data, val_data = train_test_split(
            training_data,
            test_size=validation_split,
            random_state=42,
            stratify=[example["label"] for example in training_data],
        )
        
        print(f"📊 Dataset split: {len(train_data)} train, {len(val_data)} validation")
        
        # Create PyTorch datasets
        self.train_dataset = RoutingDataset(
            texts=[ex["text"] for ex in train_data],
            labels=[ex["label"] for ex in train_data],
            context_features=[ex["context_features"] for ex in train_data],
            tokenizer=self.tokenizer,
            max_length=self.training_config["max_length"],
        )
        
        self.val_dataset = RoutingDataset(
            texts=[ex["text"] for ex in val_data],
            labels=[ex["label"] for ex in val_data],
            context_features=[ex["context_features"] for ex in val_data],
            tokenizer=self.tokenizer,
            max_length=self.training_config["max_length"],
        )
        
        return train_data, val_data
    
    def _setup_training(self) -> None:
        """Setup the Hugging Face trainer."""
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=self.training_config["num_epochs"],
            per_device_train_batch_size=self.training_config["batch_size"],
            per_device_eval_batch_size=self.training_config["batch_size"],
            learning_rate=self.training_config["learning_rate"],
            warmup_steps=self.training_config["warmup_steps"],
            weight_decay=self.training_config["weight_decay"],
            logging_steps=self.training_config["logging_steps"],
            eval_steps=self.training_config["eval_steps"],
            save_steps=self.training_config["save_steps"],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to="wandb" if self.use_wandb else None,
            remove_unused_columns=False,
        )
        
        # Custom trainer with support for context features
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.training_config["early_stopping_patience"])],
        )
    
    async def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate the trained model."""
        eval_results = self.trainer.evaluate()
        
        # Additional evaluation metrics
        predictions = self.trainer.predict(self.val_dataset)
        y_true = predictions.label_ids
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Detailed classification report
        class_names = list(self.model.action_to_idx.keys())
        class_report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
        
        return {
            "accuracy": eval_results["eval_accuracy"],
            "f1": eval_results["eval_f1"],
            "loss": eval_results["eval_loss"],
            "classification_report": class_report,
        }
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        
        return {"accuracy": accuracy, "f1": f1}
    
    async def _save_model(self) -> Path:
        """Save the trained model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.output_dir / f"ml_policy_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.model_config,
            "training_config": self.training_config,
        }, model_dir / "model.pt")
        
        # Save tokenizer
        tokenizer_dir = model_dir / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_dir)
        
        # Save training info
        with open(model_dir / "training_info.json", "w") as f:
            json.dump({
                "timestamp": timestamp,
                "model_config": self.model_config,
                "training_config": self.training_config,
            }, f, indent=2)
        
        return model_dir


class CustomTrainer(Trainer):
    """Custom trainer that handles context features."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation with context features."""
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            context_features=inputs.get("context_features"),
        )
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs["logits"], labels)
        
        return (loss, outputs) if return_outputs else loss


# Utility functions for data augmentation and analysis

async def augment_training_data(
    examples: List[Dict[str, Any]],
    augmentation_factor: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    Augment training data with variations to improve model robustness.
    
    Args:
        examples: Original training examples
        augmentation_factor: How much to increase the dataset size
        
    Returns:
        Augmented training examples
    """
    augmented = examples.copy()
    target_size = int(len(examples) * augmentation_factor)
    
    while len(augmented) < target_size:
        # Simple augmentation: randomly select and slightly modify examples
        base_example = random.choice(examples)
        
        # Create variation (simple text modifications)
        augmented_text = _apply_text_augmentation(base_example["text"])
        
        augmented_example = base_example.copy()
        augmented_example["text"] = augmented_text
        augmented_example["augmented"] = True
        
        augmented.append(augmented_example)
    
    return augmented


def _apply_text_augmentation(text: str) -> str:
    """Apply simple text augmentation techniques."""
    # Simple augmentations: case changes, punctuation variations
    augmentations = [
        lambda t: t.replace("?", "??"),  # Add extra punctuation
        lambda t: t.replace(".", "..."),  # Elongate sentences
        lambda t: t.replace(" I ", " i ").replace("I'm", "i'm"),  # Case variations
        lambda t: f"{t} (repeated)",  # Add context marker
        lambda t: t.replace("user", "person").replace("assistant", "AI"),  # Role variations
    ]
    
    augmentation = random.choice(augmentations)
    return augmentation(text)