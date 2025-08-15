"""
Machine Learning-enhanced memory routing policy using DistilBERT + LoRA.

This module implements the M3 milestone: ML-enhanced dispatcher that learns
better routing decisions than rule-based heuristics through fine-tuned transformers.
"""

import asyncio
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from ..adapters.base import MemoryAdapter, MemoryItem
from .core import (
    ConversationContext,
    MemoryAction,
    MemoryPolicy,
    Priority,
    ContentType,
)


class TriageClassifier(nn.Module):
    """
    DistilBERT-based classifier for memory routing decisions.
    
    Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning,
    enabling fast training and inference while maintaining model quality.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_actions: int = 5,  # store_memory, store_faiss, store_file, summarize, drop
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        """
        Initialize the triage classifier.
        
        Args:
            model_name: Base transformer model to use
            num_actions: Number of routing actions to classify
            lora_r: LoRA rank parameter (lower = more efficient)
            lora_alpha: LoRA alpha parameter (scaling factor)
            lora_dropout: Dropout rate for LoRA layers
            hidden_dropout: Dropout rate for hidden layers
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_actions = num_actions
        
        # Load base model
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],  # DistilBERT attention layers
            bias="none",
        )
        
        # Apply LoRA to the backbone
        self.backbone = get_peft_model(self.backbone, lora_config)
        
        # Classification head
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(hidden_size, num_actions)
        
        # Context feature embedding (for conversation context)
        self.context_embedding = nn.Linear(10, 64)  # 10 context features → 64 dim
        self.context_dropout = nn.Dropout(hidden_dropout)
        
        # Combined classifier (text + context)
        self.combined_classifier = nn.Linear(hidden_size + 64, num_actions)
        
        # Store action mappings
        self.action_to_idx = {
            MemoryAction.STORE: 0,  # Generic store - adapter chosen separately
            "store_faiss": 1,       # Specific adapter routing
            "store_file": 2,
            MemoryAction.SUMMARIZE: 3,
            MemoryAction.DROP: 4,
        }
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask for padding
            context_features: Optional conversation context features
            
        Returns:
            Dictionary with logits and hidden states
        """
        # Get transformer outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Use [CLS] token representation
        cls_hidden = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        cls_hidden = self.dropout(cls_hidden)
        
        if context_features is not None:
            # Embed context features
            context_emb = self.context_embedding(context_features)
            context_emb = self.context_dropout(context_emb)
            
            # Combine text and context representations
            combined = torch.cat([cls_hidden, context_emb], dim=-1)
            logits = self.combined_classifier(combined)
        else:
            # Text-only classification
            logits = self.classifier(cls_hidden)
        
        return {
            "logits": logits,
            "hidden_states": cls_hidden,
            "context_embedding": context_emb if context_features is not None else None,
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_features: Optional[torch.Tensor] = None,
    ) -> Tuple[str, float]:
        """
        Make a prediction for routing decision.
        
        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask
            context_features: Optional context features
            
        Returns:
            Tuple of (predicted_action, confidence_score)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, context_features)
            logits = outputs["logits"]
            
            # Get probabilities and prediction
            probs = torch.softmax(logits, dim=-1)
            predicted_idx = torch.argmax(probs, dim=-1).item()
            confidence = torch.max(probs, dim=-1).values.item()
            
            predicted_action = self.idx_to_action[predicted_idx]
            
        return predicted_action, confidence


class MLPolicy(MemoryPolicy):
    """
    Machine learning-enhanced memory routing policy.
    
    This policy uses a fine-tuned DistilBERT model to make intelligent routing
    decisions based on content analysis and conversation context. It represents
    the evolution from rule-based heuristics to learned patterns.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        confidence_threshold: float = 0.7,
        fallback_to_heuristic: bool = True,
        name: str = "ml_enhanced",
        version: str = "1.0",
    ):
        """
        Initialize the ML-enhanced policy.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on (auto-detect if None)
            max_length: Maximum token length for input text
            confidence_threshold: Minimum confidence for ML decisions
            fallback_to_heuristic: Whether to fall back to rules if confidence is low
            name: Policy name for identification
            version: Policy version
        """
        super().__init__(name, version)
        
        self.model_path = model_path
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        self.fallback_to_heuristic = fallback_to_heuristic
        
        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Model components
        self.model: Optional[TriageClassifier] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
        # Fallback heuristic policy
        if fallback_to_heuristic:
            from .heuristic_policy import HeuristicPolicy
            self.heuristic_policy = HeuristicPolicy()
        else:
            self.heuristic_policy = None
        
        # Performance tracking
        self._ml_decisions = 0
        self._heuristic_fallbacks = 0
        self._total_inference_time = 0.0
        
        # Initialize model if path provided
        if model_path and model_path.exists():
            asyncio.create_task(self._load_model())
    
    async def initialize(self) -> None:
        """Initialize the ML model and tokenizer."""
        if self.model_path and self.model_path.exists():
            await self._load_model()
        else:
            # Initialize with pre-trained weights (no fine-tuning yet)
            await self._initialize_pretrained()
    
    async def _load_model(self) -> None:
        """Load a trained model from checkpoint."""
        try:
            # Load model state
            checkpoint = torch.load(
                self.model_path / "model.pt",
                map_location=self.device,
                weights_only=True,
            )
            
            # Initialize model with saved config
            model_config = checkpoint.get("config", {})
            self.model = TriageClassifier(**model_config)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer
            tokenizer_path = self.model_path / "tokenizer"
            if tokenizer_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            
            print(f"✅ Loaded ML model from {self.model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load ML model: {e}")
            if self.fallback_to_heuristic:
                print("📝 Falling back to heuristic policy")
            else:
                raise
    
    async def _initialize_pretrained(self) -> None:
        """Initialize with pre-trained model (no fine-tuning)."""
        try:
            self.model = TriageClassifier()
            self.model.to(self.device)
            self.model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            
            print("✅ Initialized pre-trained ML model (no fine-tuning)")
            
        except Exception as e:
            print(f"❌ Failed to initialize pre-trained model: {e}")
            if not self.fallback_to_heuristic:
                raise
    
    async def decide_action(
        self,
        item: MemoryItem,
        context: ConversationContext,
    ) -> MemoryAction:
        """
        Decide what action to take using ML model with heuristic fallback.
        
        Args:
            item: Memory item to analyze
            context: Conversation context
            
        Returns:
            The action to take (store/summarize/drop/defer)
        """
        start_time = time.time()
        
        try:
            # Try ML prediction first
            if self.model is not None and self.tokenizer is not None:
                action, confidence = await self._ml_predict(item, context)
                
                # Use ML decision if confidence is high enough
                if confidence >= self.confidence_threshold:
                    self._ml_decisions += 1
                    self._total_inference_time += time.time() - start_time
                    return action
            
            # Fall back to heuristic policy if available
            if self.heuristic_policy:
                self._heuristic_fallbacks += 1
                return await self.heuristic_policy.decide_action(item, context)
            
            # Default fallback: conservative store decision
            return MemoryAction.STORE
            
        except Exception as e:
            print(f"❌ Error in ML policy decision: {e}")
            # Emergency fallback
            if self.heuristic_policy:
                return await self.heuristic_policy.decide_action(item, context)
            return MemoryAction.STORE
    
    async def _ml_predict(
        self,
        item: MemoryItem,
        context: ConversationContext,
    ) -> Tuple[MemoryAction, float]:
        """
        Make ML prediction for routing decision.
        
        Args:
            item: Memory item to analyze
            context: Conversation context
            
        Returns:
            Tuple of (predicted_action, confidence)
        """
        # Prepare input text
        input_text = self._prepare_input_text(item, context)
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Prepare context features
        context_features = self._extract_context_features(item, context)
        if context_features is not None:
            context_features = torch.tensor(context_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Make prediction
        predicted_action, confidence = self.model.predict(
            input_ids, attention_mask, context_features
        )
        
        # Convert specific adapter routing to generic store action
        # (adapter selection is handled separately)
        if predicted_action in ["store_faiss", "store_file"]:
            predicted_action = MemoryAction.STORE
        
        return MemoryAction(predicted_action), confidence
    
    def _prepare_input_text(self, item: MemoryItem, context: ConversationContext) -> str:
        """
        Prepare input text for the model including context, with smart truncation.
        
        Args:
            item: Memory item
            context: Conversation context
            
        Returns:
            Formatted input text that fits within model's max_length
        """
        # Include speaker information
        speaker_prefix = f"[{item.speaker or 'unknown'}]: " if item.speaker else ""
        
        # Include recent context (last 2 turns for brevity)
        recent_context = ""
        if context.recent_turns:
            recent_turns = context.recent_turns[-2:]
            recent_context = " | ".join([
                f"[{turn.speaker or 'unknown'}]: {turn.content[:100]}"
                for turn in recent_turns
            ])
            recent_context = f"Context: {recent_context} | "
        
        # Include conversation metadata
        metadata = f"Turn {context.turn_count}, Budget: {context.budget_type.value}"
        
        # Build prefix (everything except item content)
        prefix = f"{recent_context}{metadata} | Current: {speaker_prefix}"
        
        # Calculate remaining tokens available for item content
        if self.tokenizer is not None:
            # Reserve tokens for special tokens and padding
            reserved_tokens = 10
            prefix_tokens = len(self.tokenizer.encode(prefix, add_special_tokens=False))
            available_tokens = max(50, self.max_length - prefix_tokens - reserved_tokens)
            
            # Truncate item content to fit available tokens
            item_tokens = self.tokenizer.encode(item.content, add_special_tokens=False)
            if len(item_tokens) > available_tokens:
                # Keep the beginning and end of content for better context
                keep_start = available_tokens * 2 // 3  # 2/3 from start
                keep_end = available_tokens - keep_start  # 1/3 from end
                
                if keep_end > 0 and len(item_tokens) > keep_start + keep_end:
                    truncated_tokens = (
                        item_tokens[:keep_start] + 
                        [self.tokenizer.encode("...", add_special_tokens=False)[0]] +  # ellipsis
                        item_tokens[-keep_end:]
                    )
                else:
                    truncated_tokens = item_tokens[:available_tokens]
                
                truncated_content = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            else:
                truncated_content = item.content
        else:
            # Fallback if tokenizer not available - simple character truncation
            available_chars = max(100, (self.max_length - len(prefix)) * 4)  # rough estimate
            if len(item.content) > available_chars:
                keep_start = available_chars * 2 // 3
                keep_end = available_chars - keep_start - 3  # account for "..."
                if keep_end > 0:
                    truncated_content = item.content[:keep_start] + "..." + item.content[-keep_end:]
                else:
                    truncated_content = item.content[:available_chars]
            else:
                truncated_content = item.content
        
        # Combine prefix and (possibly truncated) content
        input_text = f"{prefix}{truncated_content}"
        
        return input_text
    
    def _extract_context_features(
        self,
        item: MemoryItem,
        context: ConversationContext,
    ) -> Optional[List[float]]:
        """
        Extract numerical context features for the model.
        
        Args:
            item: Memory item
            context: Conversation context
            
        Returns:
            List of numerical features
        """
        try:
            # Extract basic features
            features = [
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
            
            return features
        except Exception:
            return None
    
    async def choose_adapter(
        self,
        item: MemoryItem,
        available_adapters: List[MemoryAdapter],
        context: ConversationContext,
    ) -> Optional[MemoryAdapter]:
        """
        Choose adapter using ML model or fallback to heuristic.
        
        Args:
            item: Memory item to store
            available_adapters: Available adapters
            context: Conversation context
            
        Returns:
            Selected adapter
        """
        # For now, delegate to heuristic policy for adapter selection
        # Future enhancement: train a separate model for adapter selection
        if self.heuristic_policy:
            return await self.heuristic_policy.choose_adapter(
                item, available_adapters, context
            )
        
        # Simple fallback: choose first available adapter
        return available_adapters[0] if available_adapters else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ML policy performance statistics."""
        base_stats = super().get_stats()
        
        ml_stats = {
            "ml_decisions": self._ml_decisions,
            "heuristic_fallbacks": self._heuristic_fallbacks,
            "avg_inference_time_ms": (
                (self._total_inference_time * 1000 / self._ml_decisions)
                if self._ml_decisions > 0 else 0.0
            ),
            "ml_decision_rate": (
                self._ml_decisions / (self._ml_decisions + self._heuristic_fallbacks)
                if (self._ml_decisions + self._heuristic_fallbacks) > 0 else 0.0
            ),
            "device": self.device,
            "model_loaded": self.model is not None,
        }
        
        return {**base_stats, **ml_stats}