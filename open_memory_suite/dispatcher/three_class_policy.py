#!/usr/bin/env python3
"""
3-Class ML Policy for Memory Routing.

This module integrates our trained 3-class XGBoost router with the FrugalDispatcher
policy architecture, implementing the friend's recommendations for simplified, 
reliable routing decisions.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..adapters.base import MemoryAdapter, MemoryItem
from .core import (
    ConversationContext,
    MemoryAction,
    MemoryPolicy,
    Priority,
)

# Import our trained 3-class router
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ml_training.three_class_router import ThreeClassRouter, RoutingDecision
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False


class ThreeClassMLPolicy(MemoryPolicy):
    """
    Production-ready 3-class memory routing policy.
    
    Uses our trained XGBoost + calibrated abstention router to make intelligent
    routing decisions with high reliability and explainability.
    
    Class Mappings:
    - 0 (discard) -> MemoryAction.DROP
    - 1 (store) -> MemoryAction.STORE
    - 2 (compress) -> MemoryAction.SUMMARIZE
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: float = 0.75,
        fallback_to_heuristic: bool = True,
        name: str = "three_class_ml",
        version: str = "1.0"
    ):
        """
        Initialize the 3-class ML policy.
        
        Args:
            model_path: Path to trained 3-class router model
            confidence_threshold: Minimum confidence for ML decisions
            fallback_to_heuristic: Whether to fall back to rules if confidence is low
            name: Policy name for identification
            version: Policy version
        """
        super().__init__(name, version)
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.fallback_to_heuristic = fallback_to_heuristic
        
        # Initialize router
        self.router: Optional[ThreeClassRouter] = None
        
        # Performance tracking
        self._ml_decisions = 0
        self._heuristic_fallbacks = 0
        self._abstentions = 0
        self._total_inference_time = 0.0
        self._decision_distribution = {"discard": 0, "store": 0, "compress": 0}
        
        # Fallback heuristic policy
        if fallback_to_heuristic:
            try:
                from .heuristic_policy import HeuristicPolicy
                self.heuristic_policy = HeuristicPolicy()
            except ImportError:
                self.heuristic_policy = None
        else:
            self.heuristic_policy = None
        
        # Initialize model if available
        if ROUTER_AVAILABLE and model_path and model_path.exists():
            self._load_router()
    
    def _load_router(self) -> None:
        """Load the trained 3-class router."""
        try:
            self.router = ThreeClassRouter(
                confidence_threshold=self.confidence_threshold,
                model_path=self.model_path
            )
            print(f"✅ Loaded 3-class router from {self.model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load 3-class router: {e}")
            if not self.fallback_to_heuristic:
                raise
    
    def _initialize_default_router(self) -> None:
        """Initialize router with default configuration (no trained model)."""
        if not ROUTER_AVAILABLE:
            print("⚠️  3-class router not available - falling back to heuristics")
            return
            
        try:
            self.router = ThreeClassRouter(
                confidence_threshold=self.confidence_threshold,
                model_path=None  # No trained model - will use heuristic fallback
            )
            print("✅ Initialized 3-class router (heuristic mode)")
            
        except Exception as e:
            print(f"❌ Failed to initialize default router: {e}")
    
    async def initialize(self) -> None:
        """Initialize the policy (called by dispatcher)."""
        if self.router is None:
            if self.model_path and self.model_path.exists():
                self._load_router()
            else:
                self._initialize_default_router()
    
    async def decide_action(
        self,
        item: MemoryItem,
        context: ConversationContext,
    ) -> MemoryAction:
        """
        Decide what action to take using 3-class router.
        
        Args:
            item: Memory item to analyze
            context: Conversation context
            
        Returns:
            The action to take (STORE/SUMMARIZE/DROP)
        """
        start_time = time.time()
        
        try:
            # Try ML prediction first
            if self.router is not None:
                action, confidence = await self._router_predict(item, context)
                
                # Use ML decision if confidence is high enough
                if confidence >= self.confidence_threshold:
                    self._ml_decisions += 1
                    self._decision_distribution[self._map_class_name_to_key(action)] += 1
                    self._total_inference_time += time.time() - start_time
                    return self._map_class_to_action(action)
                else:
                    self._abstentions += 1
            
            # Fall back to heuristic policy if available
            if self.heuristic_policy:
                self._heuristic_fallbacks += 1
                return await self.heuristic_policy.decide_action(item, context)
            
            # Final fallback: use simple heuristics matching our 3-class schema
            return self._simple_heuristic_decision(item, context)
            
        except Exception as e:
            print(f"❌ Error in 3-class ML policy decision: {e}")
            # Emergency fallback
            if self.heuristic_policy:
                return await self.heuristic_policy.decide_action(item, context)
            return self._simple_heuristic_decision(item, context)
    
    async def _router_predict(
        self,
        item: MemoryItem,
        context: ConversationContext,
    ) -> Tuple[str, float]:
        """
        Make prediction using the 3-class router.
        
        Args:
            item: Memory item to analyze
            context: Conversation context
            
        Returns:
            Tuple of (class_name, confidence)
        """
        # Prepare metadata for router
        metadata = {
            'speaker': item.speaker,
            'turn_number': context.turn_count,
            'session_length': len(context.recent_turns),
            'time_since_last': 0,  # Could be enhanced with actual timing
            'budget_critical': context.is_budget_critical()
        }
        
        # Get router prediction
        routing_decision = self.router.predict(
            content=item.content,
            metadata=metadata,
            return_reasoning=True
        )
        
        return routing_decision.class_name, routing_decision.confidence
    
    def _map_class_to_action(self, class_name: str) -> MemoryAction:
        """Map 3-class router output to MemoryAction."""
        mapping = {
            'discard': MemoryAction.DROP,
            'store': MemoryAction.STORE,
            'compress': MemoryAction.SUMMARIZE,
        }
        return mapping.get(class_name, MemoryAction.STORE)
    
    def _map_class_name_to_key(self, class_name: str) -> str:
        """Map class name to distribution key."""
        return class_name if class_name in self._decision_distribution else 'store'
    
    def _simple_heuristic_decision(
        self, 
        item: MemoryItem, 
        context: ConversationContext
    ) -> MemoryAction:
        """
        Simple heuristic decision matching our 3-class schema.
        
        This provides a fallback when the ML router is not available.
        """
        content = item.content.strip().lower()
        content_length = len(item.content)
        word_count = len(item.content.split())
        
        # Class 0: Discard (chit-chat, acknowledgments, very short)
        discard_patterns = ['ok', 'thanks', 'yes', 'no', 'hi', 'hello', 'bye', 'sure']
        if (content_length < 10 or 
            word_count < 3 or
            any(pattern in content for pattern in discard_patterns)):
            return MemoryAction.DROP
        
        # Class 2: Compress (long content requiring summarization)
        elif content_length > 500 or word_count > 80:
            return MemoryAction.SUMMARIZE
        
        # Class 1: Store (factual content worth keeping)
        else:
            return MemoryAction.STORE
    
    async def choose_adapter(
        self,
        item: MemoryItem,
        available_adapters: List[MemoryAdapter],
        context: ConversationContext,
    ) -> Optional[MemoryAdapter]:
        """
        Choose adapter based on content analysis and cost optimization.
        
        Uses intelligent adapter selection based on the routing decision
        and content characteristics.
        """
        if not available_adapters:
            return None
        
        # Get routing insight if router is available
        routing_decision = None
        if self.router is not None:
            try:
                metadata = {
                    'speaker': item.speaker,
                    'turn_number': context.turn_count,
                    'session_length': len(context.recent_turns),
                    'budget_critical': context.is_budget_critical()
                }
                routing_decision = self.router.predict(
                    content=item.content,
                    metadata=metadata,
                    return_reasoning=True
                )
            except Exception:
                pass
        
        # Adapter selection logic based on content characteristics
        content_length = len(item.content)
        has_structured_data = any(pattern in item.content.lower() 
                                for pattern in ['email', 'phone', 'date', 'time', 'meeting'])
        is_question = '?' in item.content
        
        # Smart adapter selection
        adapter_preferences = []
        
        # For structured/factual data, prefer vector search
        if has_structured_data or is_question:
            adapter_preferences.extend(['faiss_store', 'vector'])
        
        # For short content, prefer fast in-memory storage
        if content_length < 200:
            adapter_preferences.extend(['memory_store', 'memory'])
        
        # For long content, prefer file-based storage
        if content_length > 1000:
            adapter_preferences.extend(['file_store', 'file'])
        
        # Default preferences
        adapter_preferences.extend(['memory_store', 'faiss_store', 'file_store'])
        
        # Find best available adapter
        available_names = [adapter.name for adapter in available_adapters]
        for preference in adapter_preferences:
            for adapter in available_adapters:
                if (preference in adapter.name.lower() or 
                    preference == adapter.name):
                    return adapter
        
        # Fallback to first available
        return available_adapters[0]
    
    async def analyze_content(self, item: MemoryItem) -> Dict[str, Any]:
        """
        Analyze content to extract features and insights.
        
        Returns rich analysis including routing decision reasoning.
        """
        analysis = {
            'content_length': len(item.content),
            'word_count': len(item.content.split()),
            'has_question': '?' in item.content,
            'has_structured_data': any(pattern in item.content.lower() 
                                     for pattern in ['email', 'phone', '@', 'meeting', 'date']),
            'speaker': item.speaker,
        }
        
        # Add router analysis if available
        if self.router is not None:
            try:
                routing_decision = self.router.predict(
                    content=item.content,
                    metadata={'speaker': item.speaker},
                    return_reasoning=True
                )
                
                analysis.update({
                    'router_decision': routing_decision.class_name,
                    'router_confidence': routing_decision.confidence,
                    'router_reasoning': routing_decision.reasoning,
                    'feature_importance': routing_decision.feature_importance,
                    'abstained': routing_decision.abstained
                })
                
            except Exception as e:
                analysis['router_error'] = str(e)
        
        # Determine priority based on content analysis
        if analysis.get('has_structured_data') or analysis.get('has_question'):
            analysis['priority'] = Priority.HIGH
        elif analysis['content_length'] > 500:
            analysis['priority'] = Priority.MEDIUM  
        else:
            analysis['priority'] = Priority.LOW
        
        return analysis
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive policy performance statistics."""
        base_stats = super().get_stats()
        
        total_decisions = self._ml_decisions + self._heuristic_fallbacks
        
        ml_stats = {
            'ml_decisions': self._ml_decisions,
            'heuristic_fallbacks': self._heuristic_fallbacks,
            'abstentions': self._abstentions,
            'total_decisions': total_decisions,
            'ml_decision_rate': (
                self._ml_decisions / total_decisions if total_decisions > 0 else 0.0
            ),
            'abstention_rate': (
                self._abstentions / (self._ml_decisions + self._abstentions) 
                if (self._ml_decisions + self._abstentions) > 0 else 0.0
            ),
            'avg_inference_time_ms': (
                (self._total_inference_time * 1000 / self._ml_decisions)
                if self._ml_decisions > 0 else 0.0
            ),
            'decision_distribution': self._decision_distribution.copy(),
            'router_loaded': self.router is not None,
            'confidence_threshold': self.confidence_threshold,
            'model_path': str(self.model_path) if self.model_path else None,
        }
        
        # Add router-specific stats if available
        if self.router is not None:
            try:
                router_stats = self.router.get_statistics()
                ml_stats.update({
                    'router_predictions': router_stats.get('predictions_made', 0),
                    'router_abstentions': router_stats.get('abstentions', 0),
                    'training_stats': router_stats.get('training_stats', {}),
                })
            except Exception:
                pass
        
        return {**base_stats, **ml_stats}
    
    async def explain_decision(
        self, 
        item: MemoryItem, 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Provide detailed explanation of routing decision.
        
        This is particularly valuable for the /explain endpoint mentioned
        in our architecture goals.
        """
        explanation = {
            'policy': self.name,
            'version': self.version,
            'timestamp': time.time(),
            'item_summary': {
                'content_length': len(item.content),
                'speaker': item.speaker,
                'content_preview': item.content[:100] + '...' if len(item.content) > 100 else item.content
            },
            'context_summary': {
                'turn_count': context.turn_count,
                'budget_type': context.budget_type.value,
                'budget_critical': context.is_budget_critical(),
                'total_stored': context.total_stored_items
            }
        }
        
        # Add router explanation if available
        if self.router is not None:
            try:
                metadata = {
                    'speaker': item.speaker,
                    'turn_number': context.turn_count,
                    'session_length': len(context.recent_turns),
                    'budget_critical': context.is_budget_critical()
                }
                
                routing_decision = self.router.predict(
                    content=item.content,
                    metadata=metadata,
                    return_reasoning=True
                )
                
                explanation['router_analysis'] = {
                    'predicted_class': routing_decision.class_name,
                    'confidence': routing_decision.confidence,
                    'reasoning': routing_decision.reasoning,
                    'abstained': routing_decision.abstained,
                    'feature_importance': routing_decision.feature_importance,
                    'all_probabilities': routing_decision.all_probabilities
                }
                
                if routing_decision.heuristic_fallback:
                    explanation['heuristic_fallback'] = routing_decision.heuristic_fallback
                
            except Exception as e:
                explanation['router_error'] = str(e)
        
        return explanation
ThreeClassPolicy = ThreeClassMLPolicy