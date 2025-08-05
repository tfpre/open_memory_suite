"""Frugal memory dispatcher - intelligent cost-aware memory routing."""

from .core import (
    ConversationContext,
    ContentType,
    MemoryAction,
    MemoryPolicy,
    PolicyRegistry,
    Priority,
    RoutingDecision,
)
from .frugal_dispatcher import FrugalDispatcher
from .heuristic_policy import HeuristicPolicy

# ML components (optional imports to handle missing dependencies gracefully)
try:
    from .ml_policy import MLPolicy, TriageClassifier
    from .ml_training import MLTrainer, DataCollector, augment_training_data
    
    ML_AVAILABLE = True
    __all__ = [
        "ConversationContext",
        "ContentType", 
        "DataCollector",
        "FrugalDispatcher",
        "HeuristicPolicy",
        "MemoryAction",
        "MemoryPolicy",
        "MLPolicy",
        "MLTrainer",
        "PolicyRegistry",
        "Priority",
        "RoutingDecision",
        "TriageClassifier",
        "augment_training_data",
    ]
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML components not available: {e}")
    __all__ = [
        "ConversationContext",
        "ContentType", 
        "FrugalDispatcher",
        "HeuristicPolicy",
        "MemoryAction",
        "MemoryPolicy",
        "PolicyRegistry",
        "Priority",
        "RoutingDecision",
    ]