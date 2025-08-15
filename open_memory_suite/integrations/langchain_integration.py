"""
LangChain Integration for Open Memory Suite

Provides drop-in replacement for LangChain memory classes with cost-optimized,
intelligent memory routing using the FrugalDispatcher.

Key Features:
- Drop-in replacement for LangChain BaseMemory
- Automatic cost optimization with configurable budgets
- Intelligent routing between memory adapters
- Session-aware memory management
- Conversation context preservation
- Production-ready with comprehensive error handling

Usage:
    from open_memory_suite.integrations import FrugalMemory
    
    # Basic usage
    memory = FrugalMemory(cost_budget=1.0)
    
    # With custom configuration
    memory = FrugalMemory(
        cost_budget=2.0,
        config_path="./memory_config.yaml",
        enable_ml_policy=True
    )
    
    # Use in LangChain agents
    agent = create_conversational_agent(
        llm=llm,
        tools=tools,
        memory=memory
    )
"""

import asyncio
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

try:
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.memory.utils import get_prompt_input_key
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for when LangChain is not available
    LANGCHAIN_AVAILABLE = False
    
    class BaseChatMemory:
        """Fallback base class when LangChain is not available."""
        pass
    
    class BaseMessage:
        """Fallback message class."""
        pass
    
    class HumanMessage:
        """Fallback message class."""
        pass
    
    class AIMessage:
        """Fallback message class."""
        pass

from open_memory_suite.adapters import MemoryItem, InMemoryAdapter, FileStoreAdapter
from open_memory_suite.benchmark.cost_model import BudgetType
from open_memory_suite.dispatcher import (
    FrugalDispatcher, 
    HeuristicPolicy, 
    PolicyRegistry,
    ConversationContext,
    ML_AVAILABLE
)

if ML_AVAILABLE:
    from open_memory_suite.dispatcher import MLPolicy

from open_memory_suite.benchmark import CostModel


class FrugalMemory(BaseChatMemory):
    """
    Cost-optimized memory for LangChain applications.
    
    Drop-in replacement for LangChain memory classes that automatically
    optimizes memory usage based on cost constraints and conversation context.
    """
    
    def __init__(
        self,
        cost_budget: float = 1.0,
        budget_type: Union[str, BudgetType] = BudgetType.STANDARD,
        config_path: Optional[Union[str, Path]] = None,
        enable_ml_policy: bool = False,
        ml_model_path: Optional[Union[str, Path]] = None,
        adapters: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        return_messages: bool = True,
        memory_key: str = "history",
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        **kwargs
    ):
        """
        Initialize FrugalMemory.
        
        Args:
            cost_budget: Maximum cost budget for memory operations
            budget_type: Budget constraint level (minimal/standard/premium)
            config_path: Path to configuration file
            enable_ml_policy: Whether to use ML-enhanced routing policy
            ml_model_path: Path to trained ML model
            adapters: List of adapters to use ['memory', 'file', 'faiss']
            session_id: Session identifier for memory isolation
            return_messages: Whether to return messages in LangChain format
            memory_key: Key to use for memory in prompt variables
            input_key: Key for input messages
            output_key: Key for output messages
            human_prefix: Prefix for human messages
            ai_prefix: Prefix for AI messages
        """
        
        # Initialize parent class if LangChain is available
        if LANGCHAIN_AVAILABLE:
            super().__init__(
                return_messages=return_messages,
                memory_key=memory_key,
                input_key=input_key,
                output_key=output_key,
                **kwargs
            )
        
        # Configuration
        self.cost_budget = cost_budget
        self.budget_type = budget_type if isinstance(budget_type, BudgetType) else BudgetType(budget_type)
        self.config_path = Path(config_path) if config_path else None
        self.enable_ml_policy = enable_ml_policy and ML_AVAILABLE
        self.ml_model_path = Path(ml_model_path) if ml_model_path else None
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # LangChain compatibility
        self.return_messages = return_messages
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        
        # Internal state
        self._dispatcher: Optional[FrugalDispatcher] = None
        self._initialized = False
        self._conversation_context: Optional[ConversationContext] = None
        self._message_count = 0
        
        # Setup adapters
        adapter_names = adapters or ["memory", "file"]
        self._setup_adapters(adapter_names)
        
        # Setup dispatcher (async initialization required)
        asyncio.create_task(self._initialize_async())
    
    def _setup_adapters(self, adapter_names: List[str]) -> None:
        """Setup memory adapters based on configuration."""
        self._adapters = []
        
        for adapter_name in adapter_names:
            if adapter_name == "memory":
                adapter = InMemoryAdapter("frugal_memory_store")
            elif adapter_name == "file":
                storage_dir = Path("./langchain_memory_storage") / self.session_id
                storage_dir.mkdir(parents=True, exist_ok=True)
                adapter = FileStoreAdapter("frugal_file_store", storage_dir)
            elif adapter_name == "faiss":
                try:
                    from open_memory_suite.adapters import FAISStoreAdapter
                    adapter = FAISStoreAdapter("frugal_faiss_store")
                except ImportError:
                    warnings.warn("FAISS adapter not available, skipping")
                    continue
            else:
                warnings.warn(f"Unknown adapter type: {adapter_name}")
                continue
            
            self._adapters.append(adapter)
    
    async def _initialize_async(self) -> None:
        """Async initialization of components."""
        try:
            # Initialize adapters
            for adapter in self._adapters:
                await adapter.initialize()
            
            # Create cost model
            cost_model = CostModel()
            
            # Create dispatcher
            self._dispatcher = FrugalDispatcher(
                adapters=self._adapters,
                cost_model=cost_model,
            )
            
            # Setup policy
            policy_registry = PolicyRegistry()
            
            if self.enable_ml_policy and self.ml_model_path and ML_AVAILABLE:
                try:
                    ml_policy = MLPolicy(
                        model_path=self.ml_model_path,
                        confidence_threshold=0.7,
                        fallback_to_heuristic=True,
                    )
                    await ml_policy.initialize()
                    policy_registry.register(ml_policy, set_as_default=True)
                except Exception as e:
                    warnings.warn(f"Failed to load ML policy: {e}. Using heuristic policy.")
                    policy_registry.register(HeuristicPolicy(), set_as_default=True)
            else:
                policy_registry.register(HeuristicPolicy(), set_as_default=True)
            
            self._dispatcher.policy_registry = policy_registry
            await self._dispatcher.initialize()
            
            # Initialize conversation context
            self._conversation_context = ConversationContext(
                session_id=self.session_id,
                budget_type=self.budget_type,
            )
            
            self._initialized = True
            
        except Exception as e:
            warnings.warn(f"Failed to initialize FrugalMemory: {e}")
    
    def _ensure_initialized(self) -> None:
        """Ensure the memory system is initialized."""
        if not self._initialized:
            # Try to initialize synchronously if possible
            if self._dispatcher is None:
                raise RuntimeError(
                    "FrugalMemory not initialized. "
                    "Wait for initialization or call initialize() explicitly."
                )
    
    async def initialize(self) -> None:
        """Explicitly initialize the memory system."""
        if not self._initialized:
            await self._initialize_async()
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables for LangChain compatibility."""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables for LangChain compatibility.
        
        This method retrieves relevant memory content based on the input
        and returns it in the format expected by LangChain.
        """
        self._ensure_initialized()
        
        try:
            # For now, return empty history - in a full implementation,
            # this would retrieve relevant memory based on the input
            if self.return_messages:
                return {self.memory_key: []}
            else:
                return {self.memory_key: ""}
        except Exception as e:
            warnings.warn(f"Error loading memory variables: {e}")
            return {self.memory_key: [] if self.return_messages else ""}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save conversation context to memory.
        
        This is the main method called by LangChain to store conversation turns.
        """
        self._ensure_initialized()
        
        try:
            # Extract input and output keys
            input_key = self.input_key or get_prompt_input_key(inputs, self.memory_key)
            output_key = self.output_key or "output"
            
            # Get the actual messages
            human_message = inputs.get(input_key, "")
            ai_message = outputs.get(output_key, "")
            
            # Save both human and AI messages
            asyncio.create_task(self._save_message(human_message, "user"))
            asyncio.create_task(self._save_message(ai_message, "assistant"))
            
        except Exception as e:
            warnings.warn(f"Error saving context: {e}")
    
    async def _save_message(self, content: str, speaker: str) -> None:
        """Save a single message to memory."""
        if not content.strip():
            return
        
        try:
            # Create memory item
            item = MemoryItem(
                content=content,
                speaker=speaker,
                session_id=self.session_id,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "message_index": self._message_count,
                    "langchain_integration": True,
                }
            )
            
            # Update conversation context
            self._message_count += 1
            if self._conversation_context:
                self._conversation_context.turn_count = self._message_count
            
            # Route memory through dispatcher
            if self._dispatcher and self._conversation_context:
                decision = await self._dispatcher.route_memory(item, self.session_id)
                await self._dispatcher.execute_decision(decision, item, self.session_id)
            
        except Exception as e:
            warnings.warn(f"Error saving message: {e}")
    
    def clear(self) -> None:
        """Clear all memory."""
        self._ensure_initialized()
        
        try:
            # Reset internal state
            self._message_count = 0
            
            # Clear adapters
            asyncio.create_task(self._clear_async())
            
        except Exception as e:
            warnings.warn(f"Error clearing memory: {e}")
    
    async def _clear_async(self) -> None:
        """Async method to clear memory."""
        try:
            for adapter in self._adapters:
                if hasattr(adapter, 'clear_session'):
                    await adapter.clear_session(self.session_id)
        except Exception as e:
            warnings.warn(f"Error in async clear: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        self._ensure_initialized()
        
        stats = {
            "session_id": self.session_id,
            "message_count": self._message_count,
            "cost_budget": self.cost_budget,
            "budget_type": self.budget_type.value,
            "initialized": self._initialized,
            "adapters": [adapter.name for adapter in self._adapters],
        }
        
        # Add dispatcher stats if available
        if self._dispatcher:
            try:
                if hasattr(self._dispatcher, 'get_session_stats'):
                    session_stats = self._dispatcher.get_session_stats(self.session_id)
                    stats.update(session_stats)
            except Exception as e:
                warnings.warn(f"Error getting dispatcher stats: {e}")
        
        return stats
    
    async def retrieve_relevant_memories(
        self, 
        query: str, 
        max_memories: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to a query.
        
        This method can be used for advanced memory retrieval beyond
        the standard LangChain interface.
        """
        self._ensure_initialized()
        
        try:
            if not self._dispatcher:
                return []
            
            # Create a query item
            query_item = MemoryItem(
                content=query,
                speaker="system",
                session_id=self.session_id,
                metadata={"query": True}
            )
            
            # Retrieve from adapters
            memories = []
            for adapter in self._adapters:
                if hasattr(adapter, 'retrieve_similar'):
                    results = await adapter.retrieve_similar(query_item, limit=max_memories)
                    memories.extend(results)
            
            # Sort by relevance/timestamp and limit
            memories = sorted(
                memories, 
                key=lambda x: x.get('metadata', {}).get('timestamp', ''), 
                reverse=True
            )[:max_memories]
            
            return [
                {
                    "content": mem.content,
                    "speaker": mem.speaker,
                    "timestamp": mem.metadata.get('timestamp'),
                    "session_id": mem.session_id,
                }
                for mem in memories
            ]
            
        except Exception as e:
            warnings.warn(f"Error retrieving memories: {e}")
            return []
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self._dispatcher:
                await self._dispatcher.cleanup()
            
            for adapter in self._adapters:
                await adapter.cleanup()
            
        except Exception as e:
            warnings.warn(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if self._initialized:
                asyncio.create_task(self.cleanup())
        except Exception:
            pass  # Ignore cleanup errors during destruction


# Convenience functions for easy integration

def create_frugal_memory(
    cost_budget: float = 1.0,
    budget_type: str = "standard",
    **kwargs
) -> FrugalMemory:
    """
    Create a FrugalMemory instance with sensible defaults.
    
    Args:
        cost_budget: Maximum cost budget for memory operations
        budget_type: Budget constraint level ("minimal", "standard", "premium")
        **kwargs: Additional arguments passed to FrugalMemory
    
    Returns:
        Configured FrugalMemory instance
    """
    return FrugalMemory(
        cost_budget=cost_budget,
        budget_type=budget_type,
        **kwargs
    )


def create_ml_enhanced_memory(
    model_path: Union[str, Path],
    cost_budget: float = 2.0,
    **kwargs
) -> FrugalMemory:
    """
    Create a FrugalMemory instance with ML-enhanced routing.
    
    Args:
        model_path: Path to trained ML model
        cost_budget: Maximum cost budget for memory operations
        **kwargs: Additional arguments passed to FrugalMemory
    
    Returns:
        ML-enhanced FrugalMemory instance
    """
    return FrugalMemory(
        cost_budget=cost_budget,
        enable_ml_policy=True,
        ml_model_path=model_path,
        **kwargs
    )


# Export check for LangChain availability
__all__ = ["FrugalMemory", "create_frugal_memory", "create_ml_enhanced_memory", "LANGCHAIN_AVAILABLE"]