"""Adapter capability registry for dynamic routing and extensibility."""

from typing import Dict, List, Set, Type, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .base import MemoryAdapter

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for memory adapters with capability-based routing."""
    
    _registry: Dict[str, Type["MemoryAdapter"]] = {}
    _capabilities: Dict[str, Set[str]] = {}
    _instances: Dict[str, "MemoryAdapter"] = {}
    
    # Standard capability strings to prevent typos
    CAPABILITY_VECTOR = "vector"
    CAPABILITY_SEMANTIC = "semantic"
    CAPABILITY_PERSISTENT = "persistent"
    CAPABILITY_CHEAP = "cheap"
    CAPABILITY_FAST = "fast"
    CAPABILITY_SCALABLE = "scalable"
    CAPABILITY_SEARCHABLE = "searchable"
    CAPABILITY_TEMPORAL = "temporal"
    
    VALID_CAPABILITIES = {
        CAPABILITY_VECTOR,
        CAPABILITY_SEMANTIC, 
        CAPABILITY_PERSISTENT,
        CAPABILITY_CHEAP,
        CAPABILITY_FAST,
        CAPABILITY_SCALABLE,
        CAPABILITY_SEARCHABLE,
        CAPABILITY_TEMPORAL
    }
    
    @classmethod
    def register(cls, *, capabilities: Set[str]):
        """
        Decorator to register an adapter class with its capabilities.
        
        Args:
            capabilities: Set of capability strings this adapter provides
            
        Returns:
            The decorated adapter class
            
        Example:
            @AdapterRegistry.register(capabilities={"vector", "semantic"})
            class FAISSStoreAdapter(MemoryAdapter):
                pass
        """
        def decorator(adapter_cls: Type["MemoryAdapter"]):
            # Validate capabilities
            invalid_caps = capabilities - cls.VALID_CAPABILITIES
            if invalid_caps:
                logger.warning(f"Unknown capabilities for {adapter_cls.__name__}: {invalid_caps}")
            
            # Register the adapter
            cls._registry[adapter_cls.__name__] = adapter_cls
            cls._capabilities[adapter_cls.__name__] = capabilities.copy()
            
            logger.info(f"Registered adapter {adapter_cls.__name__} with capabilities: {capabilities}")
            return adapter_cls
        
        return decorator
    
    @classmethod
    def by_capability(cls, capability: str) -> List[str]:
        """
        Get adapter names that provide a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of adapter class names that provide this capability
        """
        return [
            name for name, caps in cls._capabilities.items()
            if capability in caps
        ]
    
    @classmethod
    def get_capabilities(cls, adapter_name: str) -> Set[str]:
        """
        Get all capabilities for a specific adapter.
        
        Args:
            adapter_name: Name of the adapter class
            
        Returns:
            Set of capabilities this adapter provides
        """
        return cls._capabilities.get(adapter_name, set())
    
    @classmethod
    def list_all(cls) -> Dict[str, Set[str]]:
        """
        Get all registered adapters and their capabilities.
        
        Returns:
            Dictionary mapping adapter names to their capability sets
        """
        return cls._capabilities.copy()
    
    @classmethod
    def create_adapter(cls, adapter_name: str, instance_name: str, **kwargs) -> "MemoryAdapter":
        """
        Create an instance of a registered adapter.
        
        Args:
            adapter_name: Name of the adapter class
            instance_name: Unique name for this instance
            **kwargs: Arguments to pass to the adapter constructor
            
        Returns:
            New adapter instance
            
        Raises:
            KeyError: If adapter_name is not registered
        """
        if adapter_name not in cls._registry:
            raise KeyError(f"Adapter '{adapter_name}' not registered. Available: {list(cls._registry.keys())}")
        
        adapter_cls = cls._registry[adapter_name]
        instance = adapter_cls(name=instance_name, **kwargs)
        cls._instances[instance_name] = instance
        
        logger.info(f"Created adapter instance '{instance_name}' of type {adapter_name}")
        return instance
    
    @classmethod
    def get_instance(cls, instance_name: str) -> "MemoryAdapter":
        """
        Get a previously created adapter instance.
        
        Args:
            instance_name: Name of the adapter instance
            
        Returns:
            The adapter instance
            
        Raises:
            KeyError: If instance_name is not found
        """
        if instance_name not in cls._instances:
            raise KeyError(f"Adapter instance '{instance_name}' not found. Available: {list(cls._instances.keys())}")
        
        return cls._instances[instance_name]
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered adapters and instances. Mainly for testing."""
        cls._registry.clear()
        cls._capabilities.clear()
        cls._instances.clear()
        logger.info("Cleared adapter registry")
    
    @classmethod
    def has_capability_combination(cls, required_capabilities: Set[str]) -> List[str]:
        """
        Find adapters that provide ALL of the required capabilities.
        
        Args:
            required_capabilities: Set of capabilities that must all be present
            
        Returns:
            List of adapter names that provide all required capabilities
        """
        matching_adapters = []
        for name, caps in cls._capabilities.items():
            if required_capabilities.issubset(caps):
                matching_adapters.append(name)
        return matching_adapters