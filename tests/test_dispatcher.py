"""Tests for the frugal memory dispatcher system."""

import pytest
from datetime import datetime
from typing import List

from open_memory_suite.adapters.base import MemoryAdapter, MemoryItem
from open_memory_suite.adapters.memory_store import InMemoryAdapter
from open_memory_suite.adapters.faiss_store import FAISStoreAdapter
from open_memory_suite.adapters.file_store import FileStoreAdapter
from open_memory_suite.benchmark.cost_model import BudgetType, CostModel
from open_memory_suite.dispatcher import (
    ConversationContext,
    ContentType,
    FrugalDispatcher,
    HeuristicPolicy,
    MemoryAction,
    PolicyRegistry,
    Priority,
    RoutingDecision,
)


class TestHeuristicPolicy:
    """Test the rule-based heuristic policy."""
    
    @pytest.fixture
    def policy(self):
        """Create a heuristic policy for testing."""
        return HeuristicPolicy()
    
    @pytest.fixture
    def context(self):
        """Create a basic conversation context."""
        return ConversationContext(
            session_id="test_session",
            turn_count=5,
            budget_type=BudgetType.STANDARD
        )
    
    @pytest.fixture
    def adapters(self):
        """Create test adapters."""
        return [
            InMemoryAdapter("memory_store"),
            FAISStoreAdapter("faiss_store", embedding_model="all-MiniLM-L6-v2", dimension=384),
            FileStoreAdapter("file_store")
        ]
    
    @pytest.mark.asyncio
    async def test_drop_acknowledgments(self, policy, context):
        """Test that simple acknowledgments are dropped."""
        acknowledgments = [
            MemoryItem(content="ok", speaker="user"),
            MemoryItem(content="thanks", speaker="assistant"),
            MemoryItem(content="got it", speaker="user"),
            MemoryItem(content="👍", speaker="user"),
        ]
        
        for item in acknowledgments:
            action = await policy.decide_action(item, context)
            assert action == MemoryAction.DROP, f"Should drop acknowledgment: {item.content}"
    
    @pytest.mark.asyncio
    async def test_store_user_questions(self, policy, context):
        """Test that user questions are always stored."""
        questions = [
            MemoryItem(content="What is the weather like?", speaker="user"),
            MemoryItem(content="How do I fix this problem?", speaker="user"),
            MemoryItem(content="Can you help me with this?", speaker="user"),
            MemoryItem(content="Where is the nearest restaurant?", speaker="user"),
        ]
        
        for item in questions:
            action = await policy.decide_action(item, context)
            assert action == MemoryAction.STORE, f"Should store user question: {item.content}"
    
    @pytest.mark.asyncio
    async def test_store_factual_information(self, policy, context):
        """Test that factual information is stored."""
        factual_items = [
            MemoryItem(content="My name is John Smith", speaker="user"),
            MemoryItem(content="I was born on January 15, 1990", speaker="user"),
            MemoryItem(content="My phone number is 555-1234", speaker="user"),
            MemoryItem(content="The meeting is scheduled for 12/25/2023", speaker="assistant"),
        ]
        
        for item in factual_items:
            action = await policy.decide_action(item, context)
            assert action == MemoryAction.STORE, f"Should store factual info: {item.content}"
    
    @pytest.mark.asyncio
    async def test_summarize_long_content(self, policy, context):
        """Test that very long content is summarized."""
        long_content = " ".join(["This is a very long piece of content that contains many words and sentences."] * 20)
        item = MemoryItem(content=long_content, speaker="assistant")
        
        action = await policy.decide_action(item, context)
        assert action == MemoryAction.SUMMARIZE, "Should summarize long content"
    
    @pytest.mark.asyncio
    async def test_budget_critical_behavior(self, policy):
        """Test behavior when budget is critical."""
        # Create context with critical budget
        context = ConversationContext(
            session_id="test_session",
            budget_type=BudgetType.MINIMAL,
            budget_exhausted=True
        )
        
        # Low-value content should be dropped
        low_value = MemoryItem(content="This is just general conversation", speaker="assistant")
        action = await policy.decide_action(low_value, context)
        assert action == MemoryAction.DROP, "Should drop low-value content when budget critical"
        
        # High-value content should still be stored
        high_value = MemoryItem(content="What is your name?", speaker="user")
        action = await policy.decide_action(high_value, context)
        assert action == MemoryAction.STORE, "Should store high-value content even when budget critical"
    
    @pytest.mark.asyncio
    async def test_adapter_selection_factual_content(self, policy, context, adapters):
        """Test that factual content is routed to FAISS."""
        await self._initialize_adapters(adapters)
        
        factual_item = MemoryItem(content="My name is Alice Johnson", speaker="user")
        adapter = await policy.choose_adapter(factual_item, adapters, context)
        
        assert adapter is not None
        assert adapter.name == "faiss_store", "Factual content should go to FAISS"
        
        await self._cleanup_adapters(adapters)
    
    @pytest.mark.asyncio
    async def test_adapter_selection_recent_conversation(self, policy, adapters):
        """Test that recent conversation goes to InMemory."""
        await self._initialize_adapters(adapters)
        
        # Recent conversation context (turn_count <= 10)
        context = ConversationContext(
            session_id="test_session",
            turn_count=3,
            budget_type=BudgetType.STANDARD
        )
        
        conversational_item = MemoryItem(content="How was your day?", speaker="user")
        adapter = await policy.choose_adapter(conversational_item, adapters, context)
        
        assert adapter is not None
        assert adapter.name == "memory_store", "Recent conversation should go to InMemory"
        
        await self._cleanup_adapters(adapters)
    
    @pytest.mark.asyncio
    async def test_adapter_selection_budget_critical(self, policy, adapters):
        """Test that budget critical situations choose cheapest adapter."""
        await self._initialize_adapters(adapters)
        
        # Budget critical context
        context = ConversationContext(
            session_id="test_session",
            budget_type=BudgetType.MINIMAL,
            budget_exhausted=True
        )
        
        item = MemoryItem(content="Some content to store", speaker="user")
        adapter = await policy.choose_adapter(item, adapters, context)
        
        assert adapter is not None
        assert adapter.name == "file_store", "Budget critical should choose cheapest adapter"
        
        await self._cleanup_adapters(adapters)
    
    @pytest.mark.asyncio
    async def test_content_analysis(self, policy):
        """Test detailed content analysis."""
        # Test factual content
        factual_item = MemoryItem(content="My name is John and I live in New York", speaker="user")
        analysis = await policy.analyze_content(factual_item)
        
        assert analysis["is_factual"] == True
        assert analysis["has_names"] == True
        assert analysis["content_type"] == ContentType.FACTUAL
        assert analysis["priority"] == Priority.HIGH
        
        # Test question content
        question_item = MemoryItem(content="What is the weather like today?", speaker="user")
        analysis = await policy.analyze_content(question_item)
        
        assert analysis["is_query"] == True
        assert analysis["content_type"] == ContentType.QUERY
        assert analysis["priority"] == Priority.CRITICAL
    
    async def _initialize_adapters(self, adapters: List[MemoryAdapter]):
        """Helper to initialize test adapters."""
        for adapter in adapters:
            await adapter.initialize()
    
    async def _cleanup_adapters(self, adapters: List[MemoryAdapter]):
        """Helper to cleanup test adapters."""
        for adapter in adapters:
            await adapter.cleanup()


class TestFrugalDispatcher:
    """Test the frugal dispatcher system."""
    
    @pytest.fixture
    async def adapters(self):
        """Create and initialize test adapters."""
        adapters = [
            InMemoryAdapter("memory_store"),
            FAISStoreAdapter("faiss_store", embedding_model="all-MiniLM-L6-v2", dimension=384),
            FileStoreAdapter("file_store")
        ]
        
        for adapter in adapters:
            await adapter.initialize()
        
        yield adapters
        
        # Cleanup
        for adapter in adapters:
            await adapter.cleanup()
    
    @pytest.fixture
    def policy_registry(self):
        """Create policy registry with heuristic policy."""
        registry = PolicyRegistry()
        policy = HeuristicPolicy()
        registry.register(policy, set_as_default=True)
        return registry
    
    @pytest.fixture
    async def dispatcher(self, adapters, policy_registry):
        """Create and initialize frugal dispatcher."""
        cost_model = CostModel()
        dispatcher = FrugalDispatcher(
            adapters=adapters,
            cost_model=cost_model,
            policy_registry=policy_registry,
            default_budget=BudgetType.STANDARD
        )
        
        await dispatcher.initialize()
        yield dispatcher
        await dispatcher.cleanup()
    
    @pytest.mark.asyncio
    async def test_dispatcher_initialization(self, dispatcher):
        """Test that dispatcher initializes correctly."""
        assert len(dispatcher.adapters) == 3
        assert "memory_store" in dispatcher.adapters
        assert "faiss_store" in dispatcher.adapters
        assert "file_store" in dispatcher.adapters
    
    @pytest.mark.asyncio
    async def test_route_memory_basic(self, dispatcher):
        """Test basic memory routing functionality."""
        item = MemoryItem(content="What is your name?", speaker="user")
        
        decision = await dispatcher.route_memory(item, "test_session")
        
        assert isinstance(decision, RoutingDecision)
        assert decision.action == MemoryAction.STORE
        assert decision.selected_adapter is not None
        assert decision.reasoning is not None
        assert decision.confidence > 0
    
    @pytest.mark.asyncio
    async def test_route_and_execute(self, dispatcher):
        """Test routing and executing a decision."""
        item = MemoryItem(content="My name is Alice", speaker="user")
        
        decision, success = await dispatcher.route_and_execute(item, "test_session")
        
        assert isinstance(decision, RoutingDecision)
        assert decision.action == MemoryAction.STORE
        assert success == True, "Execution should succeed"
        
        # Verify item was actually stored
        adapter = dispatcher.adapters[decision.selected_adapter]
        count = await adapter.count()
        assert count > 0, "Item should be stored in adapter"
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, dispatcher):
        """Test batch processing of multiple items."""
        items = [
            MemoryItem(content="What is AI?", speaker="user"),
            MemoryItem(content="Thanks for the help", speaker="user"),
            MemoryItem(content="My phone number is 555-1234", speaker="user"),
        ]
        
        results = await dispatcher.batch_route_and_execute(items, "test_session")
        
        assert len(results) == 3
        for decision, success in results:
            assert isinstance(decision, RoutingDecision)
            # All should have valid decisions, but acknowledgment might be dropped
            assert decision.action in [MemoryAction.STORE, MemoryAction.DROP]
    
    @pytest.mark.asyncio
    async def test_context_management(self, dispatcher):
        """Test conversation context management."""
        session_id = "test_context_session"
        
        # First interaction - should create context
        context1 = await dispatcher.get_or_create_context(session_id)
        assert context1.session_id == session_id
        assert context1.turn_count == 0
        
        # Second call - should return same context
        context2 = await dispatcher.get_or_create_context(session_id)
        assert context2 is context1
        
        # Route some items to update context
        item = MemoryItem(content="Hello world", speaker="user")
        await dispatcher.route_memory(item, session_id)
        
        # Context should be updated
        updated_context = await dispatcher.get_or_create_context(session_id)
        assert updated_context.turn_count > 0
    
    @pytest.mark.asyncio
    async def test_retrieval_integration(self, dispatcher):
        """Test memory retrieval across adapters."""
        # Store some items first
        items = [
            MemoryItem(content="My favorite color is blue", speaker="user"),
            MemoryItem(content="I work as a software engineer", speaker="user"),
            MemoryItem(content="I live in San Francisco", speaker="user"),
        ]
        
        session_id = "retrieval_test_session"
        for item in items:
            await dispatcher.route_and_execute(item, session_id)
        
        # Retrieve relevant memories
        result = await dispatcher.retrieve_memories(
            query="Tell me about the user's job",
            session_id=session_id,
            k=3
        )
        
        assert len(result.items) > 0
        assert result.query == "Tell me about the user's job"
        # Should retrieve the software engineer item
        contents = [item.content for item in result.items]
        assert any("software engineer" in content for content in contents)
    
    @pytest.mark.asyncio
    async def test_cost_tracking(self, dispatcher):
        """Test that costs are tracked properly."""
        session_id = "cost_test_session"
        
        # Store an item that should have cost
        item = MemoryItem(content="This is important information", speaker="user")
        decision, success = await dispatcher.route_and_execute(item, session_id)
        
        assert success
        assert decision.estimated_cost is not None
        assert decision.estimated_cost.total_cost >= 0
        
        # Check session summary includes cost information
        summary = await dispatcher.get_session_summary(session_id)
        assert summary is not None
        assert "session_id" in summary
        assert "cost_summary" in summary
    
    @pytest.mark.asyncio
    async def test_stats_collection(self, dispatcher):
        """Test that dispatcher collects performance statistics."""
        # Route some items to generate stats
        items = [
            MemoryItem(content="Store this", speaker="user"),
            MemoryItem(content="ok", speaker="user"),  # Should be dropped
        ]
        
        for item in items:
            await dispatcher.route_and_execute(item, "stats_test_session")
        
        stats = dispatcher.get_stats()
        
        assert "dispatcher_stats" in stats
        assert stats["dispatcher_stats"]["total_routing_decisions"] >= 2
        assert stats["dispatcher_stats"]["items_stored"] >= 1
        assert stats["dispatcher_stats"]["items_dropped"] >= 1


class TestPolicyRegistry:
    """Test the policy registry system."""
    
    def test_policy_registration(self):
        """Test registering and retrieving policies."""
        registry = PolicyRegistry()
        policy = HeuristicPolicy()
        
        registry.register(policy, set_as_default=True)
        
        assert registry.get_policy() is policy
        assert registry.get_policy("heuristic_v1") is policy
        assert "heuristic_v1" in registry.list_policies()
    
    def test_multiple_policies(self):
        """Test managing multiple policies."""
        registry = PolicyRegistry()
        
        policy1 = HeuristicPolicy("policy_1")
        policy2 = HeuristicPolicy("policy_2")
        
        registry.register(policy1, set_as_default=True)
        registry.register(policy2)
        
        assert registry.get_policy() is policy1  # Default
        assert registry.get_policy("policy_1") is policy1
        assert registry.get_policy("policy_2") is policy2
        assert len(registry.list_policies()) == 2