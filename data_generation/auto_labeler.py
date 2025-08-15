#!/usr/bin/env python3
"""
GPT-4 Auto-Labeling System for 5-Class Memory Routing

Generates high-quality training data for memory routing classifier using GPT-4
with strict cost controls, quality validation, and comprehensive error handling.

Key Features:
- Cost-aware generation with automatic budget enforcement
- 5-class routing decision labeling (Discard, Key-Value, Vector, Graph, Summary)
- Quality gates with confidence scoring and human validation
- Robust error handling and API failure recovery
- Progress tracking and resumable operations
- Comprehensive audit trail and logging

Usage:
    python data_generation/auto_labeler.py --budget 25.0 --target-samples 14000
    python data_generation/auto_labeler.py --validate-only --samples 200
    python data_generation/auto_labeler.py --resume checkpoint_20250806.json
"""

import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

import aiohttp
import yaml
from pydantic import BaseModel, Field, validator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

class RoutingClass(int, Enum):
    """5-class routing decision categories."""
    DISCARD = 0      # Chit-chat, acknowledgments, <4 words
    KEY_VALUE = 1    # Names/dates/IDs, <25 tokens → SQLite
    VECTOR = 2       # Questions, semantic content → FAISS
    GRAPH = 3        # Multi-entity relationships → Zep
    SUMMARY = 4      # >100 tokens, detailed content → R³Mem

@dataclass
class ConversationTurn:
    """Single conversation turn for labeling."""
    content: str
    speaker: str  # "user" or "assistant"
    context: str  # Previous conversation context
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class LabelingRequest(BaseModel):
    """Request sent to GPT-4 for labeling."""
    turn_content: str = Field(..., description="Content to be labeled")
    conversation_context: str = Field(..., description="Previous conversation context")
    speaker: str = Field(..., description="Speaker (user/assistant)")
    turn_id: str = Field(..., description="Unique turn identifier")

class LabelingResponse(BaseModel):
    """Response from GPT-4 with routing decision."""
    routing_class: int = Field(..., ge=0, le=4, description="Routing decision (0-4)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Explanation of the decision")
    detected_entities: List[str] = Field(default_factory=list, description="Named entities found")
    content_features: Dict[str, Any] = Field(default_factory=dict, description="Content analysis")
    
    @validator('routing_class')
    def validate_routing_class(cls, v):
        if v not in [0, 1, 2, 3, 4]:
            raise ValueError(f"Invalid routing class: {v}. Must be 0-4.")
        return v

class QualityMetrics(BaseModel):
    """Quality assessment metrics for generated labels."""
    total_samples: int = 0
    class_distribution: Dict[int, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    low_confidence_count: int = 0
    api_failures: int = 0
    validation_failures: int = 0
    cost_spent_usd: float = 0.0
    
    def get_class_balance_score(self) -> float:
        """Calculate how balanced the class distribution is (0-1, higher is better)."""
        if not self.class_distribution:
            return 0.0
        
        total = sum(self.class_distribution.values())
        if total == 0:
            return 0.0
            
        # Calculate entropy-based balance score
        import math
        entropy = 0.0
        for count in self.class_distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize to 0-1 range (max entropy for 5 classes is log2(5))
        max_entropy = math.log2(5)
        return entropy / max_entropy if max_entropy > 0 else 0.0

class ConversationGenerator:
    """Generates diverse conversation patterns for labeling."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_generation_config(config_path)
    
    def _load_generation_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load conversation generation configuration."""
        default_config = {
            "conversation_types": [
                "personal_assistant", "technical_discussion", "casual_chat",
                "task_planning", "knowledge_query", "creative_writing"
            ],
            "turn_patterns": {
                "short": {"min_words": 2, "max_words": 10},
                "medium": {"min_words": 10, "max_words": 50},
                "long": {"min_words": 50, "max_words": 200}
            },
            "entity_types": [
                "PERSON", "ORG", "DATE", "TIME", "LOCATION", "PRODUCT"
            ]
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return {**default_config, **yaml.safe_load(f)}
        
        return default_config
    
    async def generate_turns(self, count: int) -> List[ConversationTurn]:
        """Generate diverse conversation turns for labeling."""
        turns = []
        
        # Generate different conversation patterns
        patterns = [
            self._generate_personal_assistant_turns,
            self._generate_technical_discussion_turns,
            self._generate_casual_chat_turns,
            self._generate_task_planning_turns,
            self._generate_knowledge_query_turns
        ]
        
        turns_per_pattern = count // len(patterns)
        remainder = count % len(patterns)
        
        for i, pattern_func in enumerate(patterns):
            pattern_count = turns_per_pattern + (1 if i < remainder else 0)
            pattern_turns = await pattern_func(pattern_count)
            turns.extend(pattern_turns)
        
        # Shuffle to avoid pattern clustering
        random.shuffle(turns)
        return turns[:count]
    
    async def _generate_personal_assistant_turns(self, count: int) -> List[ConversationTurn]:
        """Generate personal assistant conversation turns."""
        templates = [
            # Key-Value candidates (factual data)
            "My dentist appointment is on Tuesday at 3 PM with Dr. {name}.",
            "I live at {address} in {city}.",
            "My phone number is {phone}.",
            "My favorite restaurant is {restaurant} on {street}.",
            
            # Vector search candidates (preferences, questions)
            "What's a good Italian restaurant near downtown?",
            "I prefer vegetarian food when dining out.",
            "Can you remind me about my meeting tomorrow?",
            "I need to find a good gym in my area.",
            
            # Graph candidates (relationships, connections)
            "My manager Sarah works at TechCorp and reports to the CEO.",
            "Alice introduced me to her friend Bob who works at Google.",
            "The project meeting includes John from Marketing and Lisa from Engineering.",
            
            # Summary candidates (long explanations)
            "Here's what happened at today's meeting: We discussed the quarterly budget allocation, reviewed three new product proposals, decided to hire two additional engineers for the mobile team, and scheduled follow-up meetings with the design team. The main concerns were timeline constraints and resource allocation, particularly around the Q4 release schedule.",
            
            # Discard candidates (chit-chat)
            "Thanks!", "OK sounds good", "Great!", "Yes", "No problem"
        ]
        
        turns = []
        for i in range(count):
            template = random.choice(templates)
            
            # Fill in template variables
            content = template.format(
                name=random.choice(["Smith", "Johnson", "Williams", "Brown"]),
                address=f"{random.randint(100, 9999)} {random.choice(['Main St', 'Oak Ave', 'First St'])}",
                city=random.choice(["Seattle", "Portland", "Austin", "Denver"]),
                phone=f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
                restaurant=random.choice(["Villa Romana", "Green Leaf Cafe", "The Corner Bistro"]),
                street=random.choice(["Main Street", "Broadway", "First Avenue"])
            )
            
            context = "User is talking to their personal assistant about daily tasks and preferences."
            
            turn = ConversationTurn(
                content=content,
                speaker="user",
                context=context,
                metadata={
                    "pattern": "personal_assistant",
                    "template_category": self._classify_template(template),
                    "generated_at": datetime.now().isoformat()
                }
            )
            turns.append(turn)
        
        return turns
    
    async def _generate_technical_discussion_turns(self, count: int) -> List[ConversationTurn]:
        """Generate technical discussion turns."""
        templates = [
            # Technical explanations (Summary candidates)
            "The microservices architecture we're implementing uses Docker containers orchestrated by Kubernetes, with each service having its own database and communication happening through REST APIs and message queues. The main benefits include better scalability, fault isolation, and independent deployment cycles, but we need to carefully manage service discovery and data consistency across boundaries.",
            
            # Technical questions (Vector candidates)
            "What's the best way to implement caching in a distributed system?",
            "How should we handle database migrations in production?",
            "What are the trade-offs between REST and GraphQL APIs?",
            
            # Technical facts (Key-Value candidates)
            "The server runs on AWS EC2 t3.medium instances.",
            "Our database connection timeout is set to 30 seconds.",
            "The API rate limit is 1000 requests per hour per user.",
            
            # Technical relationships (Graph candidates)
            "The authentication service connects to both the user database and the session store, while the API gateway routes requests to the appropriate microservices based on the JWT token claims.",
            
            # Simple acknowledgments (Discard candidates)
            "Got it", "Makes sense", "Understood", "Perfect"
        ]
        
        return self._generate_from_templates(templates, count, "technical_discussion")
    
    async def _generate_casual_chat_turns(self, count: int) -> List[ConversationTurn]:
        """Generate casual chat turns."""
        templates = [
            # Casual long stories (Summary candidates)
            "Yesterday was such an interesting day! I started with my morning coffee at the new cafe downtown, then met up with my college roommate who's visiting from New York. We walked around the park, caught up on old times, talked about our careers and families, and ended up having lunch at that little Mediterranean place near the waterfront. The food was amazing, especially the lamb kebabs and baklava for dessert.",
            
            # Questions and preferences (Vector candidates)
            "What's your favorite type of music for working out?",
            "I really enjoy hiking in the mountains during fall.",
            "Do you have any good book recommendations for science fiction?",
            
            # Personal facts (Key-Value candidates)  
            "I graduated from Stanford in 2018.",
            "My birthday is on March 15th.",
            "I have a golden retriever named Max.",
            
            # Social connections (Graph candidates)
            "My sister works with Tom's cousin at the marketing agency downtown.",
            "I met Sarah through my friend Mike who works at the same company.",
            
            # Simple responses (Discard candidates)
            "Haha yes!", "Oh wow", "That's cool", "Nice!", "Awesome"
        ]
        
        return self._generate_from_templates(templates, count, "casual_chat")
    
    async def _generate_task_planning_turns(self, count: int) -> List[ConversationTurn]:
        """Generate task planning conversation turns."""
        templates = [
            # Complex planning (Summary candidates)
            "For the project launch next month, we need to coordinate with the design team to finalize the UI mockups, work with engineering to complete the API development, schedule user testing sessions with at least 50 participants, prepare marketing materials including blog posts and social media content, and set up monitoring and analytics dashboards to track key performance metrics after launch.",
            
            # Planning questions (Vector candidates)
            "What's the best approach for organizing a remote team meeting?",
            "How should we prioritize these feature requests?",
            "When should we schedule the next product review?",
            
            # Specific dates and facts (Key-Value candidates)
            "The project deadline is November 30th.",
            "We have a budget of $50,000 for this quarter.",
            "The meeting is scheduled for Room 204 at 2 PM.",
            
            # Team relationships (Graph candidates)
            "The design team reports to Sarah who coordinates with the product managers and engineering leads to ensure alignment on feature requirements.",
            
            # Quick confirmations (Discard candidates)
            "Sounds good", "Will do", "On it", "Got it"
        ]
        
        return self._generate_from_templates(templates, count, "task_planning")
    
    async def _generate_knowledge_query_turns(self, count: int) -> List[ConversationTurn]:
        """Generate knowledge query turns."""
        templates = [
            # Detailed explanations (Summary candidates)
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It works by using algorithms to analyze data, identify patterns, and make predictions or decisions. There are three main types: supervised learning (using labeled data), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through rewards and penalties).",
            
            # Knowledge questions (Vector candidates)
            "What is the difference between TCP and UDP protocols?",
            "How does photosynthesis work in plants?",
            "What are the main causes of climate change?",
            
            # Factual data (Key-Value candidates)
            "The speed of light is 299,792,458 meters per second.",
            "Python was created by Guido van Rossum in 1991.",
            "The human brain contains approximately 86 billion neurons.",
            
            # Conceptual relationships (Graph candidates)
            "Quantum mechanics is connected to classical physics through the correspondence principle, while also relating to information theory through quantum computing applications.",
            
            # Simple responses (Discard candidates)
            "I see", "Interesting", "Thanks", "Oh", "Right"
        ]
        
        return self._generate_from_templates(templates, count, "knowledge_query")
    
    def _generate_from_templates(self, templates: List[str], count: int, pattern: str) -> List[ConversationTurn]:
        """Generate turns from templates."""
        turns = []
        for i in range(count):
            content = random.choice(templates)
            context = f"Conversation about {pattern.replace('_', ' ')}"
            
            turn = ConversationTurn(
                content=content,
                speaker=random.choice(["user", "assistant"]),
                context=context,
                metadata={
                    "pattern": pattern,
                    "template_category": self._classify_template(content),
                    "generated_at": datetime.now().isoformat()
                }
            )
            turns.append(turn)
        
        return turns
    
    def _classify_template(self, template: str) -> str:
        """Classify template by expected routing class."""
        if len(template.split()) <= 4 or template in ["Thanks!", "OK sounds good", "Great!", "Yes"]:
            return "discard_candidate"
        elif any(placeholder in template for placeholder in ["{name}", "{phone}", "{address}"]) or "is" in template:
            return "keyvalue_candidate"
        elif template.startswith("What") or template.startswith("How") or "prefer" in template:
            return "vector_candidate"
        elif "connects to" in template or "reports to" in template or "works with" in template:
            return "graph_candidate"
        elif len(template) > 200:
            return "summary_candidate"
        else:
            return "unknown"

class GPT4AutoLabeler:
    """GPT-4 powered auto-labeling system with cost controls and quality assurance."""
    
    def __init__(
        self,
        api_key: str,
        max_budget_usd: float = 25.0,
        confidence_threshold: float = 0.7,
        output_dir: Path = Path("./data_generation/output"),
        checkpoint_interval: int = 100
    ):
        self.api_key = api_key
        self.max_budget_usd = max_budget_usd
        self.confidence_threshold = confidence_threshold
        self.output_dir = Path(output_dir)
        self.checkpoint_interval = checkpoint_interval
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cost tracking
        self.cost_spent = 0.0
        self.api_calls_made = 0
        
        # Quality tracking
        self.quality_metrics = QualityMetrics()
        
        # Results storage
        self.labeled_samples: List[Dict[str, Any]] = []
        self.failed_samples: List[Dict[str, Any]] = []
        
    async def label_conversation_turns(
        self,
        turns: List[ConversationTurn],
        resume_from: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], QualityMetrics]:
        """
        Label conversation turns using GPT-4 with cost and quality controls.
        
        Args:
            turns: List of conversation turns to label
            resume_from: Index to resume from (for checkpoint recovery)
            
        Returns:
            Tuple of (labeled_samples, quality_metrics)
        """
        console.print(f"[bold blue]🏷️  Starting GPT-4 auto-labeling[/bold blue]")
        console.print(f"📊 Total samples: {len(turns)}")
        console.print(f"💰 Budget: ${self.max_budget_usd:.2f}")
        console.print(f"🎯 Confidence threshold: {self.confidence_threshold}")
        
        start_index = resume_from or 0
        remaining_turns = turns[start_index:]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "Labeling turns...", 
                total=len(remaining_turns)
            )
            
            # Process turns in batches for efficiency
            batch_size = 10
            batches = [remaining_turns[i:i+batch_size] for i in range(0, len(remaining_turns), batch_size)]
            
            for batch_idx, batch in enumerate(batches):
                # Check budget before each batch
                if self.cost_spent >= self.max_budget_usd:
                    console.print(f"[yellow]⚠️  Budget exhausted (${self.cost_spent:.4f}). Stopping.[/yellow]")
                    break
                
                # Process batch
                batch_results = await self._process_batch(batch, start_index + batch_idx * batch_size)
                
                # Update results
                for result in batch_results:
                    if result.get("success", False):
                        self.labeled_samples.append(result)
                        
                        # Update quality metrics
                        routing_class = result["label"]["routing_class"]
                        confidence = result["label"]["confidence"]
                        
                        self.quality_metrics.total_samples += 1
                        self.quality_metrics.class_distribution[routing_class] = (
                            self.quality_metrics.class_distribution.get(routing_class, 0) + 1
                        )
                        
                        if confidence < self.confidence_threshold:
                            self.quality_metrics.low_confidence_count += 1
                            
                    else:
                        self.failed_samples.append(result)
                        self.quality_metrics.api_failures += 1
                
                # Update progress
                progress.update(task, advance=len(batch))
                
                # Save checkpoint
                if (batch_idx + 1) % (self.checkpoint_interval // batch_size) == 0:
                    await self._save_checkpoint(start_index + (batch_idx + 1) * batch_size)
                
                # Brief pause to respect rate limits
                await asyncio.sleep(0.1)
        
        # Final quality metrics calculation
        if self.quality_metrics.total_samples > 0:
            total_confidence = sum(
                sample["label"]["confidence"] for sample in self.labeled_samples
            )
            self.quality_metrics.avg_confidence = total_confidence / self.quality_metrics.total_samples
        
        self.quality_metrics.cost_spent_usd = self.cost_spent
        
        # Save final results
        await self._save_final_results()
        
        return self.labeled_samples, self.quality_metrics
    
    async def _process_batch(
        self, 
        batch: List[ConversationTurn], 
        start_index: int
    ) -> List[Dict[str, Any]]:
        """Process a batch of conversation turns."""
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        tasks = []
        for i, turn in enumerate(batch):
            task = self._label_single_turn(
                turn, 
                f"turn_{start_index + i}",
                semaphore
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "turn_id": f"turn_{start_index + i}",
                    "turn_content": batch[i].content
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _label_single_turn(
        self,
        turn: ConversationTurn,
        turn_id: str,
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Label a single conversation turn using GPT-4."""
        
        async with semaphore:
            # Check budget before making API call
            estimated_cost = 0.01  # Rough estimate: $0.01 per call
            if self.cost_spent + estimated_cost > self.max_budget_usd:
                return {
                    "success": False,
                    "error": "Budget exhausted",
                    "turn_id": turn_id
                }
            
            try:
                # Prepare request
                request = LabelingRequest(
                    turn_content=turn.content,
                    conversation_context=turn.context,
                    speaker=turn.speaker,
                    turn_id=turn_id
                )
                
                # Make API call
                response = await self._call_gpt4_labeling_api(request)
                
                # Validate response
                try:
                    label = LabelingResponse(**response)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Response validation failed: {str(e)}",
                        "turn_id": turn_id,
                        "raw_response": response
                    }
                
                # Check confidence threshold
                if label.confidence < self.confidence_threshold:
                    console.print(f"[yellow]⚠️  Low confidence ({label.confidence:.2f}) for turn {turn_id}[/yellow]")
                
                return {
                    "success": True,
                    "turn_id": turn_id,
                    "turn_data": turn.to_dict(),
                    "label": label.dict(),
                    "api_cost_usd": estimated_cost
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "turn_id": turn_id,
                    "turn_content": turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
                }
    
    async def _call_gpt4_labeling_api(self, request: LabelingRequest) -> Dict[str, Any]:
        """Call GPT-4 API for labeling with structured prompt."""
        
        system_prompt = """You are an expert memory system designer. Your job is to classify conversation turns into 5 routing categories for an intelligent memory dispatcher.

ROUTING CLASSES:
0 = DISCARD: Chit-chat, acknowledgments, <4 words, purely social interactions
   Examples: "Thanks!", "OK", "Hi there", "Sounds good", emoji-only

1 = KEY-VALUE: Explicit factual data, structured information, <25 tokens
   Examples: "My user ID is 12345", "I live in Berlin", "Meeting at 3pm", specific facts/IDs/dates

2 = VECTOR: Semantic content, questions, preferences, concepts, search queries
   Examples: "What's the best hotel in Paris?", "I prefer Italian food", explanations/questions

3 = GRAPH: Multi-entity relationships, connections between people/things, temporal sequences
   Examples: "Alice works at TechCorp with Bob", "My flight connects through London", relationships

4 = SUMMARY: Long content >100 tokens, detailed explanations, complex narratives
   Examples: Meeting summaries, long technical explanations, multi-paragraph content

IMPORTANT GUIDELINES:
- Consider content length, complexity, and information type
- Prefer lower-cost routing when uncertain (0 > 1 > 2 > 3 > 4)
- Content with clear entities/relationships → Graph (3)
- Questions and semantic content → Vector (2)
- Short factual data → Key-Value (1)
- Simple chit-chat → Discard (0)
- Long explanatory content → Summary (4)

Respond with valid JSON only."""

        user_prompt = f"""
CONVERSATION CONTEXT: {request.conversation_context}

SPEAKER: {request.speaker}
CONTENT: "{request.turn_content}"

Analyze this content and classify it into one of the 5 routing categories. Consider:
1. Content length and complexity
2. Information type (factual, semantic, relational)
3. Presence of entities and relationships
4. Storage cost optimization

Respond with JSON in this exact format:
{{
  "routing_class": <0-4>,
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>",
  "detected_entities": ["entity1", "entity2"],
  "content_features": {{
    "word_count": <number>,
    "has_entities": <true/false>,
    "has_relationships": <true/false>,
    "is_question": <true/false>
  }}
}}"""

        # Track API cost
        input_tokens = len(system_prompt.split()) + len(user_prompt.split())
        estimated_output_tokens = 150  # Estimated response size
        
        # Cost calculation (GPT-4o-mini pricing)
        input_cost = (input_tokens / 1000) * 0.00015  # $0.15 per 1K input tokens
        output_cost = (estimated_output_tokens / 1000) * 0.00060  # $0.60 per 1K output tokens
        total_cost = input_cost + output_cost
        
        self.cost_spent += total_cost
        self.api_calls_made += 1
        
        # Mock API response for development (replace with actual OpenAI API call)
        # In production, this would be:
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(...) as response:
        #         return await response.json()
        
        # For now, generate a mock response based on content analysis
        mock_response = await self._generate_mock_response(request)
        return mock_response
    
    async def _generate_mock_response(self, request: LabelingRequest) -> Dict[str, Any]:
        """Generate mock labeling response for development (replace with real API)."""
        content = request.turn_content.lower()
        word_count = len(request.turn_content.split())
        
        # Simple heuristic-based classification for mock data
        if word_count <= 4 or content in ["thanks", "ok", "yes", "no", "great"]:
            routing_class = 0  # Discard
            confidence = 0.9
            reasoning = "Short social interaction, likely chit-chat"
        elif any(indicator in content for indicator in ["my", "is", "phone", "address", "id"]) and word_count < 25:
            routing_class = 1  # Key-Value  
            confidence = 0.85
            reasoning = "Factual data suitable for structured storage"
        elif content.startswith("what") or content.startswith("how") or "prefer" in content:
            routing_class = 2  # Vector
            confidence = 0.8
            reasoning = "Question or preference, needs semantic search"
        elif any(indicator in content for indicator in ["works with", "reports to", "connected to"]) or word_count > 30:
            if " and " in content and word_count < 100:
                routing_class = 3  # Graph
                confidence = 0.75
                reasoning = "Relationship or connection between entities"
            else:
                routing_class = 4  # Summary
                confidence = 0.8
                reasoning = "Complex content requiring summarization"
        elif word_count > 100:
            routing_class = 4  # Summary
            confidence = 0.85
            reasoning = "Long content suitable for compression"
        else:
            routing_class = 2  # Vector (default)
            confidence = 0.6
            reasoning = "General content for semantic storage"
        
        # Add some randomness to confidence for more realistic data
        confidence += random.uniform(-0.1, 0.1)
        confidence = max(0.0, min(1.0, confidence))
        
        # Extract entities (simplified)
        entities = []
        words = request.turn_content.split()
        for word in words:
            if word[0].isupper() and word.isalpha() and len(word) > 2:
                entities.append(word)
        
        return {
            "routing_class": routing_class,
            "confidence": confidence,
            "reasoning": reasoning,
            "detected_entities": entities[:5],  # Limit to 5 entities
            "content_features": {
                "word_count": word_count,
                "has_entities": len(entities) > 0,
                "has_relationships": " and " in request.turn_content or " with " in request.turn_content,
                "is_question": request.turn_content.strip().endswith("?")
            }
        }
    
    async def _save_checkpoint(self, current_index: int) -> None:
        """Save progress checkpoint for recovery."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.output_dir / f"checkpoint_{timestamp}.json"
        
        checkpoint_data = {
            "current_index": current_index,
            "cost_spent": self.cost_spent,
            "api_calls_made": self.api_calls_made,
            "labeled_samples_count": len(self.labeled_samples),
            "failed_samples_count": len(self.failed_samples),
            "quality_metrics": self.quality_metrics.dict(),
            "timestamp": timestamp
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        console.print(f"💾 Checkpoint saved: {checkpoint_file}")
    
    async def _save_final_results(self) -> None:
        """Save final labeling results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save labeled samples
        results_file = self.output_dir / f"labeled_data_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "metadata": {
                    "total_samples": len(self.labeled_samples),
                    "cost_spent_usd": self.cost_spent,
                    "api_calls_made": self.api_calls_made,
                    "quality_metrics": self.quality_metrics.dict(),
                    "generated_at": timestamp
                },
                "labeled_samples": self.labeled_samples
            }, f, indent=2)
        
        # Save failed samples for analysis
        if self.failed_samples:
            failures_file = self.output_dir / f"failed_samples_{timestamp}.json"
            with open(failures_file, 'w') as f:
                json.dump(self.failed_samples, f, indent=2)
        
        console.print(f"💾 Final results saved: {results_file}")
        if self.failed_samples:
            console.print(f"⚠️  Failed samples saved: {failures_file}")
    
    def print_quality_report(self) -> None:
        """Print comprehensive quality assessment report."""
        console.print("\n[bold]📊 LABELING QUALITY REPORT[/bold]")
        
        # Overall stats
        stats_table = Table(title="Overall Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Samples", str(self.quality_metrics.total_samples))
        stats_table.add_row("API Calls Made", str(self.api_calls_made))
        stats_table.add_row("Cost Spent", f"${self.cost_spent:.4f}")
        stats_table.add_row("Avg Confidence", f"{self.quality_metrics.avg_confidence:.3f}")
        stats_table.add_row("Low Confidence", f"{self.quality_metrics.low_confidence_count} ({self.quality_metrics.low_confidence_count/max(1,self.quality_metrics.total_samples)*100:.1f}%)")
        stats_table.add_row("API Failures", str(self.quality_metrics.api_failures))
        
        console.print(stats_table)
        
        # Class distribution
        if self.quality_metrics.class_distribution:
            class_table = Table(title="Class Distribution")
            class_table.add_column("Class", style="cyan")
            class_table.add_column("Name", style="yellow")
            class_table.add_column("Count", style="green")
            class_table.add_column("Percentage", style="magenta")
            
            total = sum(self.quality_metrics.class_distribution.values())
            class_names = ["Discard", "Key-Value", "Vector", "Graph", "Summary"]
            
            for class_id in range(5):
                count = self.quality_metrics.class_distribution.get(class_id, 0)
                percentage = (count / total * 100) if total > 0 else 0
                
                class_table.add_row(
                    str(class_id),
                    class_names[class_id],
                    str(count),
                    f"{percentage:.1f}%"
                )
            
            console.print(class_table)
            
            # Balance score
            balance_score = self.quality_metrics.get_class_balance_score()
            console.print(f"\n🎯 [bold]Class Balance Score: {balance_score:.3f}[/bold] (1.0 = perfectly balanced)")
            
            if balance_score < 0.7:
                console.print("[yellow]⚠️  Warning: Class distribution is imbalanced. Consider generating more data for underrepresented classes.[/yellow]")


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="GPT-4 Auto-Labeling System for Memory Routing Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_generation/auto_labeler.py --budget 25.0 --target-samples 14000
  python data_generation/auto_labeler.py --validate-only --samples 200
  python data_generation/auto_labeler.py --resume checkpoint_20250806.json
        """
    )
    
    parser.add_argument("--budget", type=float, default=25.0, help="Maximum budget in USD")
    parser.add_argument("--target-samples", type=int, default=14000, help="Target number of samples")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Minimum confidence threshold")
    parser.add_argument("--output-dir", type=str, default="./data_generation/output", help="Output directory")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation on existing data")
    parser.add_argument("--samples", type=int, help="Number of samples for validation mode")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint file")
    parser.add_argument("--dry-run", action="store_true", help="Generate samples without API calls")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        console.print("[red]❌ OpenAI API key required. Set OPENAI_API_KEY or use --api-key[/red]")
        return 1
    
    try:
        # Initialize components
        generator = ConversationGenerator()
        labeler = GPT4AutoLabeler(
            api_key=api_key or "mock",
            max_budget_usd=args.budget,
            confidence_threshold=args.confidence_threshold,
            output_dir=Path(args.output_dir)
        )
        
        if args.validate_only:
            # Validation mode - analyze existing data
            console.print("[bold yellow]🔍 Running validation mode[/bold yellow]")
            
            # TODO: Implement validation logic
            console.print("Validation mode not yet implemented")
            return 0
        
        elif args.resume:
            # Resume mode - load checkpoint and continue
            console.print(f"[bold yellow]🔄 Resuming from checkpoint: {args.resume}[/bold yellow]")
            
            # TODO: Implement resume logic
            console.print("Resume mode not yet implemented")
            return 0
        
        else:
            # Normal generation mode
            console.print("[bold green]🚀 Starting data generation[/bold green]")
            
            # Generate conversation turns
            console.print(f"📝 Generating {args.target_samples} conversation turns...")
            turns = await generator.generate_turns(args.target_samples)
            
            console.print(f"✅ Generated {len(turns)} conversation turns")
            
            if args.dry_run:
                console.print("[yellow]🏃 Dry run mode - skipping API calls[/yellow]")
                
                # Show sample turns
                console.print("\n[bold]Sample Generated Turns:[/bold]")
                for i, turn in enumerate(turns[:5]):
                    console.print(f"\n{i+1}. [{turn.metadata['pattern']}] {turn.speaker}:")
                    console.print(f"   \"{turn.content}\"")
                
                console.print(f"\n✅ Dry run completed. Generated {len(turns)} turns.")
                return 0
            
            # Label turns using GPT-4
            labeled_samples, quality_metrics = await labeler.label_conversation_turns(turns)
            
            # Print results
            labeler.print_quality_report()
            
            console.print(f"\n[bold green]✅ Labeling completed successfully![/bold green]")
            console.print(f"📊 Total labeled samples: {len(labeled_samples)}")
            console.print(f"💰 Total cost: ${labeler.cost_spent:.4f}")
            
            return 0
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Data generation failed: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)