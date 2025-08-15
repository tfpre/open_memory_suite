#!/usr/bin/env python3
"""
Production Training Data Generator

Generates comprehensive, high-quality training data for ML-enhanced memory routing.
Creates 1000+ labeled examples across diverse conversation scenarios and routing decisions.

Features:
- Diverse conversation types (technical, personal, casual, business)
- Balanced action distribution across all routing decisions
- Realistic context features based on conversation analysis
- Quality validation and filtering
- Export to multiple formats (JSON, CSV, HuggingFace datasets)

Usage:
    python generate_production_training_data.py --output-dir ./production_data
    python generate_production_training_data.py --num-examples 2000 --validate
"""

import asyncio
import json
import csv
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import argparse
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

from open_memory_suite.adapters import MemoryItem, InMemoryAdapter, FileStoreAdapter
from open_memory_suite.benchmark import CostModel, BenchmarkHarness
from open_memory_suite.benchmark.cost_model import BudgetType
from open_memory_suite.dispatcher import (
    HeuristicPolicy,
    ConversationContext,
    MemoryAction,
    DataCollector,
)

console = Console()

@dataclass
class ConversationTemplate:
    """Template for generating diverse conversations."""
    name: str
    description: str
    budget_type: BudgetType
    conversation_type: str
    user_profile: Dict[str, Any]
    turns_range: Tuple[int, int]
    content_patterns: List[str]
    expected_actions: Dict[MemoryAction, float]  # Expected action distribution

@dataclass
class TrainingExample:
    """A single training example with rich metadata."""
    text: str
    label: int
    action: str
    context_features: List[float]
    session_id: str
    turn_number: int
    speaker: str
    content: str
    conversation_type: str
    budget_type: str
    user_profile: Dict[str, Any]
    expected_action: str
    confidence: float
    metadata: Dict[str, Any]

class ProductionDataGenerator:
    """Generate high-quality production training data."""
    
    def __init__(self, output_dir: Path):
        """Initialize the data generator."""
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.heuristic_policy = HeuristicPolicy()
        self.cost_model = CostModel()
        self.adapters = []
        
        # Templates for diverse conversations
        self.conversation_templates = self._create_conversation_templates()
        
        # Content generation patterns
        self.content_generators = self._create_content_generators()
        
    async def initialize(self):
        """Initialize components."""
        self.adapters = [
            InMemoryAdapter("memory_store"),
            FileStoreAdapter("file_store", self.output_dir / "temp_storage"),
        ]
        
        for adapter in self.adapters:
            await adapter.initialize()
        
        # We'll generate data directly without using DataCollector
        # since we want more control over the generation process
    
    def _create_conversation_templates(self) -> List[ConversationTemplate]:
        """Create diverse conversation templates."""
        return [
            ConversationTemplate(
                name="technical_deep_dive",
                description="In-depth technical discussions with detailed explanations",
                budget_type=BudgetType.PREMIUM,
                conversation_type="technical",
                user_profile={"expertise": "high", "domain": "engineering"},
                turns_range=(8, 15),
                content_patterns=[
                    "Can you explain how {technology} works in {context}?",
                    "What are the trade-offs between {approach1} and {approach2}?",
                    "I'm implementing {system} and facing {challenge}",
                    "The documentation says {quote} but I'm confused about {aspect}",
                ],
                expected_actions={
                    MemoryAction.STORE: 0.7,
                    MemoryAction.SUMMARIZE: 0.2,
                    MemoryAction.DROP: 0.05,
                    MemoryAction.DEFER: 0.05,
                }
            ),
            ConversationTemplate(
                name="personal_information_sharing",
                description="Users sharing personal details and experiences",
                budget_type=BudgetType.STANDARD,
                conversation_type="personal",
                user_profile={"expertise": "medium", "domain": "general"},
                turns_range=(6, 12),
                content_patterns=[
                    "Hi, my name is {name} and I work as a {profession}",
                    "I live in {location} and I'm interested in {interests}",
                    "My background is in {field} with {years} years of experience",
                    "I'm currently working on {project} at {company}",
                ],
                expected_actions={
                    MemoryAction.STORE: 0.8,
                    MemoryAction.SUMMARIZE: 0.1,
                    MemoryAction.DROP: 0.05,
                    MemoryAction.DEFER: 0.05,
                }
            ),
            ConversationTemplate(
                name="casual_chat",
                description="Light conversation with minimal information value",
                budget_type=BudgetType.MINIMAL,
                conversation_type="casual",
                user_profile={"expertise": "low", "domain": "general"},
                turns_range=(4, 8),
                content_patterns=[
                    "Hello there!",
                    "How are you doing today?",
                    "Thanks for your help",
                    "That's interesting",
                    "I see",
                    "Got it, thanks",
                    "Alright",
                    "Perfect",
                ],
                expected_actions={
                    MemoryAction.STORE: 0.1,
                    MemoryAction.SUMMARIZE: 0.1,
                    MemoryAction.DROP: 0.7,
                    MemoryAction.DEFER: 0.1,
                }
            ),
            ConversationTemplate(
                name="business_consultation",
                description="Professional consultation with actionable advice",
                budget_type=BudgetType.PREMIUM,
                conversation_type="business",
                user_profile={"expertise": "high", "domain": "business"},
                turns_range=(10, 18),
                content_patterns=[
                    "I'm the CEO of {company} and we're facing {challenge}",
                    "Our quarterly metrics show {data} and we need to {action}",
                    "The market analysis indicates {trend} in {sector}",
                    "We're considering {strategy} to achieve {goal}",
                ],
                expected_actions={
                    MemoryAction.STORE: 0.6,
                    MemoryAction.SUMMARIZE: 0.3,
                    MemoryAction.DROP: 0.05,
                    MemoryAction.DEFER: 0.05,
                }
            ),
            ConversationTemplate(
                name="educational_qa",
                description="Q&A sessions with learning objectives",
                budget_type=BudgetType.STANDARD,
                conversation_type="educational",
                user_profile={"expertise": "low", "domain": "learning"},
                turns_range=(5, 10),
                content_patterns=[
                    "I'm learning about {subject} and need help with {concept}",
                    "Can you explain {topic} in simple terms?",
                    "What's the difference between {term1} and {term2}?",
                    "I read that {fact} but I don't understand {aspect}",
                ],
                expected_actions={
                    MemoryAction.STORE: 0.6,
                    MemoryAction.SUMMARIZE: 0.3,
                    MemoryAction.DROP: 0.05,
                    MemoryAction.DEFER: 0.05,
                }
            ),
            ConversationTemplate(
                name="troubleshooting_session",
                description="Problem-solving with step-by-step guidance",
                budget_type=BudgetType.STANDARD,
                conversation_type="troubleshooting",
                user_profile={"expertise": "medium", "domain": "technical"},
                turns_range=(6, 14),
                content_patterns=[
                    "I'm having an issue with {system} where {problem}",
                    "When I try to {action}, I get {error}",
                    "I followed the steps but {issue} still occurs",
                    "The logs show {log_message} and I'm not sure what it means",
                ],
                expected_actions={
                    MemoryAction.STORE: 0.7,
                    MemoryAction.SUMMARIZE: 0.2,
                    MemoryAction.DROP: 0.05,
                    MemoryAction.DEFER: 0.05,
                }
            ),
        ]
    
    def _create_content_generators(self) -> Dict[str, Any]:
        """Create content generation patterns and data."""
        return {
            "names": ["Alex Chen", "Maria Rodriguez", "James Wilson", "Priya Patel", "David Kim", "Sarah Johnson"],
            "professions": ["software engineer", "data scientist", "product manager", "designer", "researcher", "consultant"],
            "companies": ["TechCorp", "StartupCo", "BigTech Inc", "InnovateLab", "DataSolutions", "CloudSystems"],
            "locations": ["San Francisco", "New York", "London", "Berlin", "Tokyo", "Toronto"],
            "technologies": ["machine learning", "cloud computing", "blockchain", "AI", "microservices", "data engineering"],
            "challenges": ["scalability issues", "performance problems", "integration complexity", "data quality", "security concerns"],
            "interests": ["technology", "machine learning", "startups", "photography", "travel", "reading"],
            "domains": ["healthcare", "finance", "education", "retail", "manufacturing", "entertainment"],
        }
    
    async def generate_comprehensive_dataset(
        self,
        num_examples: int = 1500,
        validate_quality: bool = True,
        export_formats: List[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive production training dataset."""
        
        if export_formats is None:
            export_formats = ["json", "csv"]
        
        rprint(f"🚀 [bold blue]Generating {num_examples} production training examples...[/bold blue]")
        
        all_examples = []
        examples_per_template = num_examples // len(self.conversation_templates)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            
            for template in self.conversation_templates:
                task = progress.add_task(
                    f"Generating {template.name} conversations...",
                    total=examples_per_template
                )
                
                template_examples = await self._generate_template_examples(
                    template, examples_per_template, progress, task
                )
                all_examples.extend(template_examples)
        
        # Quality validation
        if validate_quality:
            all_examples = await self._validate_and_filter_examples(all_examples)
        
        # Calculate statistics
        stats = self._calculate_dataset_statistics(all_examples)
        
        # Export in multiple formats
        export_paths = {}
        for format_type in export_formats:
            path = await self._export_dataset(all_examples, format_type)
            export_paths[format_type] = path
        
        # Generate metadata
        metadata = {
            "generation_time": datetime.now().isoformat(),
            "total_examples": len(all_examples),
            "templates_used": [t.name for t in self.conversation_templates],
            "statistics": stats,
            "export_paths": {k: str(v) for k, v in export_paths.items()},  # Convert Path to str
            "quality_validated": validate_quality,
        }
        
        # Save metadata
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        rprint(f"✅ [bold green]Generated {len(all_examples)} high-quality examples![/bold green]")
        return metadata
    
    async def _generate_template_examples(
        self,
        template: ConversationTemplate,
        num_examples: int,
        progress: Progress,
        task_id: Any,
    ) -> List[TrainingExample]:
        """Generate examples for a specific conversation template."""
        
        examples = []
        conversations_needed = num_examples // 8  # ~8 examples per conversation
        
        for conv_idx in range(conversations_needed):
            # Generate conversation
            conversation = await self._generate_conversation_from_template(template, conv_idx)
            
            # Convert to training examples
            conv_examples = await self._conversation_to_training_examples(
                conversation, template
            )
            examples.extend(conv_examples)
            
            progress.update(task_id, advance=len(conv_examples))
            
            # Stop if we have enough examples
            if len(examples) >= num_examples:
                break
        
        return examples[:num_examples]
    
    async def _generate_conversation_from_template(
        self,
        template: ConversationTemplate,
        conv_idx: int,
    ) -> List[Tuple[str, str]]:  # List of (speaker, content) tuples
        """Generate a single conversation from a template."""
        
        num_turns = random.randint(*template.turns_range)
        conversation = []
        
        # Generate conversation context
        context_vars = self._generate_context_variables(template)
        
        for turn_idx in range(num_turns):
            speaker = "user" if turn_idx % 2 == 0 else "assistant"
            
            if speaker == "user":
                content = self._generate_user_message(template, context_vars, turn_idx)
            else:
                content = self._generate_assistant_response(template, context_vars, turn_idx)
            
            conversation.append((speaker, content))
        
        return conversation
    
    def _generate_context_variables(self, template: ConversationTemplate) -> Dict[str, str]:
        """Generate context variables for content generation."""
        generators = self.content_generators
        
        return {
            "name": random.choice(generators["names"]),
            "profession": random.choice(generators["professions"]),
            "company": random.choice(generators["companies"]),
            "location": random.choice(generators["locations"]),
            "technology": random.choice(generators["technologies"]),
            "challenge": random.choice(generators["challenges"]),
            "interests": ", ".join(random.sample(generators["interests"], 2)),
            "domain": random.choice(generators["domains"]),
        }
    
    def _generate_user_message(
        self,
        template: ConversationTemplate,
        context_vars: Dict[str, str],
        turn_idx: int,
    ) -> str:
        """Generate a user message based on template and context."""
        
        # Create content that will trigger specific actions
        if turn_idx == 0:
            # Opening message - usually gets stored
            pattern = random.choice(template.content_patterns)
        elif turn_idx < 3:
            # Early conversation - mix of store and drop
            if template.conversation_type == "casual":
                # More likely to be dropped
                patterns = [
                    "Hi there!",
                    "Hello",
                    "How are you?",
                    "Good morning",
                    "Hey!",
                ]
            else:
                # More likely to be stored (questions, factual)
                patterns = [
                    "Can you help me understand {technology}?",
                    "I'm particularly interested in {domain}",
                    "What would you recommend for {challenge}?",
                    "My name is {name} and I work at {company}",
                ]
            pattern = random.choice(patterns)
        elif turn_idx >= len(template.content_patterns) - 2:
            # Closing messages - should be dropped
            patterns = [
                "Thanks!",
                "Got it",
                "Perfect",
                "Alright",
                "OK",
                "I see",
                "Thanks for the explanation!",
                "That's very helpful",
                "I appreciate your help"
            ]
            pattern = random.choice(patterns)
        else:
            # Middle conversation - use template patterns
            pattern = random.choice(template.content_patterns)
        
        # Fill in template variables
        try:
            content = pattern.format(**context_vars)
        except KeyError:
            # Fallback if template variables don't match
            content = pattern.replace("{", "").replace("}", "")
        
        return content
    
    def _generate_assistant_response(
        self,
        template: ConversationTemplate,
        context_vars: Dict[str, str],
        turn_idx: int,
    ) -> str:
        """Generate an assistant response based on template and context."""
        
        if template.conversation_type == "technical":
            responses = [
                f"That's a great question about {context_vars['technology']}. Let me explain...",
                f"In the context of {context_vars['domain']}, the key considerations are...",
                f"Based on your {context_vars['profession']} background, I'd suggest...",
                f"The challenge you're facing with {context_vars['challenge']} is common...",
            ]
        elif template.conversation_type == "personal":
            responses = [
                f"Nice to meet you! {context_vars['location']} is a great place for {context_vars['profession']}s.",
                f"Your experience in {context_vars['domain']} sounds fascinating.",
                f"Working at {context_vars['company']} must be exciting.",
                f"Your interests in {context_vars['interests']} align well with your profession.",
            ]
        elif template.conversation_type == "casual":
            responses = [
                "I'm doing well, thank you for asking!",
                "You're very welcome!",
                "I'm glad I could help.",
                "Feel free to ask if you have more questions.",
            ]
        else:
            responses = [
                "I understand your situation.",
                "Let me provide some guidance on that.",
                "That's an important consideration.",
                "I can help you with that.",
            ]
        
        return random.choice(responses)
    
    async def _conversation_to_training_examples(
        self,
        conversation: List[Tuple[str, str]],
        template: ConversationTemplate,
    ) -> List[TrainingExample]:
        """Convert a conversation to training examples."""
        
        examples = []
        session_id = f"{template.name}_{random.randint(1000, 9999)}"
        
        for turn_idx, (speaker, content) in enumerate(conversation):
            # Create memory item
            item = MemoryItem(
                content=content,
                speaker=speaker,
                session_id=session_id,
                metadata={"turn": turn_idx},
            )
            
            # Create conversation context
            # Build recent turns from conversation history
            recent_turns = []
            for hist_idx, (hist_speaker, hist_content) in enumerate(conversation[:turn_idx]):
                recent_item = MemoryItem(
                    content=hist_content,
                    speaker=hist_speaker,
                    session_id=session_id,
                    metadata={"turn": hist_idx},
                )
                recent_turns.append(recent_item)
            
            context = ConversationContext(
                session_id=session_id,
                budget_type=template.budget_type,
                turn_count=turn_idx,
                recent_turns=recent_turns[-3:],  # Keep last 3 turns for context
            )
            
            # Get heuristic decision for ground truth
            action = await self.heuristic_policy.decide_action(item, context)
            
            # Extract context features
            context_features = self._extract_context_features(item, context)
            
            # Create formatted text input
            text_input = self._format_text_input(item, context)
            
            # Create training example
            example = TrainingExample(
                text=text_input,
                label=self._action_to_label(action),
                action=action.value,
                context_features=context_features,
                session_id=session_id,
                turn_number=turn_idx,
                speaker=speaker,
                content=content,
                conversation_type=template.conversation_type,
                budget_type=template.budget_type.value,
                user_profile=template.user_profile,
                expected_action=action.value,
                confidence=self._calculate_confidence(action, template),
                metadata={
                    "template": template.name,
                    "turn_ratio": turn_idx / len(conversation),
                    "content_length": len(content),
                },
            )
            
            examples.append(example)
        
        return examples
    
    def _extract_context_features(self, item: MemoryItem, context: ConversationContext) -> List[float]:
        """Extract numerical context features for ML training."""
        
        # Basic conversation metrics
        turn_count = float(context.turn_count)
        content_length = float(len(item.content))
        word_count = float(len(item.content.split()))
        
        # Speaker context (0.0 for user, 1.0 for assistant)
        speaker_is_assistant = 1.0 if item.speaker == "assistant" else 0.0
        
        # Turn ratio (position in conversation)
        recent_turns_count = len(context.recent_turns)
        turn_ratio = turn_count / max(1, turn_count + 1)
        
        # Budget type encoding
        budget_encoding = {
            BudgetType.MINIMAL: 0.0,
            BudgetType.STANDARD: 0.5,
            BudgetType.PREMIUM: 1.0,
        }.get(context.budget_type, 0.5)
        
        # Budget status
        budget_exhausted = 1.0 if context.budget_exhausted else 0.0
        
        # Content type heuristics
        has_question = 1.0 if "?" in item.content else 0.0
        has_personal_info = 1.0 if any(word in item.content.lower() for word in ["my", "i'm", "i am", "name is"]) else 0.0
        is_short_response = 1.0 if len(item.content.split()) < 5 else 0.0
        has_technical_terms = 1.0 if any(word in item.content.lower() for word in ["algorithm", "system", "model", "data", "code"]) else 0.0
        
        return [
            turn_count,
            content_length,
            word_count,
            speaker_is_assistant,
            turn_ratio,
            budget_encoding,
            budget_exhausted,
            has_question,
            has_personal_info,
            is_short_response,
        ]
    
    def _format_text_input(self, item: MemoryItem, context: ConversationContext) -> str:
        """Format text input for ML model training."""
        
        # Build context string from recent turns
        context_parts = []
        
        # Add recent conversation history (last 2-3 turns)
        if context.recent_turns:
            for recent_item in context.recent_turns[-2:]:  # Last 2 turns
                content_preview = recent_item.content[:100]  # Truncate long messages
                context_parts.append(f"[{recent_item.speaker}]: {content_preview}")
        
        context_str = " | ".join(context_parts) if context_parts else "Start of conversation"
        
        # Format the main input
        formatted_input = (
            f"Context: {context_str} | "
            f"Turn {context.turn_count}, Budget: {context.budget_type.value} | "
            f"Current: [{item.speaker}]: {item.content}"
        )
        
        return formatted_input
    
    def _action_to_label(self, action: MemoryAction) -> int:
        """Convert action to numeric label."""
        action_mapping = {
            MemoryAction.STORE: 0,
            MemoryAction.SUMMARIZE: 1,
            MemoryAction.DROP: 2,
            MemoryAction.DEFER: 3,
        }
        return action_mapping[action]
    
    def _calculate_confidence(self, action: MemoryAction, template: ConversationTemplate) -> float:
        """Calculate confidence score based on expected action distribution."""
        expected_prob = template.expected_actions.get(action, 0.1)
        # Higher expected probability = higher confidence
        return min(0.95, max(0.05, expected_prob + random.uniform(-0.1, 0.1)))
    
    async def _validate_and_filter_examples(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Validate and filter examples for quality."""
        rprint("🔍 [bold yellow]Validating example quality...[/bold yellow]")
        
        valid_examples = []
        
        for example in examples:
            if self._is_valid_example(example):
                valid_examples.append(example)
        
        # Ensure balanced distribution
        valid_examples = self._balance_action_distribution(valid_examples)
        
        rprint(f"✅ Kept {len(valid_examples)}/{len(examples)} examples after quality validation")
        return valid_examples
    
    def _is_valid_example(self, example: TrainingExample) -> bool:
        """Check if an example meets quality criteria."""
        
        # Content length checks - be more lenient
        if len(example.content) < 2 or len(example.content) > 2000:
            return False
        
        # Context features validity
        if len(example.context_features) != 10:
            return False
        
        # No extreme outliers in features - be more lenient
        if any(abs(f) > 10000 for f in example.context_features):
            return False
            
        # Check for NaN values
        if any(f != f for f in example.context_features):  # NaN check
            return False
        
        # Valid label range
        if example.label < 0 or example.label > 3:
            return False
        
        # Check required fields are not empty
        if not example.text or not example.content:
            return False
        
        return True
    
    def _balance_action_distribution(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Ensure reasonable distribution across action classes."""
        
        # Group by action
        action_groups = {}
        for example in examples:
            action = example.action
            if action not in action_groups:
                action_groups[action] = []
            action_groups[action].append(example)
        
        # Don't over-balance - just ensure we have reasonable representation
        # Find target size based on available data
        total_examples = len(examples)
        num_actions = len(action_groups)
        
        if num_actions < 2:
            return examples  # Can't balance with only one action type
        
        # Aim for at least 50 examples per action, but be flexible
        min_per_action = max(20, total_examples // (num_actions * 3))  # More lenient
        
        balanced_examples = []
        for action, group in action_groups.items():
            # Take up to min_per_action examples, but don't throw away too much data
            target_size = min(len(group), min_per_action * 2)  # Allow 2x flexibility
            if len(group) > target_size:
                sampled = random.sample(group, target_size)
            else:
                sampled = group
            balanced_examples.extend(sampled)
        
        random.shuffle(balanced_examples)
        return balanced_examples
    
    def _calculate_dataset_statistics(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics."""
        
        # Action distribution
        action_counts = {}
        for example in examples:
            action_counts[example.action] = action_counts.get(example.action, 0) + 1
        
        # Conversation type distribution
        type_counts = {}
        for example in examples:
            conv_type = example.conversation_type
            type_counts[conv_type] = type_counts.get(conv_type, 0) + 1
        
        # Budget distribution
        budget_counts = {}
        for example in examples:
            budget = example.budget_type
            budget_counts[budget] = budget_counts.get(budget, 0) + 1
        
        # Content statistics
        content_lengths = [len(example.content) for example in examples]
        
        return {
            "total_examples": len(examples),
            "action_distribution": action_counts,
            "conversation_type_distribution": type_counts,
            "budget_type_distribution": budget_counts,
            "content_length_stats": {
                "min": min(content_lengths),
                "max": max(content_lengths),
                "mean": sum(content_lengths) / len(content_lengths),
            },
            "unique_sessions": len(set(ex.session_id for ex in examples)),
        }
    
    async def _export_dataset(self, examples: List[TrainingExample], format_type: str) -> Path:
        """Export dataset in specified format."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "json":
            output_path = self.output_dir / f"production_training_data_{timestamp}.json"
            
            # Convert to serializable format
            data = [asdict(example) for example in examples]
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format_type == "csv":
            output_path = self.output_dir / f"production_training_data_{timestamp}.csv"
            
            with open(output_path, 'w', newline='') as f:
                if examples:
                    fieldnames = list(asdict(examples[0]).keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for example in examples:
                        row = asdict(example)
                        # Convert complex fields to JSON strings
                        row['context_features'] = json.dumps(row['context_features'])
                        row['user_profile'] = json.dumps(row['user_profile'])
                        row['metadata'] = json.dumps(row['metadata'])
                        writer.writerow(row)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        return output_path
    
    def display_statistics(self, metadata: Dict[str, Any]) -> None:
        """Display dataset statistics in a nice table."""
        
        stats = metadata["statistics"]
        
        # Overview table
        overview_table = Table(title="Production Training Dataset Overview")
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", style="magenta")
        
        overview_table.add_row("Total Examples", str(stats["total_examples"]))
        overview_table.add_row("Unique Sessions", str(stats["unique_sessions"]))
        overview_table.add_row("Generation Time", metadata["generation_time"])
        overview_table.add_row("Quality Validated", "✅" if metadata["quality_validated"] else "❌")
        
        console.print(overview_table)
        
        # Action distribution table
        action_table = Table(title="Action Distribution")
        action_table.add_column("Action", style="cyan")
        action_table.add_column("Count", style="magenta")
        action_table.add_column("Percentage", style="yellow")
        
        total = stats["total_examples"]
        for action, count in stats["action_distribution"].items():
            percentage = (count / total) * 100
            action_table.add_row(action, str(count), f"{percentage:.1f}%")
        
        console.print(action_table)
        
        # Conversation type distribution
        type_table = Table(title="Conversation Type Distribution")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="magenta")
        type_table.add_column("Percentage", style="yellow")
        
        for conv_type, count in stats["conversation_type_distribution"].items():
            percentage = (count / total) * 100
            type_table.add_row(conv_type, str(count), f"{percentage:.1f}%")
        
        console.print(type_table)
    
    async def cleanup(self):
        """Clean up resources."""
        for adapter in self.adapters:
            await adapter.cleanup()

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate production training data")
    parser.add_argument("--output-dir", type=str, default="./production_data",
                       help="Output directory for generated data")
    parser.add_argument("--num-examples", type=int, default=1500,
                       help="Number of training examples to generate")
    parser.add_argument("--validate", action="store_true",
                       help="Enable quality validation and filtering")
    parser.add_argument("--formats", nargs="+", default=["json", "csv"],
                       help="Export formats (json, csv)")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ProductionDataGenerator(Path(args.output_dir))
    
    try:
        await generator.initialize()
        
        # Generate dataset
        metadata = await generator.generate_comprehensive_dataset(
            num_examples=args.num_examples,
            validate_quality=args.validate,
            export_formats=args.formats,
        )
        
        # Display results
        generator.display_statistics(metadata)
        
        rprint(f"\n🎉 [bold green]Production training data generated successfully![/bold green]")
        rprint(f"📁 Output directory: {args.output_dir}")
        
        for format_type, path in metadata["export_paths"].items():
            rprint(f"📄 {format_type.upper()}: {path}")
        
    finally:
        await generator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())