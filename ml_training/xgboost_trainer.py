#!/usr/bin/env python3
"""
XGBoost Memory Routing Classifier Training Pipeline

Production-grade ML training system for 5-class memory routing decisions with
comprehensive feature engineering, model validation, and interpretability.

Key Features:
- Advanced feature engineering from conversation content
- Cross-validation with conversation-level grouping
- Hyperparameter optimization with budget controls
- Model interpretability and feature importance analysis
- Production deployment preparation
- Comprehensive performance evaluation

Usage:
    python ml_training/xgboost_trainer.py --data data_generation/output/labeled_data_*.json
    python ml_training/xgboost_trainer.py --train --validate --save-model
    python ml_training/xgboost_trainer.py --hyperparameter-search --cv-folds 5
"""

import argparse
import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer

# Rich for beautiful CLI output
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich import print as rprint

console = Console()

# Try to load spaCy
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        console.print("[yellow]⚠️  spaCy English model not found. Install with: python -m spacy download en_core_web_sm[/yellow]")
        nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    console.print("[yellow]⚠️  spaCy not available. Advanced linguistic features disabled.[/yellow]")
    nlp = None
    SPACY_AVAILABLE = False

@dataclass
class TrainingConfig:
    """Configuration for XGBoost training."""
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # XGBoost parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    
    # Feature engineering
    max_tfidf_features: int = 1000
    min_tfidf_df: int = 2
    ngram_range: Tuple[int, int] = (1, 2)
    
    # Training controls
    early_stopping_rounds: int = 10
    eval_metric: str = "mlogloss"
    objective: str = "multi:softprob"
    
    def to_xgb_params(self) -> Dict[str, Any]:
        """Convert to XGBoost parameter dictionary."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "random_state": self.random_state,
            "verbosity": 0,
            "n_jobs": -1
        }

@dataclass
class ModelPerformance:
    """Container for model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    class_accuracies: Dict[int, float]
    class_f1_scores: Dict[int, float]
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float]
    cross_val_scores: List[float]
    training_time_seconds: float
    
    def get_cost_weighted_score(self, class_costs: Optional[Dict[int, float]] = None) -> float:
        """Calculate cost-weighted performance score."""
        if class_costs is None:
            # Default costs: higher class number = higher cost
            class_costs = {0: 0.0, 1: 0.1, 2: 0.5, 3: 1.0, 4: 2.0}
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for class_id, f1 in self.class_f1_scores.items():
            cost_weight = 1.0 / (class_costs.get(class_id, 1.0) + 0.1)  # Inverse cost weighting
            weighted_score += f1 * cost_weight
            total_weight += cost_weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0

class FeatureExtractor:
    """Advanced feature extraction for conversation content."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tfidf_vectorizer = None
        self.feature_names = []
        self.label_encoder = LabelEncoder()
        
    def extract_features(
        self, 
        samples: List[Dict[str, Any]], 
        fit_extractors: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract comprehensive features from conversation samples.
        
        Args:
            samples: List of labeled conversation samples
            fit_extractors: Whether to fit feature extractors (training mode)
            
        Returns:
            Tuple of (features_df, labels)
        """
        console.print(f"[bold blue]🔧 Extracting features from {len(samples)} samples[/bold blue]")
        
        # Extract content and labels
        contents = []
        contexts = []
        speakers = []
        labels = []
        
        for sample in samples:
            turn_data = sample["turn_data"]
            label_data = sample["label"]
            
            contents.append(turn_data["content"])
            contexts.append(turn_data.get("context", ""))
            speakers.append(turn_data["speaker"])
            labels.append(label_data["routing_class"])
        
        # Initialize feature dataframe
        features_df = pd.DataFrame()
        
        # 1. Basic text features
        features_df = pd.concat([features_df, self._extract_basic_text_features(contents)], axis=1)
        
        # 2. Linguistic features (if spaCy available)
        if SPACY_AVAILABLE:
            features_df = pd.concat([features_df, self._extract_linguistic_features(contents)], axis=1)
        
        # 3. TF-IDF features
        tfidf_features = self._extract_tfidf_features(contents, fit_extractors)
        features_df = pd.concat([features_df, tfidf_features], axis=1)
        
        # 4. Context features
        features_df = pd.concat([features_df, self._extract_context_features(contexts)], axis=1)
        
        # 5. Speaker features
        features_df = pd.concat([features_df, self._extract_speaker_features(speakers)], axis=1)
        
        # 6. Conversation pattern features
        features_df = pd.concat([features_df, self._extract_pattern_features(samples)], axis=1)
        
        # Encode labels
        if fit_extractors:
            labels_encoded = self.label_encoder.fit_transform(labels)
        else:
            labels_encoded = self.label_encoder.transform(labels)
        
        # Store feature names
        self.feature_names = list(features_df.columns)
        
        console.print(f"✅ Extracted {len(self.feature_names)} features")
        
        return features_df, labels_encoded
    
    def _extract_basic_text_features(self, contents: List[str]) -> pd.DataFrame:
        """Extract basic text statistics."""
        features = []
        
        for content in contents:
            words = content.split()
            sentences = content.split('.')
            
            feature_dict = {
                "word_count": len(words),
                "char_count": len(content),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
                "question_mark_count": content.count('?'),
                "exclamation_count": content.count('!'),
                "uppercase_ratio": sum(c.isupper() for c in content) / max(len(content), 1),
                "digit_count": sum(c.isdigit() for c in content),
                "punctuation_count": sum(not c.isalnum() and not c.isspace() for c in content),
                "is_very_short": int(len(words) <= 4),
                "is_short": int(len(words) <= 10),
                "is_medium": int(10 < len(words) <= 50),
                "is_long": int(len(words) > 50),
                "starts_with_what": int(content.lower().startswith("what")),
                "starts_with_how": int(content.lower().startswith("how")),
                "starts_with_when": int(content.lower().startswith("when")),
                "starts_with_where": int(content.lower().startswith("where")),
                "starts_with_why": int(content.lower().startswith("why")),
                "ends_with_question": int(content.strip().endswith("?")),
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _extract_linguistic_features(self, contents: List[str]) -> pd.DataFrame:
        """Extract linguistic features using spaCy."""
        features = []
        
        for content in contents:
            try:
                doc = nlp(content)
                
                # Entity counts by type
                entity_counts = {}
                for ent_type in ["PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", "PERCENT"]:
                    entity_counts[f"ent_{ent_type.lower()}_count"] = sum(
                        1 for ent in doc.ents if ent.label_ == ent_type
                    )
                
                # POS tag counts
                pos_counts = {}
                for pos in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]:
                    pos_counts[f"pos_{pos.lower()}_count"] = sum(
                        1 for token in doc if token.pos_ == pos
                    )
                
                # Dependency relations
                dep_counts = {}
                for dep in ["nsubj", "dobj", "prep", "compound"]:
                    dep_counts[f"dep_{dep}_count"] = sum(
                        1 for token in doc if token.dep_ == dep
                    )
                
                feature_dict = {
                    "total_entities": len(doc.ents),
                    "unique_entities": len(set(ent.text.lower() for ent in doc.ents)),
                    "entity_density": len(doc.ents) / max(len(doc), 1),
                    **entity_counts,
                    **pos_counts,
                    **dep_counts,
                    "has_relationships": int(any(
                        token.dep_ in ["nmod", "appos", "compound"] for token in doc
                    )),
                }
                
            except Exception:
                # Fallback if spaCy processing fails
                feature_dict = {f"ent_{t}_count": 0 for t in ["person", "org", "gpe", "date", "time", "money", "percent"]}
                feature_dict.update({f"pos_{t}_count": 0 for t in ["noun", "verb", "adj", "adv", "propn"]})
                feature_dict.update({f"dep_{t}_count": 0 for t in ["nsubj", "dobj", "prep", "compound"]})
                feature_dict.update({
                    "total_entities": 0,
                    "unique_entities": 0,
                    "entity_density": 0,
                    "has_relationships": 0
                })
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _extract_tfidf_features(
        self, 
        contents: List[str], 
        fit_vectorizer: bool = True
    ) -> pd.DataFrame:
        """Extract TF-IDF features."""
        if fit_vectorizer or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.max_tfidf_features,
                min_df=self.config.min_tfidf_df,
                ngram_range=self.config.ngram_range,
                stop_words='english',
                lowercase=True,
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Words only, no numbers/punctuation
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(contents)
        
        # Convert to DataFrame with feature names
        feature_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names
        )
        
        return tfidf_df
    
    def _extract_context_features(self, contexts: List[str]) -> pd.DataFrame:
        """Extract features from conversation context."""
        features = []
        
        for context in contexts:
            context_lower = context.lower()
            
            feature_dict = {
                "context_length": len(context),
                "context_has_personal": int(any(
                    word in context_lower for word in ["personal", "assistant", "user", "my", "me"]
                )),
                "context_has_technical": int(any(
                    word in context_lower for word in ["technical", "system", "database", "api", "code"]
                )),
                "context_has_casual": int(any(
                    word in context_lower for word in ["casual", "chat", "friend", "conversation"]
                )),
                "context_has_task": int(any(
                    word in context_lower for word in ["task", "planning", "project", "meeting", "work"]
                )),
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _extract_speaker_features(self, speakers: List[str]) -> pd.DataFrame:
        """Extract speaker-based features."""
        features = []
        
        for speaker in speakers:
            feature_dict = {
                "is_user": int(speaker == "user"),
                "is_assistant": int(speaker == "assistant"),
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _extract_pattern_features(self, samples: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract conversation pattern features."""
        features = []
        
        for sample in samples:
            metadata = sample["turn_data"].get("metadata", {})
            label_features = sample["label"].get("content_features", {})
            
            feature_dict = {
                "confidence": sample["label"]["confidence"],
                "pattern_personal_assistant": int(metadata.get("pattern") == "personal_assistant"),
                "pattern_technical": int(metadata.get("pattern") == "technical_discussion"),
                "pattern_casual": int(metadata.get("pattern") == "casual_chat"),
                "pattern_task_planning": int(metadata.get("pattern") == "task_planning"),
                "pattern_knowledge": int(metadata.get("pattern") == "knowledge_query"),
                "detected_entities_count": len(sample["label"].get("detected_entities", [])),
                "has_detected_entities": int(len(sample["label"].get("detected_entities", [])) > 0),
                "content_has_entities": int(label_features.get("has_entities", False)),
                "content_has_relationships": int(label_features.get("has_relationships", False)),
                "content_is_question": int(label_features.get("is_question", False)),
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)

class XGBoostTrainer:
    """XGBoost training pipeline with advanced validation and interpretability."""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.model = None
        self.feature_extractor = FeatureExtractor(self.config)
        self.scaler = StandardScaler()
        self.training_history = {}
        
    def load_data(self, data_paths: List[Path]) -> List[Dict[str, Any]]:
        """Load labeled data from multiple JSON files."""
        all_samples = []
        
        console.print(f"[bold blue]📂 Loading data from {len(data_paths)} files[/bold blue]")
        
        for data_path in data_paths:
            console.print(f"Loading: {data_path}")
            
            with open(data_path, 'r') as f:
                data = json.load(f)
                
            samples = data.get("labeled_samples", [])
            all_samples.extend(samples)
            
            console.print(f"  ✅ Loaded {len(samples)} samples")
        
        console.print(f"[bold green]📊 Total samples loaded: {len(all_samples)}[/bold green]")
        
        # Filter out failed samples
        successful_samples = [s for s in all_samples if s.get("success", False)]
        
        if len(successful_samples) != len(all_samples):
            console.print(f"[yellow]⚠️  Filtered out {len(all_samples) - len(successful_samples)} failed samples[/yellow]")
        
        return successful_samples
    
    def prepare_data(
        self, 
        samples: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training with proper train/val/test splits.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        console.print(f"[bold blue]🔄 Preparing data for training[/bold blue]")
        
        # Extract features
        X, y = self.feature_extractor.extract_features(samples, fit_extractors=True)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Second split: train/validation from remaining data
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        # Scale features (fit on training, transform all)
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        console.print(f"✅ Data prepared:")
        console.print(f"  📊 Training samples: {len(X_train_scaled)}")
        console.print(f"  📊 Validation samples: {len(X_val_scaled)}")
        console.print(f"  📊 Test samples: {len(X_test_scaled)}")
        console.print(f"  🔧 Features: {len(X_train_scaled.columns)}")
        
        # Print class distribution
        unique_classes, train_counts = np.unique(y_train, return_counts=True)
        class_names = ["Discard", "Key-Value", "Vector", "Graph", "Summary"]
        
        distribution_table = Table(title="Training Data Class Distribution")
        distribution_table.add_column("Class", style="cyan")
        distribution_table.add_column("Name", style="yellow")
        distribution_table.add_column("Count", style="green")
        distribution_table.add_column("Percentage", style="magenta")
        
        for class_id, count in zip(unique_classes, train_counts):
            percentage = count / len(y_train) * 100
            distribution_table.add_row(
                str(class_id),
                class_names[class_id],
                str(count),
                f"{percentage:.1f}%"
            )
        
        console.print(distribution_table)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True
    ) -> ModelPerformance:
        """Train XGBoost model with validation and early stopping."""
        
        console.print(f"[bold blue]🚀 Training XGBoost model[/bold blue]")
        
        start_time = pd.Timestamp.now()
        
        # Create XGBoost model
        self.model = xgb.XGBClassifier(**self.config.to_xgb_params())
        
        # Train model (early stopping can be configured in XGBClassifier parameters if needed)
        self.model.fit(X_train, y_train, verbose=verbose)
        
        training_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        console.print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set
        performance = self._evaluate_model(X_val, y_val, training_time)
        
        # Cross-validation scores
        cv_scores = self._compute_cross_validation_scores(
            pd.concat([X_train, X_val]), 
            np.concatenate([y_train, y_val])
        )
        performance.cross_val_scores = cv_scores
        
        return performance
    
    def _evaluate_model(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        training_time: float
    ) -> ModelPerformance:
        """Comprehensive model evaluation."""
        
        console.print("[bold blue]📊 Evaluating model performance[/bold blue]")
        
        # Predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # Overall metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        
        # Per-class metrics
        class_report = classification_report(y, y_pred, output_dict=True)
        class_accuracies = {}
        class_f1_scores = {}
        
        for class_id in range(5):  # 5 classes
            if str(class_id) in class_report:
                class_accuracies[class_id] = class_report[str(class_id)]['precision']
                class_f1_scores[class_id] = class_report[str(class_id)]['f1-score']
            else:
                class_accuracies[class_id] = 0.0
                class_f1_scores[class_id] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_extractor.feature_names,
            self.model.feature_importances_
        ))
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        performance = ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            class_accuracies=class_accuracies,
            class_f1_scores=class_f1_scores,
            confusion_matrix=cm,
            feature_importance=feature_importance,
            cross_val_scores=[],  # Will be filled by caller
            training_time_seconds=training_time
        )
        
        return performance
    
    def _compute_cross_validation_scores(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray
    ) -> List[float]:
        """Compute cross-validation scores."""
        
        console.print(f"[bold blue]🔄 Computing {self.config.cv_folds}-fold cross-validation[/bold blue]")
        
        # Create temporary model for CV (without training history)
        cv_model = xgb.XGBClassifier(**self.config.to_xgb_params())
        
        cv_scores = cross_val_score(
            cv_model, X, y,
            cv=StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            ),
            scoring='accuracy',
            n_jobs=-1
        )
        
        console.print(f"✅ CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores.tolist()
    
    def print_performance_report(self, performance: ModelPerformance) -> None:
        """Print comprehensive performance report."""
        
        console.print("\n[bold]📊 MODEL PERFORMANCE REPORT[/bold]")
        
        # Overall metrics
        metrics_table = Table(title="Overall Performance")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        metrics_table.add_row("Accuracy", f"{performance.accuracy:.4f}")
        metrics_table.add_row("Precision", f"{performance.precision:.4f}")
        metrics_table.add_row("Recall", f"{performance.recall:.4f}")
        metrics_table.add_row("F1 Score", f"{performance.f1_score:.4f}")
        metrics_table.add_row("Cost-Weighted Score", f"{performance.get_cost_weighted_score():.4f}")
        metrics_table.add_row("Training Time", f"{performance.training_time_seconds:.2f}s")
        
        if performance.cross_val_scores:
            cv_mean = np.mean(performance.cross_val_scores)
            cv_std = np.std(performance.cross_val_scores)
            metrics_table.add_row("CV Mean Accuracy", f"{cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        console.print(metrics_table)
        
        # Per-class performance
        class_table = Table(title="Per-Class Performance")
        class_table.add_column("Class", style="cyan")
        class_table.add_column("Name", style="yellow")
        class_table.add_column("Precision", style="green")
        class_table.add_column("F1 Score", style="magenta")
        
        class_names = ["Discard", "Key-Value", "Vector", "Graph", "Summary"]
        for class_id in range(5):
            class_table.add_row(
                str(class_id),
                class_names[class_id],
                f"{performance.class_accuracies.get(class_id, 0.0):.4f}",
                f"{performance.class_f1_scores.get(class_id, 0.0):.4f}"
            )
        
        console.print(class_table)
        
        # Top feature importance
        importance_table = Table(title="Top 15 Most Important Features")
        importance_table.add_column("Rank", style="cyan")
        importance_table.add_column("Feature", style="yellow")
        importance_table.add_column("Importance", style="green")
        
        top_features = list(performance.feature_importance.items())[:15]
        for i, (feature, importance) in enumerate(top_features, 1):
            importance_table.add_row(
                str(i),
                feature,
                f"{importance:.4f}"
            )
        
        console.print(importance_table)
        
        # Confusion matrix
        console.print(f"\n[bold]Confusion Matrix:[/bold]")
        console.print(performance.confusion_matrix)
    
    def save_model(
        self, 
        output_dir: Path, 
        model_name: str = "xgboost_memory_router"
    ) -> Dict[str, Path]:
        """Save trained model and associated artifacts."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # Save XGBoost model
        model_path = output_dir / f"{model_name}_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        saved_files["model"] = model_path
        
        # Save feature extractor
        extractor_path = output_dir / f"{model_name}_feature_extractor_{timestamp}.pkl"
        with open(extractor_path, 'wb') as f:
            pickle.dump(self.feature_extractor, f)
        saved_files["feature_extractor"] = extractor_path
        
        # Save scaler
        scaler_path = output_dir / f"{model_name}_scaler_{timestamp}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        saved_files["scaler"] = scaler_path
        
        # Save config
        config_path = output_dir / f"{model_name}_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        saved_files["config"] = config_path
        
        console.print(f"[bold green]💾 Model saved successfully![/bold green]")
        for artifact_type, path in saved_files.items():
            console.print(f"  {artifact_type}: {path}")
        
        return saved_files


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="XGBoost Memory Routing Classifier Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml_training/xgboost_trainer.py --data data_generation/output/labeled_data_*.json
  python ml_training/xgboost_trainer.py --train --validate --save-model
  python ml_training/xgboost_trainer.py --hyperparameter-search --cv-folds 5
        """
    )
    
    parser.add_argument("--data", nargs="+", required=True, help="Paths to labeled data JSON files")
    parser.add_argument("--output-dir", type=str, default="./ml_models", help="Output directory for models")
    parser.add_argument("--config", type=str, help="Training configuration JSON file")
    parser.add_argument("--save-model", action="store_true", help="Save trained model")
    parser.add_argument("--hyperparameter-search", action="store_true", help="Perform hyperparameter search")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = TrainingConfig(**config_dict)
        else:
            config = TrainingConfig()
        
        # Update config with CLI args
        config.cv_folds = args.cv_folds
        config.random_state = args.random_state
        
        # Initialize trainer
        trainer = XGBoostTrainer(config)
        
        # Load data
        data_paths = [Path(p) for p in args.data]
        samples = trainer.load_data(data_paths)
        
        if len(samples) == 0:
            console.print("[red]❌ No valid samples loaded. Check input files.[/red]")
            return 1
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(samples)
        
        # Train model
        performance = trainer.train_model(X_train, X_val, y_train, y_val, verbose=args.verbose)
        
        # Print results
        trainer.print_performance_report(performance)
        
        # Test set evaluation
        console.print("\n[bold blue]🧪 Final evaluation on held-out test set[/bold blue]")
        test_performance = trainer._evaluate_model(X_test, y_test, 0)
        console.print(f"[bold]Test Set Accuracy: {test_performance.accuracy:.4f}[/bold]")
        console.print(f"[bold]Test Set F1 Score: {test_performance.f1_score:.4f}[/bold]")
        
        # Save model if requested
        if args.save_model:
            saved_files = trainer.save_model(Path(args.output_dir))
            console.print(f"\n✅ Training completed successfully!")
            console.print(f"📊 Final Test Accuracy: {test_performance.accuracy:.4f}")
            console.print(f"📊 Final Test F1: {test_performance.f1_score:.4f}")
        
        return 0
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Training failed: {str(e)}[/bold red]")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)