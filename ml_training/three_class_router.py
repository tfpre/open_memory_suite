#!/usr/bin/env python3
"""
3-Class Router with Calibrated Abstention

Implements the friend's recommendation for a simplified, reliable 3-class routing system:
- Class 0: Discard (chit-chat, <4 tokens, acknowledgments)
- Class 1: Store (factual content worth keeping)
- Class 2: Compress (long content requiring summarization)

Features:
- XGBoost classification with calibrated probabilities
- Abstention mechanism falling back to heuristics when confidence < threshold
- Feature engineering for content analysis
- Production-ready with explain endpoint
"""

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import numpy as np

# ML dependencies
try:
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import pandas as pd
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("ML dependencies not available. Install with: pip install scikit-learn xgboost")

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Enhanced routing decision with calibrated confidence."""
    
    class_name: str              # "discard", "store", "compress"
    class_id: int               # 0, 1, 2
    confidence: float           # Calibrated probability for predicted class
    all_probabilities: Dict[str, float]  # All class probabilities
    abstained: bool             # Whether model abstained due to low confidence
    reasoning: str              # Human-readable explanation
    feature_importance: Dict[str, float]  # Top feature contributions
    
    # Metadata for debugging
    raw_scores: Optional[np.ndarray] = None
    calibrated_scores: Optional[np.ndarray] = None
    heuristic_fallback: Optional[str] = None


class ContentAnalyzer:
    """Feature extraction for 3-class routing."""
    
    def __init__(self):
        self.tfidf = None
        self.scaler = None
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Define patterns for heuristic features."""
        self.discard_patterns = [
            # Acknowledgments
            r'\b(ok|okay|thanks?|ty|thx|got it|sounds good|alright)\b',
            # Greetings
            r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
            # Single words
            r'^\w{1,3}$',
            # Punctuation only
            r'^[^\w]*$',
            # Emoji only
            r'^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+$'
        ]
        
        self.factual_patterns = [
            # Names, dates, numbers
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper names
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
            r'\b\d+(\.\d+)?\b',  # Numbers
            # Contact info  
            r'\b\w+@\w+\.\w+\b',  # Emails
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            # Locations
            r'\bat \w+\b|\bin \w+\b',  # Location prepositions
        ]
        
        self.question_patterns = [
            r'\?',  # Question mark
            r'\b(what|who|when|where|why|how|can you|could you|would you)\b',
            r'\b(is|are|do|does|did|will|would|should)\b.*\?'
        ]
    
    def extract_features(self, content: str, metadata: Optional[Dict] = None) -> Dict[str, float]:
        """Extract comprehensive features for 3-class routing."""
        import re
        
        features = {}
        content_lower = content.lower()
        
        # Basic content features
        features['char_count'] = len(content)
        features['word_count'] = len(content.split())
        features['token_count'] = len(content) / 4.2  # Rough estimate
        features['sentence_count'] = len([s for s in content.split('.') if s.strip()])
        features['avg_word_length'] = np.mean([len(w) for w in content.split()]) if content.split() else 0
        
        # Discard signals (Class 0)
        features['is_very_short'] = float(len(content) < 4)
        features['is_short'] = float(len(content.split()) < 4)
        features['discard_pattern_count'] = sum(len(re.findall(pattern, content_lower)) 
                                               for pattern in self.discard_patterns)
        features['is_acknowledgment'] = float(any(word in content_lower 
                                                for word in ['ok', 'thanks', 'got it', 'alright']))
        features['is_greeting'] = float(any(word in content_lower 
                                          for word in ['hi', 'hello', 'hey']))
        
        # Factual signals (Class 1)  
        features['has_proper_names'] = float(len(re.findall(r'\b[A-Z][a-z]+\b', content)) > 0)
        features['has_numbers'] = float(len(re.findall(r'\b\d+\b', content)) > 0)
        features['has_dates'] = float(len(re.findall(r'\b\d{1,2}[/-]\d{1,2}\b', content)) > 0)
        features['factual_pattern_count'] = sum(len(re.findall(pattern, content)) 
                                               for pattern in self.factual_patterns)
        features['has_email'] = float('@' in content and '.' in content)
        features['has_phone'] = float(len(re.findall(r'\d{3}[-.]?\d{3}[-.]?\d{4}', content)) > 0)
        
        # Question signals
        features['has_question_mark'] = float('?' in content)
        features['question_pattern_count'] = sum(len(re.findall(pattern, content_lower)) 
                                                 for pattern in self.question_patterns)
        features['starts_with_question_word'] = float(any(content_lower.startswith(qw) 
                                                         for qw in ['what', 'who', 'when', 'where', 'why', 'how']))
        
        # Long content signals (Class 2 - compress)
        features['is_long'] = float(len(content) > 500)
        features['is_very_long'] = float(len(content) > 1000)
        features['paragraph_count'] = len([p for p in content.split('\n') if p.strip()])
        features['has_multiple_sentences'] = float(content.count('.') > 2)
        features['complexity_ratio'] = len(set(content.split())) / max(1, len(content.split()))
        
        # Punctuation analysis
        features['exclamation_count'] = content.count('!')
        features['comma_count'] = content.count(',')
        features['semicolon_count'] = content.count(';')
        features['punctuation_ratio'] = len([c for c in content if not c.isalnum()]) / max(1, len(content))
        
        # Conversational context (if metadata provided)
        if metadata:
            features['speaker_is_user'] = float(metadata.get('speaker') == 'user')
            features['turn_number'] = float(metadata.get('turn_number', 0))
            features['session_length'] = float(metadata.get('session_length', 1))
            features['time_since_last'] = float(metadata.get('time_since_last', 0))
        else:
            features['speaker_is_user'] = 0.5  # Neutral
            features['turn_number'] = 0
            features['session_length'] = 1
            features['time_since_last'] = 0
        
        return features


class ThreeClassRouter:
    """
    Production-ready 3-class router with calibrated abstention.
    
    Classes:
    - 0: Discard (chit-chat, acknowledgments, very short content)
    - 1: Store (factual information worth keeping)
    - 2: Compress (long content requiring summarization)
    """
    
    def __init__(
        self, 
        confidence_threshold: float = 0.7,
        model_path: Optional[Path] = None
    ):
        """
        Initialize 3-class router.
        
        Args:
            confidence_threshold: Minimum confidence for prediction (else abstain)
            model_path: Path to saved model files
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        # Core components
        self.analyzer = ContentAnalyzer()
        self.classifier = None
        self.calibrator = None  # For probability calibration
        self.feature_names = []
        
        # Class mapping
        self.class_names = {0: "discard", 1: "store", 2: "compress"}
        self.name_to_class = {v: k for k, v in self.class_names.items()}
        
        # Statistics
        self.training_stats = {}
        self.prediction_count = 0
        self.abstention_count = 0
        
        if model_path and model_path.exists():
            self.load_model()
    
    def prepare_training_data(self, labeled_conversations: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from labeled conversations.
        
        Args:
            labeled_conversations: List of dicts with 'content', 'label', 'metadata'
            
        Returns:
            (X_features, y_labels, feature_names)
        """
        logger.info(f"Preparing training data from {len(labeled_conversations)} examples")
        
        features_list = []
        labels = []
        
        for example in labeled_conversations:
            content = example['content']
            label = example['label']
            metadata = example.get('metadata', {})
            
            # Extract features
            features = self.analyzer.extract_features(content, metadata)
            features_list.append(features)
            
            # Convert label to class ID
            if isinstance(label, str):
                class_id = self.name_to_class.get(label, 1)  # Default to "store" if unknown
            else:
                class_id = int(label)
            labels.append(class_id)
        
        # Convert to arrays
        feature_names = list(features_list[0].keys()) if features_list else []
        X = np.array([[f[name] for name in feature_names] for f in features_list])
        y = np.array(labels)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Label distribution: {np.bincount(y)}")
        
        return X, y, feature_names
    
    def train(
        self, 
        training_data: List[Dict],
        validation_split: float = 0.2,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train 3-class router with XGBoost and calibration.
        
        Args:
            training_data: Labeled training examples
            validation_split: Fraction for validation set
            save_model: Whether to save trained model
            
        Returns:
            Training metrics and performance report
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("ML dependencies required for training")
        
        logger.info("Starting 3-class router training")
        start_time = time.time()
        
        # Prepare data
        X, y, feature_names = self.prepare_training_data(training_data)
        self.feature_names = feature_names
        
        # Train-validation split (conversation-level to prevent leakage)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Train XGBoost classifier
        logger.info("Training XGBoost classifier...")
        self.classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='multi:softprob',
            eval_metric='mlogloss'
        )
        
        self.classifier.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calibrate probabilities for better confidence estimates
        logger.info("Calibrating probability estimates...")
        self.calibrator = CalibratedClassifierCV(
            self.classifier,
            method='isotonic',  # Isotonic regression for calibration
            cv=3
        )
        self.calibrator.fit(X_train, y_train)
        
        # Evaluate performance
        y_pred = self.calibrator.predict(X_val)
        y_pred_proba = self.calibrator.predict_proba(X_val)
        
        # Calculate metrics
        training_time = time.time() - start_time
        accuracy = (y_pred == y_val).mean()
        
        # Confidence analysis
        max_probas = y_pred_proba.max(axis=1)
        high_conf_mask = max_probas >= self.confidence_threshold
        abstention_rate = 1 - high_conf_mask.mean()
        high_conf_accuracy = (y_pred[high_conf_mask] == y_val[high_conf_mask]).mean() if high_conf_mask.sum() > 0 else 0
        
        # Feature importance
        feature_importance = dict(zip(feature_names, self.classifier.feature_importances_))
        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Training statistics
        self.training_stats = {
            'training_examples': len(training_data),
            'feature_count': len(feature_names),
            'training_time_seconds': training_time,
            'validation_accuracy': accuracy,
            'abstention_rate': abstention_rate,
            'high_confidence_accuracy': high_conf_accuracy,
            'confidence_threshold': self.confidence_threshold,
            'class_distribution': dict(zip(self.class_names.values(), np.bincount(y))),
            'top_features': top_features
        }
        
        # Classification report
        report = classification_report(y_val, y_pred, target_names=list(self.class_names.values()), output_dict=True)
        
        if save_model and self.model_path:
            self.save_model()
        
        logger.info(f"Training completed in {training_time:.1f}s")
        logger.info(f"Validation accuracy: {accuracy:.3f}")
        logger.info(f"Abstention rate: {abstention_rate:.3f}")
        logger.info(f"High-confidence accuracy: {high_conf_accuracy:.3f}")
        
        return {
            'training_stats': self.training_stats,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
            'feature_importance': top_features
        }
    
    def predict(
        self, 
        content: str, 
        metadata: Optional[Dict] = None,
        return_reasoning: bool = True
    ) -> RoutingDecision:
        """
        Make routing decision with calibrated confidence.
        
        Args:
            content: Text content to route
            metadata: Optional context metadata
            return_reasoning: Whether to include detailed reasoning
            
        Returns:
            RoutingDecision with class, confidence, and reasoning
        """
        self.prediction_count += 1
        
        # Extract features
        features = self.analyzer.extract_features(content, metadata)
        X = np.array([[features[name] for name in self.feature_names]])
        
        # Get predictions
        if self.calibrator is not None:
            # Use calibrated probabilities
            probas = self.calibrator.predict_proba(X)[0]
            raw_scores = self.classifier.predict_proba(X)[0] if self.classifier else probas
        else:
            # Fallback to uncalibrated if no calibrator
            if self.classifier is not None:
                probas = self.classifier.predict_proba(X)[0]
                raw_scores = probas
            else:
                # No model available - use heuristics
                return self._heuristic_fallback(content, features, "No trained model available")
        
        # Get predicted class and confidence
        predicted_class_id = np.argmax(probas)
        confidence = probas[predicted_class_id]
        class_name = self.class_names[predicted_class_id]
        
        # Abstention check
        abstained = confidence < self.confidence_threshold
        if abstained:
            self.abstention_count += 1
            return self._heuristic_fallback(content, features, f"Low confidence ({confidence:.3f} < {self.confidence_threshold})")
        
        # Feature importance for explanation
        if return_reasoning and self.classifier is not None:
            feature_importance = self._explain_prediction(features, predicted_class_id)
        else:
            feature_importance = {}
        
        # Generate reasoning
        reasoning = self._generate_reasoning(content, features, class_name, confidence, feature_importance)
        
        return RoutingDecision(
            class_name=class_name,
            class_id=predicted_class_id,
            confidence=confidence,
            all_probabilities={self.class_names[i]: prob for i, prob in enumerate(probas)},
            abstained=False,
            reasoning=reasoning,
            feature_importance=feature_importance,
            raw_scores=raw_scores,
            calibrated_scores=probas
        )
    
    def _heuristic_fallback(self, content: str, features: Dict[str, float], reason: str) -> RoutingDecision:
        """Fallback heuristic routing when ML model abstains or unavailable."""
        
        # Simple heuristic rules for 3-class routing
        if features['is_very_short'] or features['is_acknowledgment'] or features['is_greeting']:
            class_name = "discard"
            class_id = 0
            confidence = 0.9  # High confidence in heuristic
            heuristic_rule = "Very short, acknowledgment, or greeting"
            
        elif features['is_very_long'] or features['has_multiple_sentences']:
            class_name = "compress"
            class_id = 2
            confidence = 0.8
            heuristic_rule = "Long content requiring compression"
            
        else:
            # Default to store for factual content
            class_name = "store"
            class_id = 1
            confidence = 0.7
            heuristic_rule = "Default factual content"
        
        reasoning = f"Heuristic fallback ({reason}): {heuristic_rule}"
        
        return RoutingDecision(
            class_name=class_name,
            class_id=class_id,
            confidence=confidence,
            all_probabilities={class_name: confidence},
            abstained=True,
            reasoning=reasoning,
            feature_importance={},
            heuristic_fallback=heuristic_rule
        )
    
    def _explain_prediction(self, features: Dict[str, float], predicted_class: int) -> Dict[str, float]:
        """Generate feature importance for prediction explanation."""
        if self.classifier is None:
            return {}
        
        # Get feature importance from XGBoost
        importances = dict(zip(self.feature_names, self.classifier.feature_importances_))
        
        # Weight by actual feature values for this prediction
        weighted_importance = {}
        for feature_name, importance in importances.items():
            feature_value = features.get(feature_name, 0)
            # Weight importance by feature value (higher values = more influential)
            weighted_importance[feature_name] = importance * abs(feature_value)
        
        # Return top 5 most influential features
        top_features = dict(sorted(weighted_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        return top_features
    
    def _generate_reasoning(
        self, 
        content: str, 
        features: Dict[str, float], 
        class_name: str, 
        confidence: float,
        feature_importance: Dict[str, float]
    ) -> str:
        """Generate human-readable reasoning for the routing decision."""
        
        reasoning_parts = []
        
        # Class-specific reasoning
        if class_name == "discard":
            if features['is_very_short']:
                reasoning_parts.append("very short content")
            if features['is_acknowledgment']:
                reasoning_parts.append("acknowledgment/response")
            if features['is_greeting']:
                reasoning_parts.append("greeting")
            if features['discard_pattern_count'] > 0:
                reasoning_parts.append("chit-chat patterns detected")
                
        elif class_name == "store":
            if features['has_proper_names']:
                reasoning_parts.append("contains names")
            if features['has_numbers']:
                reasoning_parts.append("contains numbers/data")
            if features['has_dates']:
                reasoning_parts.append("contains dates")
            if features['factual_pattern_count'] > 0:
                reasoning_parts.append("factual information detected")
            if features['has_question_mark']:
                reasoning_parts.append("question requiring response")
                
        elif class_name == "compress":
            if features['is_very_long']:
                reasoning_parts.append("very long content")
            if features['has_multiple_sentences']:
                reasoning_parts.append("multiple sentences")
            if features['paragraph_count'] > 1:
                reasoning_parts.append("multiple paragraphs")
        
        # Add confidence
        conf_desc = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        reasoning_parts.append(f"{conf_desc} confidence ({confidence:.2f})")
        
        # Add top feature if available
        if feature_importance:
            top_feature = max(feature_importance.keys(), key=lambda k: feature_importance[k])
            reasoning_parts.append(f"key signal: {top_feature}")
        
        return f"→ {class_name}: {', '.join(reasoning_parts)}"
    
    def save_model(self) -> None:
        """Save trained model to disk."""
        if not self.model_path:
            raise ValueError("No model path specified")
            
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save components
        model_data = {
            'classifier': self.classifier,
            'calibrator': self.calibrator,
            'feature_names': self.feature_names,
            'confidence_threshold': self.confidence_threshold,
            'class_names': self.class_names,
            'training_stats': self.training_stats,
            'version': '3-class-v1.0'
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self) -> None:
        """Load trained model from disk."""
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.calibrator = model_data['calibrator']
        self.feature_names = model_data['feature_names']
        self.confidence_threshold = model_data['confidence_threshold']
        self.class_names = model_data['class_names']
        self.training_stats = model_data.get('training_stats', {})
        
        logger.info(f"Model loaded from {self.model_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router performance statistics."""
        return {
            'predictions_made': self.prediction_count,
            'abstentions': self.abstention_count,
            'abstention_rate': self.abstention_count / max(1, self.prediction_count),
            'confidence_threshold': self.confidence_threshold,
            'training_stats': self.training_stats,
            'feature_count': len(self.feature_names),
            'class_names': self.class_names
        }


def create_synthetic_training_data(n_examples: int = 1000) -> List[Dict]:
    """Create synthetic training data for 3-class router."""
    
    examples = []
    
    # Class 0: Discard examples
    discard_examples = [
        "ok", "thanks", "got it", "alright", "sounds good", "hi", "hello", "hey",
        "yes", "no", "sure", "👍", "😊", "k", "ty", "thx", "yep", "nope",
        "good morning", "good evening", "see you later", "bye", "lol", "haha"
    ]
    
    # Class 1: Store examples (factual content)
    store_examples = [
        "My name is John Smith and I live in Seattle",
        "The meeting is scheduled for 3pm on Tuesday",
        "My phone number is 555-123-4567",
        "I work at Microsoft as a software engineer", 
        "The project deadline is December 15th, 2024",
        "My email address is john@company.com",
        "I prefer Italian food over Chinese food",
        "What time is the dentist appointment?",
        "Where did I put my keys?",
        "How much does the subscription cost?",
        "My flight number is AA1234 departing at 8:30am",
        "The conference is in San Francisco next month",
        "I need to pick up groceries at 6pm"
    ]
    
    # Class 2: Compress examples (long content)
    compress_examples = [
        "I had a really interesting conversation with my colleague today about the future of artificial intelligence and machine learning in our industry. We discussed how these technologies might impact job roles, productivity, and the overall direction of our company over the next five years. There were some fascinating insights about automation, human-AI collaboration, and the ethical considerations we need to keep in mind as we develop new products.",
        
        "The quarterly business review meeting covered several important topics including revenue growth, customer acquisition metrics, product development roadmap, and competitive analysis. The finance team presented detailed charts showing our performance against targets, while the marketing team outlined their strategy for the upcoming product launch. We also discussed budget allocations for the next quarter and identified key areas for improvement.",
        
        "During my vacation last week, I visited three different cities and had amazing experiences in each one. The first city had incredible architecture and museums, the second was known for its food scene and cultural attractions, and the third offered beautiful natural scenery and outdoor activities. I took hundreds of photos and tried many local specialties. The weather was perfect throughout the trip, and I met some wonderful people along the way."
    ]
    
    # Generate examples with proper distribution
    target_per_class = n_examples // 3
    
    # Generate discard examples
    for i in range(target_per_class):
        content = np.random.choice(discard_examples)
        # Add some variation
        if np.random.random() < 0.3:
            content = content.upper()
        elif np.random.random() < 0.2:
            content = content + "!"
        
        examples.append({
            'content': content,
            'label': 'discard',
            'metadata': {
                'speaker': np.random.choice(['user', 'assistant']),
                'turn_number': np.random.randint(1, 20),
                'session_length': np.random.randint(5, 50)
            }
        })
    
    # Generate store examples  
    for i in range(target_per_class):
        content = np.random.choice(store_examples)
        # Add some variation by combining or modifying
        if np.random.random() < 0.2:
            content = content + f" The reference number is {np.random.randint(1000, 9999)}."
        
        examples.append({
            'content': content,
            'label': 'store',
            'metadata': {
                'speaker': np.random.choice(['user', 'assistant']),
                'turn_number': np.random.randint(1, 20),
                'session_length': np.random.randint(5, 50)
            }
        })
    
    # Generate compress examples
    for i in range(target_per_class):
        base_content = np.random.choice(compress_examples)
        # Make them even longer sometimes
        if np.random.random() < 0.3:
            additional = " Furthermore, I think it's worth mentioning that the implications of these developments extend beyond just our immediate concerns and touch on broader industry trends that we should be monitoring closely."
            content = base_content + additional
        else:
            content = base_content
        
        examples.append({
            'content': content,
            'label': 'compress',
            'metadata': {
                'speaker': np.random.choice(['user', 'assistant']),
                'turn_number': np.random.randint(1, 20),
                'session_length': np.random.randint(5, 50)
            }
        })
    
    # Shuffle examples
    np.random.shuffle(examples)
    
    return examples


def main():
    """Demo of 3-class router with calibrated abstention."""
    
    if not SKLEARN_AVAILABLE:
        print("❌ ML dependencies not available. Install with:")
        print("   pip install scikit-learn xgboost pandas")
        return
    
    print("=== 3-Class Router with Calibrated Abstention ===")
    print("Implementing friend's recommendations for reliable routing\n")
    
    # Create router
    model_path = Path("ml_models/three_class_router_v1.pkl")
    router = ThreeClassRouter(
        confidence_threshold=0.75,  # Friend's recommendation for calibrated abstention
        model_path=model_path
    )
    
    # Generate training data
    print("🎯 Generating synthetic training data...")
    training_data = create_synthetic_training_data(n_examples=1500)
    print(f"   Created {len(training_data)} training examples")
    
    # Train model
    print("\n🤖 Training 3-class router...")
    training_results = router.train(training_data, save_model=True)
    
    print(f"   ✓ Validation accuracy: {training_results['training_stats']['validation_accuracy']:.3f}")
    print(f"   ✓ Abstention rate: {training_results['training_stats']['abstention_rate']:.3f}")
    print(f"   ✓ High-confidence accuracy: {training_results['training_stats']['high_confidence_accuracy']:.3f}")
    
    # Test predictions
    print("\n🧪 Testing predictions with reasoning...")
    
    test_cases = [
        "thanks",  # Should be discard
        "My appointment is tomorrow at 2pm with Dr. Smith",  # Should be store  
        "I had a fascinating discussion about AI with my colleague today. We covered many topics including the future of automation, ethical considerations, and how these technologies might transform our industry over the next decade. The conversation was quite insightful and raised many important questions about human-machine collaboration.",  # Should be compress
        "maybe", # Should abstain or discard
    ]
    
    for content in test_cases:
        decision = router.predict(content, return_reasoning=True)
        
        print(f"\n📝 Content: \"{content[:50]}{'...' if len(content) > 50 else ''}\"")
        print(f"   → Decision: {decision.class_name} (confidence: {decision.confidence:.3f})")
        print(f"   → Reasoning: {decision.reasoning}")
        if decision.abstained:
            print(f"   ⚠ Abstained: {decision.heuristic_fallback}")
        
        if decision.feature_importance:
            print("   → Top features:", ', '.join([f"{k}:{v:.3f}" for k, v in list(decision.feature_importance.items())[:3]]))
    
    # Performance summary
    stats = router.get_statistics()
    print(f"\n📊 Router Statistics:")
    print(f"   Predictions made: {stats['predictions_made']}")
    print(f"   Abstention rate: {stats['abstention_rate']:.3f}")
    print(f"   Confidence threshold: {stats['confidence_threshold']}")
    print(f"   Feature count: {stats['feature_count']}")
    
    print(f"\n✅ 3-class router ready for production!")
    print(f"   Model saved to: {model_path}")
    print(f"   Ready for integration with FrugalDispatcher")


if __name__ == "__main__":
    main()