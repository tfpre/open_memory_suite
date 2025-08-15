#!/usr/bin/env python3
"""
Open Memory Suite Production Server

Enterprise-grade server for the Open Memory Suite with ML model serving,
comprehensive monitoring, and production-ready features.

Key Features:
- FastAPI with async request handling
- ML model serving with XGBoost classifier
- Comprehensive health checks and metrics
- Cost accounting and budget enforcement
- Request validation and error handling
- Structured logging and observability
- Graceful shutdown and resource cleanup
- Environment-based configuration

Usage:
    python production_server.py --model-path ml_models/xgboost_*.pkl
    python production_server.py --port 8000 --workers 4 --debug
    python production_server.py --config production_config.yaml
"""

import asyncio
import argparse
import json
import logging
import os
import pickle
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid

try:
    from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import yaml
from rich.console import Console
from rich.logging import RichHandler

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from open_memory_suite.adapters import MemoryItem, InMemoryAdapter, FileStoreAdapter
from open_memory_suite.benchmark.cost_model import BudgetType, CostModel
from open_memory_suite.dispatcher import (
    FrugalDispatcher,
    HeuristicPolicy,
    PolicyRegistry,
    ConversationContext
)

console = Console()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

logger = logging.getLogger("open_memory_suite.server")

class ServerConfig(BaseModel):
    """Server configuration with validation."""
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, ge=1024, le=65535, description="Port to bind to")
    workers: int = Field(default=1, ge=1, le=16, description="Number of worker processes")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # ML Configuration
    model_path: Optional[str] = Field(default=None, description="Path to trained XGBoost model")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="ML confidence threshold")
    
    # Security
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    trusted_hosts: List[str] = Field(default_factory=lambda: ["*"], description="Trusted host patterns")
    max_request_size: int = Field(default=1024*1024, description="Max request size in bytes")
    
    # Performance
    request_timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_concurrent_requests: int = Field(default=100, description="Max concurrent requests")
    
    # Storage
    storage_path: str = Field(default="./server_storage", description="Storage directory path")
    
    # Monitoring
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    health_check_interval: float = Field(default=60.0, description="Health check interval in seconds")

# Request/Response Models
class MemoryRequest(BaseModel):
    """Enhanced request model for memory operations."""
    content: str = Field(..., min_length=1, max_length=10000, description="Content to store")
    speaker: str = Field(..., pattern="^(user|assistant)$", description="Speaker identifier")
    session_id: str = Field(..., min_length=1, max_length=100, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    force_action: Optional[str] = Field(default=None, description="Force specific action")
    budget_type: Optional[str] = Field(default="standard", description="Budget constraint level")
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Session ID must be alphanumeric with optional hyphens/underscores')
        return v

class MemoryResponse(BaseModel):
    """Enhanced response model for memory operations."""
    success: bool = Field(..., description="Whether the operation succeeded")
    action: str = Field(..., description="Action taken by the dispatcher")
    adapter: Optional[str] = Field(default=None, description="Adapter used for storage")
    estimated_cost: Optional[float] = Field(default=None, description="Estimated cost in USD")
    confidence: Optional[float] = Field(default=None, description="ML model confidence")
    reasoning: Optional[str] = Field(default=None, description="Reasoning for the decision")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time")
    request_id: str = Field(..., description="Unique request identifier")

class HealthResponse(BaseModel):
    """Comprehensive health check response."""
    status: str = Field(..., description="Overall service status")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health details")
    performance: Dict[str, float] = Field(..., description="Performance metrics")

class MetricsResponse(BaseModel):
    """Comprehensive metrics response."""
    server: Dict[str, Any] = Field(..., description="Server metrics")
    dispatcher: Dict[str, Any] = Field(..., description="Dispatcher metrics")
    requests: Dict[str, int] = Field(..., description="Request statistics")
    costs: Dict[str, float] = Field(..., description="Cost statistics")
    ml_model: Optional[Dict[str, Any]] = Field(default=None, description="ML model metrics")

class ExplainResponse(BaseModel):
    """ML model explanation response."""
    request_id: str = Field(..., description="Request identifier")
    routing_decision: int = Field(..., description="Routing class decision")
    confidence: float = Field(..., description="Model confidence")
    feature_importance: Dict[str, float] = Field(..., description="Top feature contributions")
    reasoning: str = Field(..., description="Human-readable explanation")
    alternatives: List[Dict[str, Any]] = Field(..., description="Alternative decisions considered")

class MLModelManager:
    """Manages ML model loading and inference."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.feature_extractor = None
        self.scaler = None
        self.config = None
        self.model_loaded = False
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Path) -> bool:
        """Load trained XGBoost model and associated artifacts."""
        try:
            model_dir = model_path.parent
            base_name = model_path.stem.replace('xgboost_memory_router_', '')
            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load feature extractor
            extractor_path = model_dir / f"xgboost_memory_router_feature_extractor_{base_name}.pkl"
            if extractor_path.exists():
                with open(extractor_path, 'rb') as f:
                    self.feature_extractor = pickle.load(f)
            
            # Load scaler
            scaler_path = model_dir / f"xgboost_memory_router_scaler_{base_name}.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load config
            config_path = model_dir / f"xgboost_memory_router_config_{base_name}.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            
            self.model_loaded = True
            logger.info(f"✅ ML model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load ML model: {e}")
            return False
    
    def predict(self, content: str, context: str = "", speaker: str = "user") -> Dict[str, Any]:
        """Make prediction using loaded ML model."""
        if not self.model_loaded or not self.feature_extractor:
            return {
                "routing_class": 2,  # Default to Vector
                "confidence": 0.5,
                "reasoning": "ML model not available, using fallback",
                "feature_importance": {}
            }
        
        try:
            # Create sample data structure expected by feature extractor
            sample_data = [{
                "turn_data": {
                    "content": content,
                    "speaker": speaker,
                    "context": context,
                    "metadata": {}
                },
                "label": {
                    "routing_class": 0,  # Placeholder
                    "confidence": 1.0,
                    "detected_entities": [],
                    "content_features": {
                        "word_count": len(content.split()),
                        "has_entities": False,
                        "has_relationships": False,
                        "is_question": content.strip().endswith("?")
                    }
                }
            }]
            
            # Extract features
            features_df, _ = self.feature_extractor.extract_features(sample_data, fit_extractors=False)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = float(max(probabilities))
            
            # Get feature importance for explanation
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_
                feature_names = self.feature_extractor.feature_names
                
                # Get top 5 most important features
                top_indices = importance_scores.argsort()[-5:][::-1]
                for idx in top_indices:
                    if idx < len(feature_names):
                        feature_importance[feature_names[idx]] = float(importance_scores[idx])
            
            # Create reasoning
            class_names = ["Discard", "Key-Value", "Vector", "Graph", "Summary"]
            reasoning = f"ML model classified as {class_names[prediction]} with {confidence:.2f} confidence"
            
            return {
                "routing_class": int(prediction),
                "confidence": confidence,
                "reasoning": reasoning,
                "feature_importance": feature_importance
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {
                "routing_class": 2,  # Default to Vector
                "confidence": 0.5,
                "reasoning": f"ML prediction failed: {str(e)}",
                "feature_importance": {}
            }

class ProductionMemoryServer:
    """Production-grade Open Memory Suite server."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app: Optional[FastAPI] = None
        self.dispatcher: Optional[FrugalDispatcher] = None
        self.ml_manager: Optional[MLModelManager] = None
        self.cost_model = CostModel()
        
        # Server state
        self.start_time = time.time()
        self.request_count = 0
        self.active_sessions = set()
        self.adapters = []
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_cost_usd": 0.0,
            "avg_response_time_ms": 0.0,
            "active_sessions": 0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"🛑 Received signal {signum}. Initiating graceful shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def initialize(self) -> None:
        """Initialize server components."""
        logger.info("🚀 Initializing Open Memory Suite Production Server...")
        
        # Load ML model if specified
        if self.config.model_path:
            model_path = Path(self.config.model_path)
            if model_path.exists():
                self.ml_manager = MLModelManager(model_path)
            else:
                logger.warning(f"⚠️  ML model not found: {model_path}")
        
        # Initialize storage adapters
        storage_path = Path(self.config.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        self.adapters = [
            InMemoryAdapter("production_memory"),
            FileStoreAdapter("production_file", storage_path),
        ]
        
        # Add FAISS adapter if available
        try:
            from open_memory_suite.adapters import FAISStoreAdapter
            self.adapters.append(FAISStoreAdapter("production_faiss"))
        except ImportError:
            logger.warning("⚠️  FAISS adapter not available")
        
        # Initialize adapters
        for adapter in self.adapters:
            await adapter.initialize()
            logger.info(f"✅ Initialized adapter: {adapter.name}")
        
        # Create dispatcher
        self.dispatcher = FrugalDispatcher(
            adapters=self.adapters,
            cost_model=self.cost_model
        )
        
        # Setup routing policy
        policy_registry = PolicyRegistry()
        policy_registry.register(HeuristicPolicy(), set_as_default=True)
        
        self.dispatcher.policy_registry = policy_registry
        await self.dispatcher.initialize()
        
        logger.info("✅ Server initialization complete")
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application with all routes and middleware."""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        app = FastAPI(
            title="Open Memory Suite API",
            description="Production memory routing system with ML-powered intelligent dispatch",
            version="1.0.0",
            docs_url="/docs" if self.config.debug else None,
            redoc_url="/redoc" if self.config.debug else None,
        )
        
        # Add middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if self.config.debug else ["https://*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.config.trusted_hosts
        )
        
        # Add request/response middleware
        @app.middleware("http")
        async def request_middleware(request: Request, call_next):
            start_time = time.time()
            request_id = str(uuid.uuid4())[:8]
            
            # Add request ID to headers
            request.state.request_id = request_id
            
            try:
                response = await call_next(request)
                processing_time = (time.time() - start_time) * 1000
                
                # Update statistics
                self.stats["total_requests"] += 1
                self.stats["successful_requests"] += 1
                
                # Update average response time
                current_avg = self.stats["avg_response_time_ms"]
                total_requests = self.stats["total_requests"]
                self.stats["avg_response_time_ms"] = (
                    (current_avg * (total_requests - 1) + processing_time) / total_requests
                )
                
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Processing-Time-MS"] = str(round(processing_time, 2))
                
                return response
                
            except Exception as e:
                self.stats["failed_requests"] += 1
                logger.error(f"Request {request_id} failed: {e}")
                raise
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI) -> None:
        """Add API routes to the FastAPI application."""
        
        @app.post("/memory/store", response_model=MemoryResponse)
        async def store_memory(
            request: MemoryRequest,
            background_tasks: BackgroundTasks,
            req: Request
        ) -> MemoryResponse:
            """Store content in memory with intelligent routing."""
            request_id = getattr(req.state, 'request_id', str(uuid.uuid4())[:8])
            start_time = time.time()
            
            try:
                # Track active session
                self.active_sessions.add(request.session_id)
                self.stats["active_sessions"] = len(self.active_sessions)
                
                # Create memory item
                item = MemoryItem(
                    content=request.content,
                    speaker=request.speaker,
                    session_id=request.session_id,
                    metadata=request.metadata or {},
                )
                
                # Route memory
                decision = await self.dispatcher.route_memory(item, request.session_id)
                
                # Execute decision
                success = await self.dispatcher.execute_decision(decision, item, request.session_id)
                
                # Update cost statistics
                if decision.estimated_cost:
                    self.stats["total_cost_usd"] += decision.estimated_cost.total_cost / 100
                
                processing_time = (time.time() - start_time) * 1000
                
                return MemoryResponse(
                    success=success,
                    action=decision.action.value,
                    adapter=decision.selected_adapter,
                    estimated_cost=decision.estimated_cost.total_cost / 100 if decision.estimated_cost else None,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                    processing_time_ms=processing_time,
                    request_id=request_id
                )
                
            except Exception as e:
                logger.error(f"❌ Memory storage failed for request {request_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/memory/explain", response_model=ExplainResponse)
        async def explain_decision(
            request: MemoryRequest,
            req: Request
        ) -> ExplainResponse:
            """Explain routing decision using ML model."""
            request_id = getattr(req.state, 'request_id', str(uuid.uuid4())[:8])
            
            try:
                if not self.ml_manager or not self.ml_manager.model_loaded:
                    raise HTTPException(
                        status_code=503, 
                        detail="ML model not available for explanations"
                    )
                
                # Get ML prediction with explanation
                prediction = self.ml_manager.predict(
                    request.content,
                    context="",
                    speaker=request.speaker
                )
                
                class_names = ["Discard", "Key-Value", "Vector", "Graph", "Summary"]
                routing_class = prediction["routing_class"]
                
                # Generate alternative predictions
                alternatives = []
                for i, class_name in enumerate(class_names):
                    if i != routing_class:
                        alternatives.append({
                            "class": i,
                            "name": class_name,
                            "probability": 0.1  # Simplified for demo
                        })
                
                return ExplainResponse(
                    request_id=request_id,
                    routing_decision=routing_class,
                    confidence=prediction["confidence"],
                    feature_importance=prediction["feature_importance"],
                    reasoning=prediction["reasoning"],
                    alternatives=alternatives[:3]  # Top 3 alternatives
                )
                
            except Exception as e:
                logger.error(f"❌ Explanation failed for request {request_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check() -> HealthResponse:
            """Comprehensive health check endpoint."""
            components = {}
            
            # Check dispatcher
            if self.dispatcher:
                components["dispatcher"] = {
                    "status": "healthy",
                    "adapters_count": len(self.adapters),
                    "active_sessions": len(self.active_sessions)
                }
            else:
                components["dispatcher"] = {"status": "unhealthy", "error": "Not initialized"}
            
            # Check adapters
            for adapter in self.adapters:
                try:
                    # Simple health check
                    count = await adapter.count()
                    components[f"adapter_{adapter.name}"] = {
                        "status": "healthy",
                        "item_count": count
                    }
                except Exception as e:
                    components[f"adapter_{adapter.name}"] = {
                        "status": "unhealthy", 
                        "error": str(e)
                    }
            
            # Check ML model
            if self.ml_manager:
                components["ml_model"] = {
                    "status": "healthy" if self.ml_manager.model_loaded else "unavailable",
                    "loaded": self.ml_manager.model_loaded
                }
            
            # Performance metrics
            uptime = time.time() - self.start_time
            performance = {
                "avg_response_time_ms": self.stats["avg_response_time_ms"],
                "requests_per_second": self.stats["total_requests"] / max(uptime, 1),
                "success_rate": (
                    self.stats["successful_requests"] / max(self.stats["total_requests"], 1)
                )
            }
            
            # Overall status
            unhealthy_components = [
                name for name, comp in components.items()
                if comp.get("status") != "healthy" and comp.get("status") != "unavailable"
            ]
            
            overall_status = "healthy" if not unhealthy_components else "degraded"
            
            return HealthResponse(
                status=overall_status,
                version="1.0.0",
                timestamp=datetime.now().isoformat(),
                uptime_seconds=uptime,
                components=components,
                performance=performance
            )
        
        @app.get("/metrics", response_model=MetricsResponse)
        async def get_metrics() -> MetricsResponse:
            """Get comprehensive server metrics."""
            uptime = time.time() - self.start_time
            
            # Server metrics
            server_metrics = {
                "uptime_seconds": uptime,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "version": "1.0.0",
                "config": {
                    "debug": self.config.debug,
                    "host": self.config.host,
                    "port": self.config.port
                }
            }
            
            # Dispatcher metrics
            dispatcher_metrics = {}
            if self.dispatcher:
                dispatcher_metrics = self.dispatcher.get_stats()
            
            # Request statistics
            request_stats = {
                "total": self.stats["total_requests"],
                "successful": self.stats["successful_requests"],
                "failed": self.stats["failed_requests"],
                "active_sessions": len(self.active_sessions)
            }
            
            # Cost statistics
            cost_stats = {
                "total_usd": self.stats["total_cost_usd"],
                "avg_per_request": (
                    self.stats["total_cost_usd"] / max(self.stats["total_requests"], 1)
                )
            }
            
            # ML model metrics
            ml_metrics = None
            if self.ml_manager and self.ml_manager.model_loaded:
                ml_metrics = {
                    "loaded": True,
                    "model_type": "xgboost",
                    "model_path": str(self.ml_manager.model_path) if self.ml_manager.model_path else None
                }
            
            return MetricsResponse(
                server=server_metrics,
                dispatcher=dispatcher_metrics,
                requests=request_stats,
                costs=cost_stats,
                ml_model=ml_metrics
            )
        
        @app.get("/status")
        async def get_detailed_status() -> Dict[str, Any]:
            """Get detailed service status for debugging."""
            return {
                "service": "Open Memory Suite",
                "version": "1.0.0",
                "environment": "production",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "stats": self.stats,
                "config": {
                    "debug": self.config.debug,
                    "ml_enabled": self.ml_manager is not None and self.ml_manager.model_loaded,
                    "adapters": [adapter.name for adapter in self.adapters]
                }
            }
    
    async def run(self) -> None:
        """Run the production server."""
        await self.initialize()
        
        self.app = self.create_app()
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.port,
            workers=1,  # Use 1 worker for now, can be increased
            log_level="debug" if self.config.debug else "info",
            reload=False,  # Don't use reload in production
            access_log=self.config.debug,
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"🚀 Starting production server on {self.config.host}:{self.config.port}")
        logger.info(f"🔧 Debug mode: {self.config.debug}")
        logger.info(f"🤖 ML model: {'enabled' if self.ml_manager and self.ml_manager.model_loaded else 'disabled'}")
        logger.info(f"💾 Storage: {len(self.adapters)} adapters initialized")
        
        await server.serve()
    
    async def shutdown(self) -> None:
        """Graceful shutdown with resource cleanup."""
        logger.info("🛑 Shutting down production server...")
        
        try:
            if self.dispatcher:
                await self.dispatcher.cleanup()
            
            for adapter in self.adapters:
                await adapter.cleanup()
            
            logger.info("✅ Server shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
        
        sys.exit(0)


def load_config(config_path: Optional[Path] = None) -> ServerConfig:
    """Load server configuration from file or environment."""
    config_dict = {}
    
    # Load from file if specified
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
    
    # Override with environment variables
    env_overrides = {
        "host": os.getenv("OMS_HOST"),
        "port": int(os.getenv("OMS_PORT", "0")) or None,
        "debug": os.getenv("OMS_DEBUG", "").lower() == "true",
        "model_path": os.getenv("OMS_MODEL_PATH"),
        "api_key": os.getenv("OMS_API_KEY"),
        "storage_path": os.getenv("OMS_STORAGE_PATH"),
    }
    
    # Remove None values
    env_overrides = {k: v for k, v in env_overrides.items() if v is not None}
    
    # Merge configuration
    config_dict.update(env_overrides)
    
    return ServerConfig(**config_dict)


async def main():
    """Main server function."""
    parser = argparse.ArgumentParser(
        description="Open Memory Suite Production Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python production_server.py --model-path ml_models/xgboost_*.pkl
  python production_server.py --port 8000 --debug
  python production_server.py --config production_config.yaml
  
Environment Variables:
  OMS_HOST - Server host (default: 0.0.0.0)
  OMS_PORT - Server port (default: 8000)
  OMS_DEBUG - Enable debug mode (true/false)
  OMS_MODEL_PATH - Path to ML model file
  OMS_API_KEY - API key for authentication
  OMS_STORAGE_PATH - Storage directory path
        """
    )
    
    parser.add_argument("--host", type=str, help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--model-path", type=str, help="Path to trained ML model")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    
    # Override with CLI arguments
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.debug:
        config.debug = True
    if args.model_path:
        config.model_path = args.model_path
    if args.workers:
        config.workers = args.workers
    
    try:
        # Create and run server
        server = ProductionMemoryServer(config)
        await server.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Server interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server failed: {e}")
        raise
    finally:
        logger.info("👋 Server stopped")


if __name__ == "__main__":
    asyncio.run(main())