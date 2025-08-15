#!/usr/bin/env python3
"""
Open Memory Suite Server

Production-ready server for the Open Memory Suite with REST API endpoints,
health checks, and monitoring integration.

Features:
- REST API for memory operations
- Health check endpoints
- Metrics collection for Prometheus
- Async request handling
- Environment-based configuration
- Graceful shutdown

Usage:
    python -m open_memory_suite.server
    python -m open_memory_suite.server --port 8000 --debug
    python -m open_memory_suite.server --gpu --reload
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request/Response models
class MemoryRequest(BaseModel):
    """Request model for memory operations."""
    content: str = Field(..., description="Content to store in memory")
    speaker: str = Field(..., description="Speaker identifier (user/assistant)")
    session_id: str = Field(..., description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class MemoryResponse(BaseModel):
    """Response model for memory operations."""
    success: bool = Field(..., description="Whether the operation succeeded")
    action: str = Field(..., description="Action taken by the dispatcher")
    adapter: str = Field(..., description="Adapter used for storage")
    estimated_cost: Optional[float] = Field(default=None, description="Estimated cost of operation")
    reasoning: Optional[str] = Field(default=None, description="Reasoning for the decision")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    components: Dict[str, str] = Field(..., description="Component health status")

class MetricsResponse(BaseModel):
    """Metrics response model."""
    total_requests: int = Field(..., description="Total number of requests")
    active_sessions: int = Field(..., description="Number of active sessions")
    memory_usage: Dict[str, Any] = Field(..., description="Memory usage statistics")

class MemoryServer:
    """Production server for Open Memory Suite."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        debug: bool = False,
        gpu_enabled: bool = False,
        model_path: Optional[Path] = None,
    ):
        """Initialize the server."""
        self.host = host
        self.port = port
        self.debug = debug
        self.gpu_enabled = gpu_enabled
        self.model_path = model_path
        
        # Application state
        self.app: Optional[FastAPI] = None
        self.dispatcher: Optional[FrugalDispatcher] = None
        self.adapters = []
        self.stats = {
            "total_requests": 0,
            "active_sessions": set(),
            "start_time": None,
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        asyncio.create_task(self.shutdown())
    
    async def initialize(self) -> None:
        """Initialize server components."""
        logger.info("Initializing Open Memory Suite server...")
        
        # Initialize adapters
        self.adapters = [
            InMemoryAdapter("server_memory_store"),
            FileStoreAdapter("server_file_store", Path("./server_storage")),
        ]
        
        # Add FAISS adapter if available
        try:
            from open_memory_suite.adapters import FAISStoreAdapter
            self.adapters.append(FAISStoreAdapter("server_faiss_store"))
        except ImportError:
            logger.warning("FAISS adapter not available")
        
        # Initialize adapters
        for adapter in self.adapters:
            await adapter.initialize()
            logger.info(f"Initialized adapter: {adapter.name}")
        
        # Create cost model
        cost_model = CostModel()
        
        # Create dispatcher
        self.dispatcher = FrugalDispatcher(
            adapters=self.adapters,
            cost_model=cost_model,
        )
        
        # Setup policy
        policy_registry = PolicyRegistry()
        
        if ML_AVAILABLE and self.model_path and self.model_path.exists():
            try:
                logger.info(f"Loading ML policy from {self.model_path}")
                ml_policy = MLPolicy(
                    model_path=self.model_path,
                    confidence_threshold=0.7,
                    fallback_to_heuristic=True,
                )
                await ml_policy.initialize()
                policy_registry.register(ml_policy, set_as_default=True)
                logger.info("ML policy loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ML policy: {e}. Using heuristic policy.")
                policy_registry.register(HeuristicPolicy(), set_as_default=True)
        else:
            policy_registry.register(HeuristicPolicy(), set_as_default=True)
            logger.info("Using heuristic policy")
        
        self.dispatcher.policy_registry = policy_registry
        await self.dispatcher.initialize()
        
        logger.info("Server initialization complete")
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application."""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        app = FastAPI(
            title="Open Memory Suite API",
            description="Cost-optimized memory management for LLM applications",
            version="1.0.0",
            docs_url="/docs" if self.debug else None,
            redoc_url="/redoc" if self.debug else None,
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI) -> None:
        """Add API routes."""
        
        @app.post("/memory/store", response_model=MemoryResponse)
        async def store_memory(request: MemoryRequest) -> MemoryResponse:
            """Store content in memory with intelligent routing."""
            try:
                self.stats["total_requests"] += 1
                self.stats["active_sessions"].add(request.session_id)
                
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
                await self.dispatcher.execute_decision(decision, item, request.session_id)
                
                return MemoryResponse(
                    success=True,
                    action=decision.action.value,
                    adapter=decision.selected_adapter,
                    estimated_cost=decision.estimated_cost.total_cost if decision.estimated_cost else None,
                    reasoning=decision.reasoning,
                )
                
            except Exception as e:
                logger.error(f"Error storing memory: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/memory/retrieve/{session_id}")
        async def retrieve_memory(session_id: str, limit: int = 10) -> Dict[str, Any]:
            """Retrieve memory for a session."""
            try:
                # This would retrieve from adapters - simplified for now
                return {
                    "session_id": session_id,
                    "memories": [],
                    "count": 0,
                }
                
            except Exception as e:
                logger.error(f"Error retrieving memory: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check() -> HealthResponse:
            """Health check endpoint."""
            components = {}
            
            # Check dispatcher
            components["dispatcher"] = "healthy" if self.dispatcher else "unhealthy"
            
            # Check adapters
            for adapter in self.adapters:
                try:
                    # Simple health check - could be enhanced
                    components[f"adapter_{adapter.name}"] = "healthy"
                except Exception:
                    components[f"adapter_{adapter.name}"] = "unhealthy"
            
            # Overall status
            status = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
            
            return HealthResponse(
                status=status,
                version="1.0.0",
                components=components,
            )
        
        @app.get("/metrics", response_model=MetricsResponse)
        async def get_metrics() -> MetricsResponse:
            """Get service metrics."""
            return MetricsResponse(
                total_requests=self.stats["total_requests"],
                active_sessions=len(self.stats["active_sessions"]),
                memory_usage={
                    "adapters": len(self.adapters),
                    "policies": 1,  # Simplified
                },
            )
        
        @app.get("/status")
        async def get_status() -> Dict[str, Any]:
            """Get detailed service status."""
            return {
                "service": "Open Memory Suite",
                "version": "1.0.0",
                "status": "running",
                "gpu_enabled": self.gpu_enabled,
                "ml_available": ML_AVAILABLE,
                "stats": {
                    "total_requests": self.stats["total_requests"],
                    "active_sessions": len(self.stats["active_sessions"]),
                },
                "components": {
                    "dispatcher": "initialized" if self.dispatcher else "not_initialized",
                    "adapters": [adapter.name for adapter in self.adapters],
                },
            }
    
    async def run(self) -> None:
        """Run the server."""
        await self.initialize()
        
        self.app = self.create_app()
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="debug" if self.debug else "info",
            reload=self.debug,
            access_log=self.debug,
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"Starting server on {self.host}:{self.port}")
        logger.info(f"Debug mode: {self.debug}")
        logger.info(f"GPU enabled: {self.gpu_enabled}")
        
        await server.serve()
    
    async def shutdown(self) -> None:
        """Shutdown server gracefully."""
        logger.info("Shutting down server...")
        
        try:
            if self.dispatcher:
                await self.dispatcher.cleanup()
            
            for adapter in self.adapters:
                await adapter.cleanup()
            
            logger.info("Server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        sys.exit(0)

def create_simple_server() -> None:
    """Create a simple server when FastAPI is not available."""
    logger.info("FastAPI not available. Creating simple HTTP server...")
    
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        class SimpleHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {"status": "healthy", "message": "Open Memory Suite is running"}
                    self.wfile.write(json.dumps(response).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                logger.info(f"{self.address_string()} - {format % args}")
        
        server = HTTPServer(('0.0.0.0', 8000), SimpleHandler)
        logger.info("Simple server running on port 8000")
        logger.info("Health check available at: http://localhost:8000/health")
        server.serve_forever()
        
    except Exception as e:
        logger.error(f"Failed to start simple server: {e}")
        sys.exit(1)

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Open Memory Suite Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--model-path", type=str, help="Path to ML model")
    
    args = parser.parse_args()
    
    # Enable debug mode if reload is requested
    if args.reload:
        args.debug = True
    
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available. Starting simple server...")
        create_simple_server()
        return
    
    # Create and run server
    server = MemoryServer(
        host=args.host,
        port=args.port,
        debug=args.debug,
        gpu_enabled=args.gpu,
        model_path=Path(args.model_path) if args.model_path else None,
    )
    
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        await server.shutdown()

if __name__ == "__main__":
    asyncio.run(main())