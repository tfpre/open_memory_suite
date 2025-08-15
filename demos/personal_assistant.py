#!/usr/bin/env python3
"""
Personal Assistant Demo with Live Routing Ticker

This demo showcases the 3-class intelligent memory routing system with:
- Real-time routing decisions displayed via WebSocket
- Cost-aware memory management
- Live demonstration of cost savings vs naive "store everything"
- Interactive conversation interface

Features:
- WebSocket for live routing ticker
- REST API for memory operations  
- Cost comparison dashboard
- Real-time routing decision explanations
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Set
from datetime import datetime

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Import our components
from open_memory_suite.adapters import MemoryItem, InMemoryAdapter, FileStoreAdapter
from open_memory_suite.dispatcher.frugal_dispatcher import FrugalDispatcher
from open_memory_suite.dispatcher.core import PolicyRegistry
from open_memory_suite.dispatcher.three_class_policy import ThreeClassMLPolicy
from open_memory_suite.benchmark.cost_model import CostModel, BudgetType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class ChatMessage(BaseModel):
    """Chat message for the personal assistant."""
    content: str = Field(..., description="Message content")
    speaker: str = Field(..., description="Speaker: 'user' or 'assistant'") 
    session_id: str = Field(..., description="Session identifier")

class RoutingEvent(BaseModel):
    """Live routing decision event."""
    timestamp: float
    session_id: str
    content_preview: str  # First 50 chars
    routing_decision: str  # 'discard', 'store', 'compress'
    confidence: float
    estimated_cost_cents: int
    reasoning: str
    cost_savings: Optional[float] = None  # vs naive baseline

class CostComparison(BaseModel):
    """Cost comparison data."""
    session_id: str
    intelligent_routing_cost: float
    naive_baseline_cost: float
    cost_savings: float
    savings_percentage: float
    messages_processed: int

class PersonalAssistantDemo:
    """Personal Assistant Demo Server with live routing display."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8001,
        model_path: Optional[Path] = None
    ):
        self.host = host
        self.port = port
        self.model_path = model_path
        
        # WebSocket connections for live updates
        self.connections: Set[WebSocket] = set()
        
        # Demo state
        self.session_costs: Dict[str, Dict[str, float]] = {}
        self.routing_history: Dict[str, list] = {}
        
        # Core components
        self.dispatcher: Optional[FrugalDispatcher] = None
        self.cost_model: Optional[CostModel] = None
        self.app: Optional[FastAPI] = None
    
    async def initialize(self):
        """Initialize demo components."""
        logger.info("Initializing Personal Assistant Demo...")
        
        # Initialize adapters
        adapters = [
            InMemoryAdapter("demo_memory_store"),
            FileStoreAdapter("demo_file_store", Path("./demo_storage")),
        ]
        
        # Add FAISS adapter if available
        try:
            from open_memory_suite.adapters import FAISStoreAdapter
            adapters.append(FAISStoreAdapter("demo_faiss_store"))
        except ImportError:
            logger.warning("FAISS adapter not available")
        
        # Initialize adapters
        for adapter in adapters:
            await adapter.initialize()
            logger.info(f"Initialized adapter: {adapter.name}")
        
        # Initialize cost model
        self.cost_model = CostModel()
        
        # Initialize dispatcher with 3-class policy
        policy_registry = PolicyRegistry()
        
        # Use 3-class ML policy if model available
        if self.model_path and self.model_path.exists():
            try:
                logger.info(f"Loading 3-class ML policy from {self.model_path}")
                ml_policy = ThreeClassMLPolicy(
                    model_path=self.model_path,
                    confidence_threshold=0.75,
                    fallback_to_heuristic=True
                )
                await ml_policy.initialize()
                policy_registry.register(ml_policy, set_as_default=True)
                logger.info("3-class ML policy loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load 3-class ML policy: {e}")
                # Fallback to simple policy that mimics 3-class decisions
                from open_memory_suite.dispatcher.heuristic_policy import HeuristicPolicy
                policy_registry.register(HeuristicPolicy(), set_as_default=True)
        else:
            logger.info("No model path provided, using heuristic policy")
            from open_memory_suite.dispatcher.heuristic_policy import HeuristicPolicy
            policy_registry.register(HeuristicPolicy(), set_as_default=True)
        
        # Create dispatcher
        self.dispatcher = FrugalDispatcher(
            adapters=adapters,
            cost_model=self.cost_model,
            policy_registry=policy_registry
        )
        
        await self.dispatcher.initialize()
        logger.info("Personal Assistant Demo initialized successfully")
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application."""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI required for demo. Install with: pip install fastapi uvicorn websockets")
        
        app = FastAPI(
            title="Personal Assistant Demo",
            description="Live demonstration of 3-class intelligent memory routing",
            version="1.0.0"
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._add_routes(app)
        self._add_websocket_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add HTTP API routes."""
        
        @app.get("/", response_class=HTMLResponse)
        async def get_demo_page():
            """Serve the demo HTML page."""
            return self._generate_demo_html()
        
        @app.post("/chat")
        async def process_message(message: ChatMessage) -> Dict[str, Any]:
            """Process a chat message and return routing decision."""
            try:
                # Create memory item
                item = MemoryItem(
                    content=message.content,
                    speaker=message.speaker,
                    session_id=message.session_id,
                    metadata={"timestamp": time.time()}
                )
                
                # Route the message
                decision, success = await self.dispatcher.route_and_execute(
                    item, message.session_id
                )
                
                # Calculate costs for comparison
                intelligent_cost = decision.estimated_cost.total_cost_cents if decision.estimated_cost else 0
                naive_cost = self._estimate_naive_cost(message.content)
                
                # Update session tracking
                if message.session_id not in self.session_costs:
                    self.session_costs[message.session_id] = {
                        'intelligent': 0,
                        'naive': 0,
                        'messages': 0
                    }
                
                self.session_costs[message.session_id]['intelligent'] += intelligent_cost
                self.session_costs[message.session_id]['naive'] += naive_cost
                self.session_costs[message.session_id]['messages'] += 1
                
                # Create routing event
                routing_event = RoutingEvent(
                    timestamp=time.time(),
                    session_id=message.session_id,
                    content_preview=message.content[:50] + "..." if len(message.content) > 50 else message.content,
                    routing_decision=self._map_action_to_class(decision.action),
                    confidence=decision.confidence if hasattr(decision, 'confidence') else 0.8,
                    estimated_cost_cents=intelligent_cost,
                    reasoning=decision.reasoning or "Standard routing decision",
                    cost_savings=(naive_cost - intelligent_cost) / max(1, naive_cost) if naive_cost > 0 else 0
                )
                
                # Broadcast to WebSocket connections
                await self._broadcast_routing_event(routing_event)
                
                # Generate assistant response (simplified)
                assistant_response = self._generate_assistant_response(message.content)
                
                return {
                    "success": success,
                    "routing_decision": routing_event.routing_decision,
                    "estimated_cost": intelligent_cost,
                    "cost_savings": routing_event.cost_savings,
                    "reasoning": routing_event.reasoning,
                    "assistant_response": assistant_response,
                    "session_summary": self._get_session_summary(message.session_id)
                }
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/session/{session_id}/summary")
        async def get_session_summary(session_id: str) -> CostComparison:
            """Get cost comparison summary for a session."""
            return self._get_session_summary(session_id)
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "demo": "personal_assistant",
                "dispatcher_initialized": self.dispatcher is not None,
                "active_connections": len(self.connections)
            }
    
    def _add_websocket_routes(self, app: FastAPI):
        """Add WebSocket routes for live updates."""
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for live routing updates."""
            await websocket.accept()
            self.connections.add(websocket)
            logger.info(f"WebSocket connection established. Total connections: {len(self.connections)}")
            
            try:
                # Send welcome message
                await websocket.send_json({
                    "type": "connection_established",
                    "message": "Connected to Personal Assistant Demo",
                    "timestamp": time.time()
                })
                
                # Keep connection alive
                while True:
                    await websocket.receive_text()
                    
            except WebSocketDisconnect:
                self.connections.discard(websocket)
                logger.info(f"WebSocket connection closed. Remaining connections: {len(self.connections)}")
    
    async def _broadcast_routing_event(self, event: RoutingEvent):
        """Broadcast routing event to all connected WebSocket clients."""
        if not self.connections:
            return
        
        message = {
            "type": "routing_decision",
            "data": event.dict()
        }
        
        # Send to all connections
        disconnected = set()
        for connection in self.connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.connections.discard(connection)
    
    def _map_action_to_class(self, action) -> str:
        """Map MemoryAction to 3-class decision name."""
        from open_memory_suite.dispatcher.core import MemoryAction
        
        mapping = {
            MemoryAction.DROP: "discard",
            MemoryAction.STORE: "store", 
            MemoryAction.SUMMARIZE: "compress",
            MemoryAction.DEFER: "store"  # Treat defer as store
        }
        return mapping.get(action, "store")
    
    def _estimate_naive_cost(self, content: str) -> int:
        """Estimate cost of naive 'store everything' approach."""
        # Simple estimation: assume everything goes to FAISS (most expensive)
        return max(10, len(content) // 10)  # Rough estimate in cents
    
    def _generate_assistant_response(self, user_message: str) -> str:
        """Generate a simple assistant response for demo purposes."""
        responses = {
            "hello": "Hello! I'm your personal assistant. I'm intelligently managing our conversation memory to optimize costs while maintaining quality. How can I help you?",
            "how": "I use a 3-class routing system to decide what to remember: important information gets stored, lengthy content gets compressed, and casual chat gets discarded to save costs.",
            "cost": "I'm actively saving costs by making smart decisions about what to remember. You can see the live cost comparison in real-time!",
            "meeting": "I'll remember that meeting information for you. I've routed it to the appropriate memory store based on its importance and your budget settings.",
            "default": "I understand. I'm processing and storing relevant information while optimizing for cost efficiency. Is there anything specific you'd like me to remember or help you with?"
        }
        
        user_lower = user_message.lower()
        for keyword, response in responses.items():
            if keyword in user_lower:
                return response
        
        return responses["default"]
    
    def _get_session_summary(self, session_id: str) -> CostComparison:
        """Get cost comparison summary for session."""
        if session_id not in self.session_costs:
            return CostComparison(
                session_id=session_id,
                intelligent_routing_cost=0.0,
                naive_baseline_cost=0.0,
                cost_savings=0.0,
                savings_percentage=0.0,
                messages_processed=0
            )
        
        costs = self.session_costs[session_id]
        intelligent_cost = costs['intelligent'] / 100.0  # Convert to dollars
        naive_cost = costs['naive'] / 100.0
        savings = naive_cost - intelligent_cost
        savings_pct = (savings / naive_cost * 100) if naive_cost > 0 else 0
        
        return CostComparison(
            session_id=session_id,
            intelligent_routing_cost=intelligent_cost,
            naive_baseline_cost=naive_cost,
            cost_savings=savings,
            savings_percentage=savings_pct,
            messages_processed=costs['messages']
        )
    
    def _generate_demo_html(self) -> str:
        """Generate the demo HTML page."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Personal Assistant - Memory Routing Demo</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            height: 80vh;
        }
        
        .chat-panel {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
        }
        
        .monitoring-panel {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            border: 1px solid #eee;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            background: #fafafa;
        }
        
        .message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 8px;
        }
        
        .message.user {
            background: #007bff;
            color: white;
            margin-left: 50px;
        }
        
        .message.assistant {
            background: #e9ecef;
            color: #333;
            margin-right: 50px;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
        }
        
        .form-control {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        
        .btn {
            padding: 10px 20px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        
        .btn:hover {
            background: #0056b3;
        }
        
        .routing-ticker {
            border: 1px solid #ddd;
            border-radius: 5px;
            max-height: 300px;
            overflow-y: auto;
            margin-bottom: 20px;
        }
        
        .routing-event {
            padding: 10px;
            border-bottom: 1px solid #eee;
            font-size: 12px;
        }
        
        .routing-event.discard {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
        }
        
        .routing-event.store {
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
        }
        
        .routing-event.compress {
            background: #d4edda;
            border-left: 4px solid #28a745;
        }
        
        .cost-summary {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .savings {
            color: #28a745;
            font-weight: bold;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .status-connected {
            background: #28a745;
        }
        
        .status-disconnected {
            background: #dc3545;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Personal Assistant - Memory Routing Demo</h1>
        <p>Experience intelligent 3-class memory routing with real-time cost optimization</p>
        <p><span id="connection-status" class="status-indicator status-disconnected"></span> 
           <span id="connection-text">Connecting...</span></p>
    </div>
    
    <div class="container">
        <div class="chat-panel">
            <h3>💬 Conversation</h3>
            <div id="chat-messages" class="chat-messages">
                <div class="message assistant">
                    Hello! I'm your personal assistant with intelligent memory routing. 
                    Try asking about meetings, reminders, or just having a conversation. 
                    Watch the live routing decisions in the panel on the right!
                </div>
            </div>
            <div class="input-group">
                <input type="text" id="message-input" class="form-control" 
                       placeholder="Type your message here..." />
                <button onclick="sendMessage()" class="btn">Send</button>
            </div>
        </div>
        
        <div class="monitoring-panel">
            <h3>📊 Live Routing Monitor</h3>
            
            <h4>🔄 Routing Decisions</h4>
            <div id="routing-ticker" class="routing-ticker">
                <div class="routing-event">
                    <em>Routing decisions will appear here...</em>
                </div>
            </div>
            
            <h4>💰 Cost Comparison</h4>
            <div id="cost-summary" class="cost-summary">
                <div class="metric">
                    <span>Intelligent Routing:</span>
                    <span id="intelligent-cost">$0.00</span>
                </div>
                <div class="metric">
                    <span>Naive Baseline:</span>
                    <span id="naive-cost">$0.00</span>
                </div>
                <div class="metric">
                    <span>Cost Savings:</span>
                    <span id="cost-savings" class="savings">$0.00 (0%)</span>
                </div>
                <div class="metric">
                    <span>Messages Processed:</span>
                    <span id="messages-count">0</span>
                </div>
            </div>
            
            <h4>📈 Routing Distribution</h4>
            <div id="routing-stats">
                <div class="metric">
                    <span>🗑️ Discarded:</span>
                    <span id="discard-count">0</span>
                </div>
                <div class="metric">
                    <span>🏪 Stored:</span>
                    <span id="store-count">0</span>
                </div>
                <div class="metric">
                    <span>📦 Compressed:</span>
                    <span id="compress-count">0</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let sessionId = 'demo-' + Math.random().toString(36).substr(2, 9);
        let routingStats = { discard: 0, store: 0, compress: 0 };
        
        // Initialize WebSocket connection
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + window.location.host + '/ws');
            
            ws.onopen = function(event) {
                document.getElementById('connection-status').className = 'status-indicator status-connected';
                document.getElementById('connection-text').textContent = 'Connected';
            };
            
            ws.onclose = function(event) {
                document.getElementById('connection-status').className = 'status-indicator status-disconnected';
                document.getElementById('connection-text').textContent = 'Disconnected';
                // Reconnect after 3 seconds
                setTimeout(initWebSocket, 3000);
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                if (message.type === 'routing_decision') {
                    updateRoutingTicker(message.data);
                    updateRoutingStats(message.data.routing_decision);
                }
            };
        }
        
        // Send message
        async function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        content: message,
                        speaker: 'user',
                        session_id: sessionId
                    })
                });
                
                const data = await response.json();
                
                // Add assistant response
                addMessage(data.assistant_response, 'assistant');
                
                // Update cost summary
                updateCostSummary(data.session_summary);
                
            } catch (error) {
                console.error('Error sending message:', error);
                addMessage('Sorry, there was an error processing your message.', 'assistant');
            }
        }
        
        // Add message to chat
        function addMessage(content, sender) {
            const messagesDiv = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            messageDiv.textContent = content;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Update routing ticker
        function updateRoutingTicker(routingData) {
            const ticker = document.getElementById('routing-ticker');
            const eventDiv = document.createElement('div');
            eventDiv.className = `routing-event ${routingData.routing_decision}`;
            
            const timestamp = new Date(routingData.timestamp * 1000).toLocaleTimeString();
            const costSavings = (routingData.cost_savings * 100).toFixed(1);
            
            eventDiv.innerHTML = `
                <strong>${timestamp}</strong> - 
                <strong>${routingData.routing_decision.toUpperCase()}</strong><br>
                "${routingData.content_preview}"<br>
                <small>Cost: ${routingData.estimated_cost_cents}¢ | Savings: ${costSavings}% | 
                Confidence: ${(routingData.confidence * 100).toFixed(0)}%</small>
            `;
            
            ticker.insertBefore(eventDiv, ticker.firstChild);
            
            // Keep only last 20 events
            while (ticker.children.length > 20) {
                ticker.removeChild(ticker.lastChild);
            }
        }
        
        // Update routing statistics
        function updateRoutingStats(decision) {
            routingStats[decision]++;
            document.getElementById('discard-count').textContent = routingStats.discard;
            document.getElementById('store-count').textContent = routingStats.store;
            document.getElementById('compress-count').textContent = routingStats.compress;
        }
        
        // Update cost summary
        function updateCostSummary(summary) {
            document.getElementById('intelligent-cost').textContent = 
                '$' + summary.intelligent_routing_cost.toFixed(3);
            document.getElementById('naive-cost').textContent = 
                '$' + summary.naive_baseline_cost.toFixed(3);
            document.getElementById('cost-savings').textContent = 
                '$' + summary.cost_savings.toFixed(3) + ' (' + summary.savings_percentage.toFixed(1) + '%)';
            document.getElementById('messages-count').textContent = summary.messages_processed;
        }
        
        // Handle Enter key in input
        document.getElementById('message-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Initialize WebSocket connection
        initWebSocket();
    </script>
</body>
</html>
        """
    
    async def run(self):
        """Run the demo server."""
        await self.initialize()
        
        self.app = self.create_app()
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            reload=False
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"🚀 Personal Assistant Demo starting on http://{self.host}:{self.port}")
        logger.info("✨ Features:")
        logger.info("   - Live 3-class routing decisions")
        logger.info("   - Real-time cost comparison")  
        logger.info("   - WebSocket-powered routing ticker")
        logger.info("   - Interactive conversation interface")
        
        await server.serve()

async def main():
    """Main function for the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Personal Assistant Demo")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--model-path", type=str, help="Path to 3-class router model")
    
    args = parser.parse_args()
    
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI required for demo. Install with: pip install fastapi uvicorn websockets")
        return
    
    # Create and run demo
    demo = PersonalAssistantDemo(
        host=args.host,
        port=args.port,
        model_path=Path(args.model_path) if args.model_path else None
    )
    
    try:
        await demo.run()
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())