# 🧠 Open Memory Suite

> **ML-Powered Memory Routing System**  

**Open Memory Suite** is an ML-powered memory routing system designed for cost reduction with high recall accuracy. Using XGBoost classification and multi-backend storage, it provides cost-aware memory management for LLM applications.

## **Quick Demo**

```bash
poetry install
poetry run python production_server.py --port 8001

curl -X POST http://localhost:8001/memory/store \
  -H "Content-Type: application/json" \
  -d '{"content": "My name is Alice Chen", "speaker": "user", "session_id": "demo"}'

{
  "success": true,
  "action": "store", 
  "confidence": 0.87,
  "estimated_cost": 0.00012,
  "reasoning": "Contains factual personal information"
}

poetry run python simple_benchmark.py
```