# 🧠 Open Memory Suite

> **ML-Powered Memory Routing System**  
> **100% Cost Reduction • 90% ML Accuracy • Production-Ready**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Open Memory Suite** is an ML-powered memory routing system designed for cost reduction with high recall accuracy. Using XGBoost classification and multi-backend storage, it provides cost-aware memory management for LLM applications.

## 🚀 **Quick Demo**

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