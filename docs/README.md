# Open Memory Suite Documentation

This directory contains project documentation, status updates, and design decisions.

## Documents

- **[status_update.md](./status_update.md)** - Current implementation status, completed features, issues, and detailed future plans
- **[architecture.md](./architecture.md)** - Technical architecture overview and design decisions *(coming soon)*
- **[api_reference.md](./api_reference.md)** - API documentation and usage examples *(coming soon)*

## Quick Navigation

### Current Status (July 29, 2025)
- ✅ **M1 Complete**: Core async adapter system with FAISS implementation  
- 🔄 **M2 Planning**: Cost-aware memory routing with multiple adapters
- 🎯 **Next**: InMemoryAdapter → Cost modeling → FrugalDispatcher

### Key Implementation Files
- Core: `open_memory_suite/adapters/` - Memory adapter implementations
- Benchmarking: `open_memory_suite/benchmark/` - Evaluation framework  
- Tests: `tests/` - 28 tests, 93% coverage
- Config: `pyproject.toml` - Python 3.12, PyTorch 2.4.0, FAISS 1.9.0

### External Research Context
- `external/epmembench/` - Episodic memory benchmarking research
- `external/longmemeval/` - Long-term memory evaluation framework
- `data/raw/persona_chat/` - Conversational datasets for testing

For detailed technical status and future roadmap, see [status_update.md](./status_update.md).