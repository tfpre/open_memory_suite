#!/usr/bin/env python3
"""
Basic structure test to verify our implementation without external dependencies.
"""

import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that our core modules can be imported."""
    try:
        # Test core structure
        import open_memory_suite
        print("✅ Core package imports successfully")
        print(f"   Version: {open_memory_suite.__version__}")
        
        # Test that base classes are well-defined
        from open_memory_suite.adapters.base import MemoryAdapter, MemoryItem, RetrievalResult
        print("✅ Base adapter classes import successfully")
        
        # Test pydantic models
        test_item = MemoryItem(content="test")
        print(f"✅ MemoryItem creates successfully: {test_item.content}")
        
        # Test concrete adapter interface
        from open_memory_suite.adapters.faiss_store import FAISStoreAdapter
        adapter = FAISStoreAdapter("test")
        print(f"✅ FAISStoreAdapter instantiates: {adapter.name}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_file_structure():
    """Test that required files exist."""
    base_path = Path(__file__).parent.parent
    
    required_files = [
        "pyproject.toml",
        "open_memory_suite/__init__.py",
        "open_memory_suite/adapters/__init__.py", 
        "open_memory_suite/adapters/base.py",
        "open_memory_suite/adapters/faiss_store.py",
        "open_memory_suite/benchmark/__init__.py",
        "open_memory_suite/benchmark/trace.py",
        "open_memory_suite/benchmark/harness.py",
        "tests/conftest.py",
        "tests/test_adapters.py",
        "tests/test_trace.py",
        "tests/test_harness.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """Run basic tests."""
    print("🔍 Open Memory Suite - Basic Structure Test")
    print("=" * 50)
    
    print("\\n📁 Testing File Structure:")
    structure_ok = test_file_structure()
    
    print("\\n📦 Testing Imports:")
    imports_ok = test_imports()
    
    print("\\n🏆 Summary:")
    if structure_ok and imports_ok:
        print("✅ Basic implementation structure is correct!")
        print("✅ Core components are implemented and importable!")
        print("\\n🎯 Ready for M1 milestone verification once dependencies are installed.")
        return True
    else:
        print("❌ Some basic tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)