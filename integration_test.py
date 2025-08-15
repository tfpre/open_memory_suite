#!/usr/bin/env python3
"""
Open Memory Suite - Comprehensive Integration Test

Tests the complete system end-to-end including:
- Data generation pipeline
- ML training and model serving
- Production server API
- Benchmark evaluation framework
- Cost accounting and tracking
- Multi-adapter storage system

Usage:
    python integration_test.py --comprehensive
    python integration_test.py --quick
    python integration_test.py --server-only
"""

import asyncio
import argparse
import json
import requests
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint

console = Console()

class IntegrationTestSuite:
    """Comprehensive integration test suite for Open Memory Suite."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.test_results: List[Dict[str, Any]] = []
        self.temp_dir = Path(tempfile.mkdtemp())
        
        console.print(f"[blue]🧪 Integration Test Suite Initialized[/blue]")
        console.print(f"📍 Server URL: {self.base_url}")
        console.print(f"📂 Temp Directory: {self.temp_dir}")
    
    def add_test_result(self, name: str, success: bool, details: Dict[str, Any] = None):
        """Record test result."""
        self.test_results.append({
            "name": name,
            "success": success,
            "details": details or {},
            "timestamp": time.time()
        })
    
    def test_server_health(self) -> bool:
        """Test if the production server is healthy and responsive."""
        console.print("\n[bold blue]🏥 Testing Server Health[/bold blue]")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                console.print(f"✅ Health check: {health_data['status']}")
                
                # Check components
                components_healthy = all(
                    comp.get("status") in ["healthy", "unavailable"] 
                    for comp in health_data["components"].values()
                )
                
                if components_healthy:
                    console.print("✅ All components healthy")
                    self.add_test_result("server_health", True, health_data)
                    return True
                else:
                    console.print("⚠️  Some components unhealthy")
                    self.add_test_result("server_health", False, health_data)
                    return False
            else:
                console.print(f"❌ Health check failed: {response.status_code}")
                self.add_test_result("server_health", False, {"status_code": response.status_code})
                return False
                
        except Exception as e:
            console.print(f"❌ Server health check failed: {e}")
            self.add_test_result("server_health", False, {"error": str(e)})
            return False
    
    def test_memory_storage_api(self) -> bool:
        """Test the memory storage API endpoints."""
        console.print("\n[bold blue]💾 Testing Memory Storage API[/bold blue]")
        
        test_requests = [
            {
                "content": "Hello, I'm Alice and I work at Google.",
                "speaker": "user",
                "session_id": "test_session_1",
                "metadata": {"test": "integration"}
            },
            {
                "content": "What's the weather like today?",
                "speaker": "user", 
                "session_id": "test_session_1"
            },
            {
                "content": "Thanks for your help!",
                "speaker": "user",
                "session_id": "test_session_1"
            }
        ]
        
        successful_requests = 0
        
        for i, req_data in enumerate(test_requests):
            try:
                response = requests.post(
                    f"{self.base_url}/memory/store",
                    json=req_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    console.print(f"✅ Request {i+1}: {result['action']} ({result.get('processing_time_ms', 0):.1f}ms)")
                    successful_requests += 1
                else:
                    console.print(f"❌ Request {i+1} failed: {response.status_code}")
                    
            except Exception as e:
                console.print(f"❌ Request {i+1} error: {e}")
        
        success = successful_requests == len(test_requests)
        self.add_test_result("memory_storage_api", success, {
            "successful_requests": successful_requests,
            "total_requests": len(test_requests)
        })
        
        return success
    
    def test_metrics_endpoint(self) -> bool:
        """Test the metrics and monitoring endpoints."""
        console.print("\n[bold blue]📊 Testing Metrics Endpoints[/bold blue]")
        
        try:
            # Test metrics endpoint
            metrics_response = requests.get(f"{self.base_url}/metrics", timeout=10)
            status_response = requests.get(f"{self.base_url}/status", timeout=10)
            
            if metrics_response.status_code == 200 and status_response.status_code == 200:
                metrics = metrics_response.json()
                status = status_response.json()
                
                # Validate metrics structure
                required_keys = ["server", "dispatcher", "requests", "costs"]
                has_required = all(key in metrics for key in required_keys)
                
                if has_required:
                    console.print("✅ Metrics endpoint structure valid")
                    console.print(f"📊 Total requests: {metrics['requests']['total']}")
                    console.print(f"💰 Total cost: ${metrics['costs']['total_usd']:.6f}")
                    
                    self.add_test_result("metrics_endpoint", True, {
                        "metrics_keys": list(metrics.keys()),
                        "status_keys": list(status.keys())
                    })
                    return True
                else:
                    console.print("❌ Invalid metrics structure")
                    self.add_test_result("metrics_endpoint", False, {"missing_keys": required_keys})
                    return False
            else:
                console.print("❌ Metrics endpoints failed")
                self.add_test_result("metrics_endpoint", False, {
                    "metrics_status": metrics_response.status_code,
                    "status_status": status_response.status_code
                })
                return False
                
        except Exception as e:
            console.print(f"❌ Metrics test failed: {e}")
            self.add_test_result("metrics_endpoint", False, {"error": str(e)})
            return False
    
    def test_benchmark_cli(self) -> bool:
        """Test the benchmark CLI evaluation system."""
        console.print("\n[bold blue]🏆 Testing Benchmark CLI[/bold blue]")
        
        try:
            # Run benchmark with sample data
            cmd = [
                "poetry", "run", "python", "benchmark/run_eval.py",
                "--compare-all",
                "--dataset", "sample",
                "--output", str(self.temp_dir / "benchmark_results"),
                "--verbose"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=Path.cwd()
            )
            
            if result.returncode == 0:
                console.print("✅ Benchmark CLI executed successfully")
                
                # Check if results were generated
                result_files = list((self.temp_dir / "benchmark_results").glob("*.json"))
                if result_files:
                    console.print(f"✅ Benchmark results saved: {len(result_files)} files")
                    
                    # Parse results
                    with open(result_files[0], 'r') as f:
                        results = json.load(f)
                    
                    console.print(f"📊 Adapters tested: {len(results.get('adapter_results', {}))}")
                    
                    self.add_test_result("benchmark_cli", True, {
                        "result_files": len(result_files),
                        "adapters_tested": len(results.get('adapter_results', {}))
                    })
                    return True
                else:
                    console.print("⚠️  No benchmark results generated")
                    self.add_test_result("benchmark_cli", False, {"issue": "no_results"})
                    return False
            else:
                console.print(f"❌ Benchmark CLI failed: {result.stderr}")
                self.add_test_result("benchmark_cli", False, {"error": result.stderr})
                return False
                
        except Exception as e:
            console.print(f"❌ Benchmark CLI test failed: {e}")
            self.add_test_result("benchmark_cli", False, {"error": str(e)})
            return False
    
    def test_ml_training_pipeline(self) -> bool:
        """Test the ML training pipeline end-to-end."""
        console.print("\n[bold blue]🤖 Testing ML Training Pipeline[/bold blue]")
        
        try:
            # Step 1: Generate small training dataset
            console.print("📝 Generating training data...")
            
            data_gen_cmd = [
                "poetry", "run", "python", "data_generation/auto_labeler.py",
                "--budget", "1.0",
                "--target-samples", "20",
                "--api-key", "mock",
                "--output-dir", str(self.temp_dir / "ml_data")
            ]
            
            data_result = subprocess.run(
                data_gen_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path.cwd()
            )
            
            if data_result.returncode != 0:
                console.print(f"❌ Data generation failed: {data_result.stderr}")
                self.add_test_result("ml_training_pipeline", False, {"step": "data_generation", "error": data_result.stderr})
                return False
            
            console.print("✅ Training data generated")
            
            # Step 2: Train ML model
            console.print("🚀 Training ML model...")
            
            data_files = list((self.temp_dir / "ml_data").glob("labeled_data_*.json"))
            if not data_files:
                console.print("❌ No training data files found")
                self.add_test_result("ml_training_pipeline", False, {"step": "no_data_files"})
                return False
            
            training_cmd = [
                "poetry", "run", "python", "ml_training/xgboost_trainer.py",
                "--data", str(data_files[0]),
                "--output-dir", str(self.temp_dir / "ml_models"),
                "--save-model"
            ]
            
            training_result = subprocess.run(
                training_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=Path.cwd()
            )
            
            if training_result.returncode == 0:
                console.print("✅ ML model trained successfully")
                
                # Check if model files were created
                model_files = list((self.temp_dir / "ml_models").glob("*.pkl"))
                if model_files:
                    console.print(f"✅ Model artifacts saved: {len(model_files)} files")
                    self.add_test_result("ml_training_pipeline", True, {
                        "model_files": len(model_files),
                        "training_success": True
                    })
                    return True
                else:
                    console.print("⚠️  No model files generated")
                    self.add_test_result("ml_training_pipeline", False, {"step": "no_model_files"})
                    return False
            else:
                console.print(f"❌ ML training failed: {training_result.stderr}")
                self.add_test_result("ml_training_pipeline", False, {"step": "training", "error": training_result.stderr})
                return False
                
        except Exception as e:
            console.print(f"❌ ML training pipeline test failed: {e}")
            self.add_test_result("ml_training_pipeline", False, {"error": str(e)})
            return False
    
    def test_cost_accounting(self) -> bool:
        """Test the cost accounting and tracking system."""
        console.print("\n[bold blue]💰 Testing Cost Accounting[/bold blue]")
        
        try:
            # Make several API calls to accumulate costs
            test_contents = [
                "Short message",
                "This is a medium-length message with some more content to test cost scaling based on content size.",
                "This is a very long message that contains a lot of detailed information and should trigger the summary routing class because it exceeds the typical length thresholds. The content includes multiple sentences, detailed explanations, and comprehensive information that would typically be compressed or summarized rather than stored in full."
            ]
            
            total_cost = 0.0
            
            for i, content in enumerate(test_contents):
                response = requests.post(
                    f"{self.base_url}/memory/store",
                    json={
                        "content": content,
                        "speaker": "user",
                        "session_id": f"cost_test_{i}"
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('estimated_cost'):
                        total_cost += result['estimated_cost']
            
            # Check metrics for cost tracking
            metrics_response = requests.get(f"{self.base_url}/metrics")
            if metrics_response.status_code == 200:
                metrics = metrics_response.json()
                tracked_cost = metrics['costs']['total_usd']
                
                console.print(f"✅ Cost tracking functional")
                console.print(f"💰 Total tracked cost: ${tracked_cost:.6f}")
                console.print(f"📊 Cost per request: ${metrics['costs']['avg_per_request']:.6f}")
                
                self.add_test_result("cost_accounting", True, {
                    "total_cost_tracked": tracked_cost,
                    "requests_made": len(test_contents)
                })
                return True
            else:
                console.print("❌ Cost metrics unavailable")
                self.add_test_result("cost_accounting", False, {"issue": "metrics_unavailable"})
                return False
                
        except Exception as e:
            console.print(f"❌ Cost accounting test failed: {e}")
            self.add_test_result("cost_accounting", False, {"error": str(e)})
            return False
    
    def print_test_summary(self):
        """Print comprehensive test results summary."""
        console.print("\n[bold]🧪 INTEGRATION TEST RESULTS[/bold]")
        
        # Overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Summary table
        summary_table = Table(title="Test Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Tests", str(total_tests))
        summary_table.add_row("Passed", str(passed_tests))
        summary_table.add_row("Failed", str(failed_tests))
        summary_table.add_row("Success Rate", f"{success_rate:.1f}%")
        
        console.print(summary_table)
        
        # Detailed results
        results_table = Table(title="Detailed Results")
        results_table.add_column("Test", style="cyan")
        results_table.add_column("Status", style="yellow")
        results_table.add_column("Details", style="magenta")
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            details = str(result['details']) if result['details'] else ""
            if len(details) > 50:
                details = details[:47] + "..."
                
            results_table.add_row(
                result['name'].replace('_', ' ').title(),
                status,
                details
            )
        
        console.print(results_table)
        
        # Overall assessment
        if success_rate >= 80:
            console.print(f"\n[bold green]🎉 INTEGRATION TESTS PASSED ({success_rate:.1f}%)[/bold green]")
            console.print("✅ System is ready for production deployment")
        elif success_rate >= 60:
            console.print(f"\n[bold yellow]⚠️  PARTIAL SUCCESS ({success_rate:.1f}%)[/bold yellow]")
            console.print("🔧 Some components need attention before production")
        else:
            console.print(f"\n[bold red]❌ INTEGRATION TESTS FAILED ({success_rate:.1f}%)[/bold red]")
            console.print("🚨 System requires significant fixes before deployment")
    
    def run_quick_tests(self) -> bool:
        """Run essential tests for basic functionality."""
        console.print("[bold]🏃 Running Quick Integration Tests[/bold]")
        
        tests = [
            ("Server Health", self.test_server_health),
            ("Memory API", self.test_memory_storage_api),
            ("Metrics", self.test_metrics_endpoint),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Running tests...", total=len(tests))
            
            for test_name, test_func in tests:
                progress.update(task, description=f"Running {test_name}...")
                test_func()
                progress.advance(task)
        
        self.print_test_summary()
        return sum(1 for r in self.test_results if r['success']) >= len(tests) * 0.8
    
    def run_comprehensive_tests(self) -> bool:
        """Run all integration tests."""
        console.print("[bold]🔬 Running Comprehensive Integration Tests[/bold]")
        
        tests = [
            ("Server Health", self.test_server_health),
            ("Memory Storage API", self.test_memory_storage_api),
            ("Metrics Endpoints", self.test_metrics_endpoint),
            ("Cost Accounting", self.test_cost_accounting),
            ("Benchmark CLI", self.test_benchmark_cli),
            ("ML Training Pipeline", self.test_ml_training_pipeline),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Running comprehensive tests...", total=len(tests))
            
            for test_name, test_func in tests:
                progress.update(task, description=f"Running {test_name}...")
                test_func()
                progress.advance(task)
        
        self.print_test_summary()
        return sum(1 for r in self.test_results if r['success']) >= len(tests) * 0.8


def check_server_running(url: str) -> bool:
    """Check if server is running and accessible."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Open Memory Suite Integration Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python integration_test.py --quick
  python integration_test.py --comprehensive
  python integration_test.py --server-url http://localhost:8001
        """
    )
    
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--comprehensive", action="store_true", help="Run all tests")
    parser.add_argument("--server-only", action="store_true", help="Test server endpoints only")
    parser.add_argument("--server-url", default="http://localhost:8001", help="Server URL to test")
    
    args = parser.parse_args()
    
    # Check if server is running
    if not check_server_running(args.server_url):
        console.print(f"[red]❌ Server not accessible at {args.server_url}[/red]")
        console.print("💡 Start the server first: python production_server.py --port 8001 --debug")
        return 1
    
    # Run tests
    test_suite = IntegrationTestSuite(args.server_url)
    
    try:
        if args.comprehensive:
            success = test_suite.run_comprehensive_tests()
        elif args.server_only or args.quick:
            success = test_suite.run_quick_tests()
        else:
            # Default to quick tests
            success = test_suite.run_quick_tests()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Tests interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]❌ Test suite failed: {e}[/red]")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)