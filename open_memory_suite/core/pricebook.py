"""Adaptive pricebook system for learned cost/latency coefficients."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List
import json
import math
import statistics
import time
import pathlib


@dataclass
class AdapterCoeffs:
    """Enhanced cost/latency coefficients with 5-component model from friend's recommendations."""
    
    # Friend's 5-component cost model (cents per unit)
    write: float = 0.05           # Cost to store new data (per item)
    read: float = 0.02            # Cost to retrieve data (per query)  
    index_maint: float = 0.01     # Cost for indexing and upkeep (per item per month)
    gc_maint: float = 0.005       # Cost for garbage collection (per item per GC cycle)
    storage_month: float = 0.001  # Storage cost per MB per month
    
    # Legacy compatibility (derived from core components for backward compatibility)
    base_cents: int = 5          # fixed overhead per op
    per_token_micros: int = 50   # microcents per token (int!)
    per_k_cents: float = 1.0     # retrieval scaling with k
    per_logN_cents: float = 2.0  # scaling with log(index/items)
    p50_ms: float = 45.0         # median latency baseline
    p95_ms: float = 120.0        # tail latency baseline
    
    def get_write_cost_cents(self, tokens: int = 100) -> float:
        """Get write cost using new 5-component model."""
        # Base write cost + token scaling + index maintenance amortized
        return self.write + (tokens * 0.0001) + (self.index_maint / 30)  # Monthly cost spread daily
    
    def get_read_cost_cents(self, k: int = 5, item_count: int = 1000) -> float:
        """Get read cost using new 5-component model."""
        # Base read cost + retrieval complexity + storage access cost
        import math
        log_factor = math.log(max(1, item_count), 10) * 0.001  # Logarithmic scaling
        return self.read + (k * 0.002) + log_factor
    
    def get_storage_cost_monthly(self, mb_stored: float = 1.0) -> float:
        """Get monthly storage cost using new model."""
        return self.storage_month * mb_stored
    
    def get_maintenance_cost_cents(self, item_count: int = 1000, gc_cycles_per_month: int = 4) -> float:
        """Get monthly maintenance costs (indexing + GC)."""
        monthly_index = self.index_maint * item_count
        monthly_gc = self.gc_maint * item_count * gc_cycles_per_month
        return monthly_index + monthly_gc


@dataclass
class Pricebook:
    """
    Learned cost/latency model persisted to disk.
    
    Keys are like "faiss_store|retrieve", "file_store|store", "summarize|model=gpt-4o"
    """
    entries: Dict[str, AdapterCoeffs]
    version: int
    updated_at: float

    @staticmethod
    def load(path: pathlib.Path) -> "Pricebook":
        """Load pricebook from disk or create empty if not found."""
        if not path.exists(): 
            return Pricebook(entries={}, version=1, updated_at=time.time())
        obj = json.loads(path.read_text())
        entries = {k: AdapterCoeffs(**v) for k, v in obj["entries"].items()}
        return Pricebook(
            entries=entries, 
            version=obj["version"], 
            updated_at=obj["updated_at"]
        )

    def save(self, path: pathlib.Path) -> None:
        """Save pricebook to disk atomically."""
        obj = {
            "entries": {k: asdict(v) for k, v in self.entries.items()},
            "version": self.version,
            "updated_at": time.time()
        }
        # Atomic save: write to temp file, then rename
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=2))
        tmp.replace(path)  # atomic on POSIX

    def key(self, adapter: str, op: str, model: str | None = None) -> str:
        """Generate pricebook key for adapter/operation/model combination."""
        return f"{adapter}|{op}" + (f"|model={model}" if model else "")


def fit_coeffs(samples: List[dict]) -> AdapterCoeffs:
    """
    Enhanced fitting with 5-component cost model from friend's recommendations.
    
    Separates cost fitting (API ops) from latency fitting (all ops) to prevent
    observed_cents=None from contaminating cost slopes.
    
    Args:
        samples: List of dicts with tokens, k, logN, observed_cents, observed_ms
    """
    def med_ratio(y, x, eps=1e-6):
        """Compute median of y/x ratios, handling division by zero."""
        vals = [(yi/(xi+eps)) for yi, xi in zip(y, x) if xi > 0]
        return statistics.median(vals) if vals else 0.0
    
    def robust_median(values, default=0.0):
        """Robust median with fallback."""
        return statistics.median(values) if values else default

    # CRITICAL FIX: Separate cost vs latency fitting by regime
    cost_samples = [s for s in samples if s["observed_cents"] is not None]
    lat_samples = [s for s in samples if s["observed_ms"] is not None]
    
    # Enhanced 5-component cost fitting (only from samples with real cost data)
    if cost_samples:
        # Determine operation type from samples to fit appropriate component
        write_samples = [s for s in cost_samples if s.get("op") == "store"]
        read_samples = [s for s in cost_samples if s.get("op") == "retrieve"]
        
        # Write costs (per item stored)
        if write_samples:
            write_costs = [s["observed_cents"] for s in write_samples]
            write_cost = robust_median(write_costs, 0.05)
        else:
            write_cost = 0.05  # Default from YAML config
            
        # Read costs (per query)
        if read_samples:
            read_costs = [s["observed_cents"] for s in read_samples] 
            read_cost = robust_median(read_costs, 0.02)
        else:
            read_cost = 0.02  # Default from YAML config
            
        # Maintenance costs (estimated from overall cost patterns)
        all_costs = [s["observed_cents"] for s in cost_samples]
        median_cost = robust_median(all_costs, 0.01)
        
        # Derive maintenance components from median costs
        index_maint_cost = min(0.02, median_cost * 0.1)  # 10% of median for index maintenance
        gc_maint_cost = min(0.01, median_cost * 0.05)    # 5% of median for GC
        storage_month_cost = min(0.005, median_cost * 0.02)  # 2% of median for storage
        
        # Legacy coefficient fitting for backward compatibility
        base_cents = round(statistics.median([s["observed_cents"] for s in cost_samples]))
        
        per_token_micros = round(1e6 * med_ratio(
            [max(0, s["observed_cents"] - base_cents) for s in cost_samples],
            [s.get("tokens", 0) for s in cost_samples]
        ))
        
        per_k_cents = med_ratio(
            [s["observed_cents"] for s in cost_samples],
            [max(1, s.get("k", 1)) for s in cost_samples]
        )
        
        per_logN_cents = med_ratio(
            [s["observed_cents"] for s in cost_samples],
            [max(1, math.log(max(1, s.get("item_count", 1)), 10)) for s in cost_samples]
        )
    else:
        # No cost data available (e.g., local FAISS) - use config defaults for 5-component model
        write_cost = 0.0
        read_cost = 0.0
        index_maint_cost = 0.0
        gc_maint_cost = 0.0
        storage_month_cost = 0.0
        
        # Legacy coefficients - zero for local operations
        base_cents = 0
        per_token_micros = 0
        per_k_cents = 0.0
        per_logN_cents = 0.0
    
    # Latency coefficient fitting (from all samples with timing data)
    if lat_samples:
        latencies = [s["observed_ms"] for s in lat_samples]
        p50_ms = statistics.median(latencies)
        
        # p95: use quantiles if available, else estimate
        if len(latencies) >= 20:
            p95_ms = statistics.quantiles(latencies, n=20)[18]
        else:
            p95_ms = max(latencies) if len(latencies) > 1 else p50_ms * 1.5
    else:
        # No latency data - conservative defaults
        p50_ms = 50.0
        p95_ms = 120.0
    
    return AdapterCoeffs(
        # New 5-component model
        write=float(write_cost),
        read=float(read_cost),
        index_maint=float(index_maint_cost),
        gc_maint=float(gc_maint_cost),
        storage_month=float(storage_month_cost),
        
        # Legacy compatibility
        base_cents=base_cents,
        per_token_micros=int(per_token_micros),
        per_k_cents=float(per_k_cents),
        per_logN_cents=float(per_logN_cents),
        p50_ms=float(p50_ms),
        p95_ms=float(p95_ms)
    )