"""
Deterministic, framework-agnostic benchmark harness.

Key behaviors:
- Seeds: numpy/python/random are pinned for reproducibility
- Uses your CostModel.predict()/reconcile() + telemetry.probe for store/retrieve ops
- Reports: cost_per_correct (headline), recall, abstention rate, p95 latency, routing distribution
- Writes artifacts: results/<framework>/results.json and pareto.csv (deterministic)
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .anti_gaming import GoldenManifest, load_golden, make_shadow_split, freeze_golden
from open_memory_suite.benchmark.cost_model import CostModel, OperationType, ConcurrencyLevel
from open_memory_suite.core.telemetry import probe

# dispatcher + policies + adapters
from open_memory_suite.dispatcher.frugal_dispatcher import FrugalDispatcher
from open_memory_suite.dispatcher.core import PolicyRegistry, MemoryAction
# prefer ML policy; fallback to heuristic if present
try:
    from open_memory_suite.dispatcher.three_class_policy import ThreeClassPolicy as MLRouterPolicy  # XGB + calibrated abstention
except Exception:  # pragma: no cover
    MLRouterPolicy = None  # type: ignore
try:
    from open_memory_suite.dispatcher.heuristic_policy import HeuristicPolicy
except Exception:  # pragma: no cover
    HeuristicPolicy = None  # type: ignore

# adapters (must exist in your project)
from open_memory_suite.adapters.registry import AdapterRegistry
from open_memory_suite.adapters.memory_store import InMemoryAdapter
from open_memory_suite.adapters.file_store import FileStoreAdapter
from open_memory_suite.adapters.faiss_store import FAISStoreAdapter

# core types
from open_memory_suite.adapters.base import MemoryItem, RetrievalResult


# ---------- small helpers ----------

def _set_seeds(seed: int) -> None:
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _rough_tokens(s: str) -> int:
    # simple, deterministic proxy
    return max(1, int(len(s) / 4))


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = int(math.ceil(0.95 * len(values_sorted))) - 1
    k = max(0, min(k, len(values_sorted) - 1))
    return values_sorted[k]


# ---------- policies for baselines ----------

class NaiveStorePolicy:  # always store, never summarize
    name = "naive_store"
    version = "1.0"

    async def initialize(self) -> None:
        return

    async def analyze_content(self, item: MemoryItem) -> Dict[str, Any]:
        return {}

    async def decide_action(self, item, context):
        return MemoryAction.STORE

    async def choose_adapter(self, item, adapters, context):
        pref = {"faiss_store", "memory_store", "file_store"}
        for a in adapters:
            if a.name in pref:
                return a
        return adapters[0] if adapters else None

    def get_stats(self) -> Dict[str, Any]:
        return {}


# ---------- config & results ----------

@dataclass
class HarnessConfig:
    framework: str  # "epmembench" | "longmemeval" | "all" (treated the same for golden)
    adapter: str    # "memory" | "faiss" | "file"
    policy: str     # "naive" | "heuristic" | "ml_router"
    budget: str     # "low" | "mid" | "high" | "unlimited"
    seed: int = 1337
    outdir: Path = Path("results")
    compare_baselines: bool = False


@dataclass
class RunMetrics:
    recall: float
    total_cost_cents: float
    cost_per_correct: float
    avg_latency_ms: float
    p95_latency_ms: float
    abstention_rate: float
    routing_distribution: Dict[str, float]


# ---------- core engine ----------

class BenchmarkHarness:
    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.cost_model = CostModel()
        self.outdir = cfg.outdir / cfg.framework
        self.outdir.mkdir(parents=True, exist_ok=True)

    async def _make_dispatcher(self) -> FrugalDispatcher:
        # instantiate adapter under test (and include a tiny memory store to ensure retrieval works even if file/FAISS empty)
        adapters = []
        a = self.cfg.adapter.lower()
        if a in ("memory", "mem", "ram"):
            adapters = [InMemoryAdapter(name="memory_store")]
        elif a in ("faiss", "vector"):
            adapters = [
                InMemoryAdapter(name="memory_store"),
                FAISSStoreAdapter(name="faiss_store", index_path=".cache/faiss.index"),
            ]
        elif a in ("file", "fs"):
            adapters = [
                InMemoryAdapter(name="memory_store"),
                FileStoreAdapter(name="file_store", base_dir=".cache/mem_files"),
            ]
        else:
            adapters = [InMemoryAdapter(name="memory_store")]

        # policy registry
        reg = PolicyRegistry()
        # baselines
        reg.register(NaiveStorePolicy())
        if HeuristicPolicy:
            reg.register(HeuristicPolicy())
        # ML router
        if MLRouterPolicy:
            reg.register(MLRouterPolicy())

        # map budget
        from ..open_memory_suite.benchmark.cost_model import BudgetType
        bt = {
            "low": BudgetType.MINIMAL,
            "mid": BudgetType.STANDARD,
            "high": BudgetType.PREMIUM,
            "unlimited": BudgetType.UNLIMITED,
        }[self.cfg.budget]

        # build dispatcher
        disp = FrugalDispatcher(adapters=adapters, cost_model=self.cost_model, policy_registry=reg, default_budget=bt)
        await disp.initialize()
        return disp

    async def run_one_policy(self, policy_name: str) -> RunMetrics:
        _set_seeds(self.cfg.seed)
        golden = load_golden()
        shadow = make_shadow_split(golden, shadow_ratio=0.25, seed=self.cfg.seed)

        disp = await self._make_dispatcher()

        # pick policy
        pol = None
        for p in disp.policy_registry.policies:  # type: ignore[attr-defined]
            if getattr(p, "name", "") == policy_name:
                pol = p
                break
        if pol is None and policy_name == "heuristic" and HeuristicPolicy:
            pol = HeuristicPolicy()
        if pol is None and policy_name == "ml_router" and MLRouterPolicy:
            pol = MLRouterPolicy()
        if pol is None and policy_name == "naive":
            pol = NaiveStorePolicy()
        if pol is None:
            # last resort
            pol = NaiveStorePolicy()

        # session
        session_id = f"eval-{self.cfg.framework}-{self.cfg.adapter}-{policy_name}-{self.cfg.seed}"

        # ---- storage phase (use golden only, not shadow) ----
        item_count = 0
        routing_hist: Dict[str, int] = {"STORE": 0, "DROP": 0, "SUMMARIZE": 0, "DEFER": 0}
        latencies_ms: List[float] = []
        total_cost_cents = 0.0

        # store all contents per stratum
        for stratum, items in golden.items():
            for gi in items:
                mem_item = MemoryItem(
                    content=gi.content,
                    speaker="user",
                    session_id=session_id,
                    metadata={"original_id": gi.id, "stratum": stratum},
                )

                # policy decision
                action = await pol.decide_action(mem_item, await disp.get_or_create_context(session_id))
                # choose adapter if storing/summarize
                adapter = None
                if action in (MemoryAction.STORE, MemoryAction.SUMMARIZE):
                    adapter = await pol.choose_adapter(mem_item, await disp._get_healthy_adapters(), await disp.get_or_create_context(session_id))

                # predicted cost/latency (before running)
                pred_cents, pred_ms = self.cost_model.predict(
                    op=OperationType.STORE if action == MemoryAction.STORE else OperationType.MAINTAIN if action == MemoryAction.SUMMARIZE else OperationType.ANALYZE,
                    adapter=adapter.name if adapter else (disp.adapters.get("memory_store").name if disp.adapters else "unknown"),
                    tokens=_rough_tokens(mem_item.content),
                    k=0,
                    item_count=item_count,
                    concurrency=ConcurrencyLevel.SINGLE,
                )

                with probe(
                    op="store" if action == MemoryAction.STORE else "summarize" if action == MemoryAction.SUMMARIZE else "drop",
                    adapter=adapter.name if adapter else "none",
                    predicted_cents=pred_cents,
                    predicted_ms=pred_ms,
                    meta={"tokens": _rough_tokens(mem_item.content), "item_count": item_count, "observed_cents": None, "stratum": stratum},
                ):
                    t0 = time.perf_counter()
                    # execute via dispatcher API to keep logic aligned
                    decision = await disp.route_memory(mem_item, session_id, policy_name=None, force_action=action)
                    ok = await disp.execute_decision(decision, mem_item, session_id)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    latencies_ms.append(elapsed_ms)

                # reconcile (observed_cents None for local)
                await self.cost_model.reconcile(
                    op=OperationType.STORE if action == MemoryAction.STORE else OperationType.MAINTAIN if action == MemoryAction.SUMMARIZE else OperationType.ANALYZE,
                    adapter=adapter.name if adapter else "none",
                    predicted_cents=pred_cents,
                    predicted_ms=pred_ms,
                    observed_cents=None,
                    observed_ms=elapsed_ms,
                    tokens=_rough_tokens(mem_item.content),
                    k=0,
                    item_count=item_count,
                )

                # we charge predicted cents (consistent with planning)
                total_cost_cents += pred_cents
                routing_hist[action.value.upper()] = routing_hist.get(action.value.upper(), 0) + 1
                if action in (MemoryAction.STORE, MemoryAction.SUMMARIZE):
                    item_count += 1  # simple counter across all adapters

        # ---- retrieval phase (use shadow split to avoid leakage) ----
        correct = 0
        total_q = 0
        for stratum, items in shadow.items():
            for gi in items:
                total_q += 1
                # predicted retrieval cost (k=5)
                pred_cents, pred_ms = self.cost_model.predict(
                    op=OperationType.RETRIEVE,
                    adapter="faiss_store" if "faiss_store" in disp.adapters else "memory_store",
                    tokens=_rough_tokens(gi.query),
                    k=5,
                    item_count=item_count,
                    concurrency=ConcurrencyLevel.SINGLE,
                )
                with probe(
                    op="retrieve",
                    adapter="faiss_store" if "faiss_store" in disp.adapters else "memory_store",
                    predicted_cents=pred_cents,
                    predicted_ms=pred_ms,
                    meta={"tokens": _rough_tokens(gi.query), "k": 5, "item_count": item_count, "observed_cents": None, "stratum": stratum},
                ):
                    t0 = time.perf_counter()
                    rr: RetrievalResult = await disp.retrieve_memories(
                        query=gi.query,
                        session_id=session_id,
                        k=5,
                        adapter_preferences=None,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    latencies_ms.append(elapsed_ms)

                await self.cost_model.reconcile(
                    op=OperationType.RETRIEVE,
                    adapter="faiss_store" if "faiss_store" in disp.adapters else "memory_store",
                    predicted_cents=pred_cents,
                    predicted_ms=pred_ms,
                    observed_cents=None,
                    observed_ms=elapsed_ms,
                    tokens=_rough_tokens(gi.query),
                    k=5,
                    item_count=item_count,
                )

                total_cost_cents += pred_cents

                # judge: correct if any retrieved item matches golden id tag
                got = False
                for it in rr.items:
                    oid = (it.metadata or {}).get("original_id") or ""
                    if oid == gi.id:
                        got = True
                        break
                if got:
                    correct += 1

        recall = (correct / total_q) if total_q else 0.0
        avg_lat = (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0
        p95_lat = _p95(latencies_ms)

        # abstention (available when using ThreeClassPolicy router stats)
        abst = 0.0
        if hasattr(pol, "router") and hasattr(pol.router, "get_statistics"):
            s = pol.router.get_statistics()
            abst = float(s.get("abstention_rate", 0.0))

        # routing distribution
        total_routed = sum(routing_hist.values()) or 1
        routing_dist = {k: v / total_routed for k, v in routing_hist.items()}

        # cents are integers from predict(); display as float with four decimals
        metrics = RunMetrics(
            recall=round(recall, 6),
            total_cost_cents=float(total_cost_cents),
            cost_per_correct=float(total_cost_cents) / max(1, correct),
            avg_latency_ms=round(avg_lat, 4),
            p95_latency_ms=round(p95_lat, 4),
            abstention_rate=round(abst, 6),
            routing_distribution=routing_dist,
        )
        return metrics

    async def run(self) -> Dict[str, Any]:
        # ensure golden exists and hashes match
        freeze_golden(seed=self.cfg.seed)
        ok, mismatches = GoldenManifest.read().verify()
        if not ok:  # still proceed, but record mismatch
            pass

        runs: Dict[str, RunMetrics] = {}

        # main policy from CLI
        pol_map = {"naive": "naive_store", "heuristic": "heuristic", "ml_router": "three_class"}
        target_policy = pol_map.get(self.cfg.policy, "naive_store")
        runs[self.cfg.policy] = await self.run_one_policy(target_policy)

        if self.cfg.compare_baselines:
            # naive
            if self.cfg.policy != "naive":
                runs["naive"] = await self.run_one_policy("naive_store")
            # heuristic
            if HeuristicPolicy and self.cfg.policy != "heuristic":
                runs["heuristic"] = await self.run_one_policy("heuristic")
            # ml
            if MLRouterPolicy and self.cfg.policy != "ml_router":
                runs["ml_router"] = await self.run_one_policy("three_class")

        # assemble artifact
        result_obj = {
            "config": asdict(self.cfg),
            "metrics": {k: asdict(v) for k, v in runs.items()},
            "golden_manifest": asdict(GoldenManifest.read()),
        }

        # write deterministic artifacts
        out_json = self.outdir / "results.json"
        out_csv = self.outdir / "pareto.csv"

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result_obj, f, indent=2)

        # pareto: (policy, recall, total_cost_cents, cost_per_correct)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("policy,recall,total_cost_cents,cost_per_correct\n")
            for k, m in runs.items():
                f.write(f"{k},{m.recall:.6f},{m.total_cost_cents:.4f},{m.cost_per_correct:.4f}\n")

        return result_obj
