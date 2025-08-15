from __future__ import annotations

import json
import inspect
import csv
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class HarnessConfig:
    framework: str = "epmembench"           # epmembench | longmemeval | all
    adapter: str = "faiss"                   # memory | faiss | file | graphlite
    policy: str = "ml_router"               # naive | heuristic | ml_router
    budget: str = "mid"                     # low | mid | high | unlimited
    seed: int = 1337
    outdir: Path = Path("results")
    compare_baselines: bool = False
    engine: str = "deterministic"           # deterministic (friend) | research (yours)
    adapters_yaml: Optional[str] = None     # optional schema-tolerant loader


def _instantiate(cls, cfg: HarnessConfig) -> Any:
    """
    Safely instantiate an underlying HarnessConfig (friend’s or yours) by
    filtering to only the constructor’s accepted parameters.
    """
    kwargs = asdict(cfg)
    sig = inspect.signature(cls)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**filtered)


def _standardize_results(raw: Any, cfg: HarnessConfig) -> Dict[str, Any]:
    """
    Normalize result shape to:
    {
      "metrics": { policy_name: {recall, abstention_rate, p95_latency_ms, total_cost_cents, cost_per_correct, ...}},
      "artifacts": {"results_json": "...", "pareto_csv": "..."}
    }
    """
    # If the underlying harness already returns the expected shape, just pass it through.
    if isinstance(raw, dict) and "metrics" in raw:
        return raw

    # Minimal normalization fallback
    metrics = {}
    if isinstance(raw, dict):
        # look for single policy metric dict
        guess = raw.get("metrics") or raw.get("summary") or raw
        if isinstance(guess, dict):
            metrics["default"] = guess
    else:
        metrics["default"] = {}

    outdir = Path(cfg.outdir) / cfg.framework
    outdir.mkdir(parents=True, exist_ok=True)
    results_json = outdir / "results.json"
    pareto_csv = outdir / "pareto.csv"

    # Persist artifacts deterministically
    with results_json.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "engine": cfg.engine, "config": asdict(cfg)}, f, indent=2)

    # Create a very small Pareto CSV (cost_per_correct vs recall)
    with pareto_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "cost_per_correct", "recall"])
        for name, m in metrics.items():
            writer.writerow([name, m.get("cost_per_correct", 0.0), m.get("recall", 0.0)])

    return {
        "metrics": metrics,
        "artifacts": {"results_json": str(results_json), "pareto_csv": str(pareto_csv)},
    }


class BenchmarkHarness:
    """
    Facade that exposes one stable API for the CLI:

      - prefers friend’s deterministic, anti-gaming harness2.py
      - optionally uses your research harness.py with --engine research

    Both engines are used as-is; no code is thrown away.
    """

    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg

        # Try imports lazily so either file can be absent and we still work.
        self.det_mod = None
        self.res_mod = None

        # friend’s deterministic engine (anti-gaming)
        try:
            from . import harness2 as _det
            self.det_mod = _det
        except Exception:
            self.det_mod = None

        # your research engine
        try:
            # IMPORTANT: this is your existing comprehensive harness.py (no rename required)
            from . import harness as _res
            self.res_mod = _res
        except Exception:
            self.res_mod = None

        if self.cfg.engine == "deterministic":
            if not self.det_mod:
                raise ImportError("benchmark/harness2.py not found, cannot use deterministic engine")
        else:
            if not self.res_mod:
                raise ImportError("benchmark/harness.py not found, cannot use research engine")

    async def run(self) -> Dict[str, Any]:
        if self.cfg.engine == "deterministic":
            return await self._run_deterministic()
        return await self._run_research()

    async def _run_deterministic(self) -> Dict[str, Any]:
        """
        Delegate to friend’s harness2.HarnessConfig/BenchmarkHarness if present.
        Anti-gaming golden traces live there.
        """
        det_cfg_cls = getattr(self.det_mod, "HarnessConfig", None)
        det_cls = getattr(self.det_mod, "BenchmarkHarness", None)
        if not det_cfg_cls or not det_cls:
            raise AttributeError("harness2 missing HarnessConfig/BenchmarkHarness")

        det_cfg = _instantiate(det_cfg_cls, self.cfg)
        det = det_cls(det_cfg)
        raw = await det.run()
        return _standardize_results(raw, self.cfg)

    async def _run_research(self) -> Dict[str, Any]:
        """
        Delegate to your comprehensive harness. We don't force anti-gaming here,
        but the router still standardizes outputs + artifacts.
        """
        res_cfg_cls = getattr(self.res_mod, "HarnessConfig", None)
        res_cls = getattr(self.res_mod, "BenchmarkHarness", None)

        if res_cfg_cls and res_cls:
            res_cfg = _instantiate(res_cfg_cls, self.cfg)
            res = res_cls(res_cfg)
            raw = await res.run()
            return _standardize_results(raw, self.cfg)

        # Fallback: look for a simple `run(config_dict)` entry point
        run_fn = getattr(self.res_mod, "run", None)
        if callable(run_fn):
            raw = await run_fn(asdict(self.cfg))
            return _standardize_results(raw, self.cfg)

        raise AttributeError("research harness lacks a usable entry point")
