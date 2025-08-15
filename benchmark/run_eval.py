#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

# Route through the facade so we don’t care which underlying harness is active.
from .harness_router import BenchmarkHarness, HarnessConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic, cost-aware benchmark harness")
    p.add_argument("--framework", default="epmembench", choices=["epmembench", "longmemeval", "all"])
    p.add_argument("--adapter", default="faiss", choices=["memory", "faiss", "file", "graphlite"])
    p.add_argument("--policy", default="ml_router", choices=["naive", "heuristic", "ml_router"])
    p.add_argument("--budget", default="mid", choices=["low", "mid", "high", "unlimited"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--outdir", default="results")
    p.add_argument("--compare-baselines", action="store_true")
    p.add_argument("--engine", default="deterministic", choices=["deterministic", "research"],
                   help="Pick friend’s anti-gaming engine (default) or your research harness")
    p.add_argument("--adapters-yaml", default=None, help="Optional adapters YAML; loader is schema-tolerant")
    return p.parse_args()


def _print_table(metrics: dict) -> None:
    print("\n=== Results ===")
    print("policy                recall   abstain   p95(ms)   total¢     cost/✓")
    print("------------------  -------  --------  --------  --------  ---------")
    for name, m in metrics.items():
        print(
            f"{name:<18}  {m.get('recall', 0.0):.3f}    {m.get('abstention_rate', 0.0):.3f}    "
            f"{m.get('p95_latency_ms', 0.0):.1f}   {m.get('total_cost_cents', 0.0):.4f}   "
            f"{m.get('cost_per_correct', 0.0):.4f}"
        )


def main() -> None:
    args = parse_args()
    cfg = HarnessConfig(
        framework=args.framework,
        adapter=args.adapter,
        policy=args.policy,
        budget=args.budget,
        seed=args.seed,
        outdir=Path(args.outdir),
        compare_baselines=args.compare_baselines,
        engine=args.engine,
        adapters_yaml=args.adapters_yaml,
    )
    results = asyncio.run(BenchmarkHarness(cfg).run())

    # always print a compact table
    _print_table(results.get("metrics", {}))

    out_json = Path(args.outdir) / args.framework / "results.json"
    out_csv = Path(args.outdir) / args.framework / "pareto.csv"
    print(f"\n✅ results written:\n  - {out_json.resolve()}\n  - {out_csv.resolve()}")


if __name__ == "__main__":
    main()
