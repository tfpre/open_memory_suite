"""
Anti-gaming spine: deterministic golden traces + manifest hashing.

- Four strata: factual_lookup / semantic_search / relationship_memory / conversation_flow
- Deterministic synthetic generator (so this runs out-of-the-box)
- Stable SHA256 manifest of all golden files
- Shadow split is produced in-memory only (never written), to prevent leakage
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

GOLDEN_DIR = Path("benchmark/golden")
MANIFEST = GOLDEN_DIR / "MANIFEST.sha256"

STRATA = (
    "factual_lookup",       # keyword/kv lookups
    "semantic_search",      # longer paraphrased questions
    "relationship_memory",  # entity-relationship joins
    "conversation_flow",    # chit-chat vs actionable discrimination
)


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class GoldenItem:
    """A single “memory + query” instance for evaluation."""
    id: str
    stratum: str
    content: str
    query: str
    # ground truth is the item's id; evaluator succeeds if retrieved item.id matches
    meta: Dict[str, str]


def _mk_factual(idx: int) -> GoldenItem:
    # simple KV style: "Order 4827 is ready" —> query: "status of order 4827?"
    order = 4000 + idx
    name = ["Alice", "Bob", "Carol", "Diego", "Erin"][idx % 5]
    content = f"Order {order} for {name} is ready for pickup at 5pm on 04/2{idx%10}/2025."
    query = f"What's the pickup time for order {order}?"
    return GoldenItem(
        id=f"F{idx}",
        stratum="factual_lookup",
        content=content,
        query=query,
        meta={"order": str(order), "name": name},
    )


def _mk_semantic(idx: int) -> GoldenItem:
    # longish paragraph + paraphrased question
    topic = ["pricing", "returns", "roadmap", "outage", "migration"][idx % 5]
    content = (
        f"Our updated {topic} policy clarifies edge-cases. Customers with annual plans "
        f"receive pro-rated credit if they downgrade within 30 days; otherwise credits "
        f"apply on the next billing cycle. This note supersedes prior guidance."
    )
    query = f"Can you summarize the new {topic} policy about billing credits?"
    return GoldenItem(
        id=f"S{idx}",
        stratum="semantic_search",
        content=content,
        query=query,
        meta={"topic": topic},
    )


def _mk_rel(idx: int) -> GoldenItem:
    # relationship: person -> company -> role
    person = ["Maya Patel", "Jun Park", "Liam Chen", "Nora Iqbal", "Omar Aziz"][idx % 5]
    company = ["Acme", "Globex", "Initech", "Soylent", "Umbrella"][idx % 5]
    role = ["PM", "Engineer", "Designer", "Analyst", "Support"][idx % 5]
    content = f"{person} joined {company} as a {role} on 2025-03-{10 + (idx % 10)}."
    query = f"Which company did {person} join?"
    return GoldenItem(
        id=f"R{idx}",
        stratum="relationship_memory",
        content=content,
        query=query,
        meta={"person": person, "company": company, "role": role},
    )


def _mk_flow(idx: int) -> GoldenItem:
    # conversational flow: some are chit-chat (should DROP), some actionable (should STORE)
    polite = ["thanks", "ok", "alright", "cool", "got it"][idx % 5]
    actionable = [
        "Book a dentist appointment for tomorrow at 2pm.",
        "Remind me to pay rent on the 1st.",
        "My email is user@example.com — store it.",
        "The router serial number is SN-8842.",
        "Flight DL210 departs at 8:10am from SFO.",
    ][idx % 5]
    content = actionable if (idx % 2 == 0) else polite
    query = "What actionable item should I schedule?" if (idx % 2 == 0) else "Is there anything to store here?"
    return GoldenItem(
        id=f"C{idx}",
        stratum="conversation_flow",
        content=content,
        query=query,
        meta={"actionable": str(idx % 2 == 0)},
    )


def _gen_stratum(stratum: str, n: int) -> List[GoldenItem]:
    mk = {
        "factual_lookup": _mk_factual,
        "semantic_search": _mk_semantic,
        "relationship_memory": _mk_rel,
        "conversation_flow": _mk_flow,
    }[stratum]
    return [mk(i) for i in range(n)]


def freeze_golden(seed: int = 1337, counts: Dict[str, int] | None = None) -> None:
    """
    Materialize deterministic golden traces on disk (jsonl per stratum) + manifest.
    Safe to call repeatedly; overwrites with the same data for a given seed.
    """
    _set_all_seeds(seed)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    counts = counts or {s: 64 for s in STRATA}

    manifest: Dict[str, str] = {}
    for s in STRATA:
        items = _gen_stratum(s, counts[s])
        path = GOLDEN_DIR / f"{s}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
        manifest[str(path)] = _sha256_file(path)

    with open(MANIFEST, "w", encoding="utf-8") as mf:
        mf.write(json.dumps({"seed": seed, "files": manifest}, indent=2))


def load_golden() -> Dict[str, List[GoldenItem]]:
    """Load golden traces from disk (freezing them first if missing)."""
    if not MANIFEST.exists():
        freeze_golden()
    out: Dict[str, List[GoldenItem]] = {}
    for s in STRATA:
        path = GOLDEN_DIR / f"{s}.jsonl"
        arr: List[GoldenItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                arr.append(GoldenItem(**obj))
        out[s] = arr
    return out


def make_shadow_split(golden: Dict[str, List[GoldenItem]], shadow_ratio: float = 0.25, seed: int = 1337) -> Dict[str, List[GoldenItem]]:
    """
    Produce an in-memory shadow split (never written to disk).
    We take a deterministic subset per stratum.
    """
    _set_all_seeds(seed + 1)
    shadow: Dict[str, List[GoldenItem]] = {}
    for s, items in golden.items():
        n = max(1, int(len(items) * shadow_ratio))
        idxs = list(range(len(items)))
        random.shuffle(idxs)
        chosen = [items[i] for i in idxs[:n]]
        shadow[s] = chosen
    return shadow


@dataclass
class GoldenManifest:
    seed: int
    files: Dict[str, str]

    @staticmethod
    def read() -> "GoldenManifest":
        data = json.loads(MANIFEST.read_text())
        return GoldenManifest(seed=data["seed"], files=data["files"])

    def verify(self) -> Tuple[bool, List[str]]:
        """Verify on-disk files match the manifest hashes. Returns (ok, mismatches)."""
        mismatches: List[str] = []
        for p, digest in self.files.items():
            if not Path(p).exists() or _sha256_file(Path(p)) != digest:
                mismatches.append(p)
        return (len(mismatches) == 0, mismatches)
