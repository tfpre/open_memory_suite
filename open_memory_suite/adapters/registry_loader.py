"""
YAML loader for adapter instances (schema-tolerant).

Supported shapes:
1) Friend-style (flat list)
   adapters:
     - name: faiss_store
       type: FAISSStoreAdapter   # or "faiss" / "vector" alias
       args: { index_path: ".cache/faiss.index" }
       disabled: false

2) Rich schema (map of instances)
   instances:
     faiss_store:
       class: FAISSStoreAdapter
       args:
         index_path: ".cache/faiss.index"
       capabilities: [vector, semantic]

Env expansion: "${ENV_VAR:default}" inside strings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from .registry import AdapterRegistry
from .base import MemoryAdapter


_ALIASES = {
    "faiss": "FAISSStoreAdapter",
    "vector": "FAISSStoreAdapter",
    "memory": "MemoryStoreAdapter",
    "mem": "MemoryStoreAdapter",
    "file": "FileStoreAdapter",
    "fs": "FileStoreAdapter",
}


def _expand_env(val: Any) -> Any:
    if isinstance(val, str) and "${" in val:
        # very small, safe expansion: ${VAR:default}
        out = ""
        i = 0
        while i < len(val):
            if val[i : i + 2] == "${":
                j = val.find("}", i + 2)
                if j == -1:
                    out += val[i:]
                    break
                expr = val[i + 2 : j]
                if ":" in expr:
                    key, default = expr.split(":", 1)
                else:
                    key, default = expr, ""
                out += os.getenv(key, default)
                i = j + 1
            else:
                out += val[i]
                i += 1
        return out
    if isinstance(val, dict):
        return {k: _expand_env(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_expand_env(v) for v in val]
    return val


def _normalize_class(type_or_class: str) -> str:
    t = (type_or_class or "").strip()
    return _ALIASES.get(t.lower(), t)


def _mk_instance(class_name: str, instance_name: str, args: Dict[str, Any]) -> MemoryAdapter:
    if not class_name:
        raise ValueError(f"Adapter '{instance_name}' missing class/type")
    klass = _normalize_class(class_name)
    # Let AdapterRegistry resolve the actual class + construct
    return AdapterRegistry.create_adapter(adapter_name=klass, instance_name=instance_name, **(args or {}))


def _load_friend_style(doc: Dict[str, Any]) -> List[MemoryAdapter]:
    out: List[MemoryAdapter] = []
    for row in doc.get("adapters", []) or []:
        if row.get("disabled"):
            continue
        name = row.get("name") or row.get("instance") or row.get("id")
        klass = row.get("type") or row.get("class")
        args = _expand_env(row.get("args") or row.get("init") or {})
        if not name:
            raise ValueError("Adapter entry missing 'name'")
        out.append(_mk_instance(klass, name, args))
    return out


def _load_rich_style(doc: Dict[str, Any]) -> List[MemoryAdapter]:
    out: List[MemoryAdapter] = []
    instances = doc.get("instances") or {}
    for name, spec in instances.items():
        if spec.get("disabled"):
            continue
        klass = spec.get("class") or spec.get("type")
        args = _expand_env(spec.get("args") or {})
        out.append(_mk_instance(klass, name, args))
    return out


def load_adapters_from_yaml(path: Union[str, Path]) -> List[MemoryAdapter]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Adapters YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    # tolerant dispatcher: prefer rich style if present
    if "instances" in doc:
        return _load_rich_style(doc)
    if "adapters" in doc:
        return _load_friend_style(doc)
    # empty allowed → zero adapters
    return []
