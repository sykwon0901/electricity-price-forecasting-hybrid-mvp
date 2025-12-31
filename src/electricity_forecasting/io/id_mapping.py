"""ID scanning and stable ID<->code mapping.

IDs in the dataset can be strings (e.g., "Fri00Q1"). This module keeps raw IDs as strings
and provides deterministic integer codes for modeling/window_index.

This file also supports saving/loading the mapping to JSON so that:
  - window_index files can store integer codes only
  - DL/ML experiments can share the exact same mapping
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


def scan_unique_ids_from_parquet(parquet_path: str, id_col: str = "ID", batch_size: int = 1_000_000) -> List[str]:
    """Scan unique IDs from a Parquet dataset in a RAM-safe way."""
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(columns=[id_col], batch_size=batch_size)

    ids_set = set()
    for batch in scanner.to_batches():
        arr = batch.column(0).to_numpy(zero_copy_only=False)
        ids_set.update(np.unique(arr).tolist())

    return sorted([str(x) for x in ids_set])


def build_id_mapping(ids_raw: List[str]) -> Tuple[Dict[str, int], Dict[int, str], List[int]]:
    """Build stable raw<->code dictionaries and the full code list.

    Codes are assigned by sorting raw IDs lexicographically (deterministic).
    """
    ids_raw_sorted = sorted([str(x) for x in ids_raw])
    id_to_code = {rid: int(i) for i, rid in enumerate(ids_raw_sorted)}
    code_to_id = {int(i): rid for rid, i in id_to_code.items()}
    all_codes = list(range(len(ids_raw_sorted)))
    return id_to_code, code_to_id, all_codes


def save_id_mapping_json(path: str | Path, *, id_to_code: Dict[str, int], code_to_id: Dict[int, str]) -> None:
    """Save mapping to JSON (portable, human-readable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSON keys must be strings
    rec = {
        "version": 1,
        "ids_in_code_order": [code_to_id[i] for i in sorted(code_to_id.keys())],
        "id_to_code": {str(k): int(v) for k, v in id_to_code.items()},
        "code_to_id": {str(int(k)): str(v) for k, v in code_to_id.items()},
    }
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


def load_id_mapping_json(path: str | Path) -> Tuple[Dict[str, int], Dict[int, str], List[int]]:
    """Load mapping from JSON.

    Supports several possible schemas for robustness:
      1) {"id_to_code": {...}, "code_to_id": {...}}
      2) {"ids_in_code_order": [...]}
      3) {...} where keys are raw IDs and values are codes
      4) [...] a list of raw IDs in code order
    """
    path = Path(path)
    data: Any = json.loads(path.read_text(encoding="utf-8"))

    # Case 4: list -> ids_in_code_order
    if isinstance(data, list):
        ids = [str(x) for x in data]
        id_to_code = {rid: int(i) for i, rid in enumerate(ids)}
        code_to_id = {int(i): rid for rid, i in id_to_code.items()}
        all_codes = list(range(len(ids)))
        return id_to_code, code_to_id, all_codes

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported mapping JSON type: {type(data)}")

    # Case 2: ids_in_code_order
    if "ids_in_code_order" in data and isinstance(data["ids_in_code_order"], list):
        ids = [str(x) for x in data["ids_in_code_order"]]
        id_to_code = {rid: int(i) for i, rid in enumerate(ids)}
        code_to_id = {int(i): rid for rid, i in id_to_code.items()}
        all_codes = list(range(len(ids)))
        return id_to_code, code_to_id, all_codes

    # Case 1: explicit dicts
    if "id_to_code" in data and isinstance(data["id_to_code"], dict):
        id_to_code = {str(k): int(v) for k, v in data["id_to_code"].items()}
        if "code_to_id" in data and isinstance(data["code_to_id"], dict):
            code_to_id = {int(k): str(v) for k, v in data["code_to_id"].items()}
        else:
            code_to_id = {int(v): str(k) for k, v in id_to_code.items()}
        all_codes = sorted(code_to_id.keys())
        return id_to_code, code_to_id, all_codes

    # Case 3: assume raw->code mapping
    # (e.g., {"Fri00Q1": 0, ...})
    if all(isinstance(v, (int, float, str)) for v in data.values()):
        id_to_code = {str(k): int(v) for k, v in data.items()}
        code_to_id = {int(v): str(k) for k, v in id_to_code.items()}
        all_codes = sorted(code_to_id.keys())
        return id_to_code, code_to_id, all_codes

    raise ValueError(f"Could not parse mapping JSON schema in: {path}")


def load_or_build_id_mapping(
    *,
    mapping_json_path: str | Path,
    test_parquet_path: str,
    id_col: str = "ID",
    batch_size: int = 1_000_000,
) -> Tuple[Dict[str, int], Dict[int, str], List[int]]:
    """Load mapping JSON if exists, otherwise build from TEST parquet and save it."""
    mapping_json_path = Path(mapping_json_path)
    if mapping_json_path.exists():
        return load_id_mapping_json(mapping_json_path)

    ids_raw = scan_unique_ids_from_parquet(test_parquet_path, id_col=id_col, batch_size=batch_size)
    id_to_code, code_to_id, all_codes = build_id_mapping(ids_raw)
    save_id_mapping_json(mapping_json_path, id_to_code=id_to_code, code_to_id=code_to_id)
    return id_to_code, code_to_id, all_codes
