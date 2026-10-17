#!/usr/bin/env python3
"""Gate a runtime upgrade using matching host-driver inference snapshots.

Compares actual token ids and logits, including optional bridge input tensors.
This verifies CPU reference inference; it does not execute HiAI or OMC graphs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(baseline, candidate):
    result = {"baseline": str(baseline), "candidate": str(candidate),
              "device_executed": False, "files": {}}
    required = ("tokens.bin", "logits.bin", "generated.bin")
    optional = ("positions.bin", "embeddings.bin", "reference_logits.bin", "decode_logits.bin")
    for name in required + optional:
        before, after = baseline / name, candidate / name
        if name in optional and not before.exists() and not after.exists():
            continue
        if not before.is_file() or not after.is_file():
            raise ValueError("Missing matching snapshot: " + name)
        integer = name in ("tokens.bin", "positions.bin", "generated.bin")
        dtype = np.int32 if integer else np.float32
        if before.stat().st_size % 4 or after.stat().st_size % 4:
            raise ValueError("Truncated snapshot: " + name)
        x, y = np.fromfile(before, dtype), np.fromfile(after, dtype)
        if not x.size or x.shape != y.shape:
            raise ValueError("Empty or mismatched snapshot: " + name)
        entry = {"elements": int(x.size), "baseline_sha256": digest(before),
                 "candidate_sha256": digest(after), "exact": bool(np.array_equal(x, y))}
        if integer:
            np.testing.assert_array_equal(y, x, err_msg=name)
        else:
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Non-finite snapshot: " + name)
            a, b = x.astype(np.float64), y.astype(np.float64)
            entry["max_abs"] = float(np.max(np.abs(a - b)))
            entry["relative_l2"] = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
            # Same gate as the independently injected hybrid CPU reference.
            np.testing.assert_allclose(y, x, atol=1e-4, rtol=1e-5, err_msg=name)
        result["files"][name] = entry
    result["passed"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(args.baseline, args.candidate)
    except (OSError, ValueError, AssertionError) as error:
        report = {"passed": False, "baseline": str(args.baseline), "candidate": str(args.candidate),
                  "device_executed": False, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
