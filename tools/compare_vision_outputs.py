#!/usr/bin/env python3
"""Compare Qwen3Vision Python prints against omni_x86.log dumps.

Usage:
  python tools/compare_vision_outputs.py --py-log python_dump.txt --omni-log omni_x86.log
"""

import argparse
import ast
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


def _parse_shape(line: str) -> Optional[Tuple[int, ...]]:
    match = re.search(r"shape=\(([^)]*)\)", line)
    if not match:
        return None
    raw = match.group(1).strip()
    if not raw:
        return tuple()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    dims = []
    for part in parts:
        try:
            dims.append(int(part))
        except ValueError:
            return None
    return tuple(dims)


def _extract_list_literal(text: str) -> Optional[str]:
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _parse_torch_repr(lines: List[str]) -> Optional[np.ndarray]:
    text = "\n".join(lines).strip()
    if not text:
        return None
    if "..." in text:
        return None
    list_literal = _extract_list_literal(text)
    if list_literal is None:
        return None
    try:
        data = ast.literal_eval(list_literal)
    except Exception:
        return None
    return np.array(data)


def parse_python_log(path: str) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    current_var = None
    while i < len(lines):
        line = lines[i].rstrip("\n")
        match = re.match(r"\[Qwen3Vision\.forward\] (\w+):", line)
        if match:
            current_var = match.group(1)
            i += 1
            shape = None
            if i < len(lines) and "shape=" in lines[i]:
                shape = _parse_shape(lines[i])
                i += 1
            repr_lines: List[str] = []
            while i < len(lines):
                if re.match(r"\[Qwen3Vision\.forward\] (\w+):", lines[i]):
                    break
                repr_lines.append(lines[i])
                i += 1
            arr = _parse_torch_repr(repr_lines)
            if arr is not None and shape is not None:
                if arr.size == int(np.prod(shape)):
                    arr = arr.reshape(shape)
            if arr is not None:
                data[current_var] = arr
            continue
        i += 1
    return data


def parse_omni_log(path: str) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = re.match(r"(\w+) shape=\[([^\]]*)\] type=\w+ size=(\d+)", line)
        if not match:
            i += 1
            continue
        name = match.group(1)
        shape = [int(x.strip()) for x in match.group(2).split(",") if x.strip()]
        size = int(match.group(3))
        values: List[float] = []
        if i + 1 < len(lines) and f"{name} values:" in lines[i + 1]:
            i += 2
            while i < len(lines):
                cur = lines[i].strip()
                if not cur:
                    i += 1
                    continue
                if re.match(r"\w+ shape=\[", cur):
                    break
                if re.match(r"\w+ values:", cur):
                    break
                for token in cur.split():
                    try:
                        values.append(float(token))
                    except ValueError:
                        pass
                i += 1
        else:
            i += 1
        if values and len(values) == size:
            arr = np.array(values, dtype=np.float32).reshape(shape)
            data[name] = arr
    return data


def compare_arrays(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> Tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    if a.dtype.kind in "iu" and b.dtype.kind in "iu":
        ok = np.array_equal(a, b)
        return ok, "exact match" if ok else "value mismatch"
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    if ok:
        return True, "allclose"
    diff = np.max(np.abs(a - b))
    return False, f"max_abs_diff={diff}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--py-log", required=True)
    parser.add_argument("--omni-log", required=True)
    parser.add_argument("--vars", default="flatten_patches,position_ids,attention_mask,idx_tensor,weight_tensor")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    py_data = parse_python_log(args.py_log)
    omni_data = parse_omni_log(args.omni_log)

    names = [v.strip() for v in args.vars.split(",") if v.strip()]
    for name in names:
        py_arr = py_data.get(name)
        omni_arr = omni_data.get(name)
        if py_arr is None:
            print(f"{name}: missing in python log (or truncated)")
            continue
        if omni_arr is None:
            print(f"{name}: missing in omni log")
            continue
        ok, msg = compare_arrays(py_arr, omni_arr, args.rtol, args.atol)
        status = "OK" if ok else "DIFF"
        print(f"{name}: {status} ({msg})")


if __name__ == "__main__":
    main()
