#!/usr/bin/env python3
"""Report code size of emlearn modules in a MicroPython build.

Scans the build directory for emlearn .o files and reports per-module
section sizes (code, rodata, data, bss).

Usage:
    python3 code_size.py <build_dir>

Examples:
    python3 code_size.py build-standard
    python3 tools/code_size.py micropython/ports/unix/build-standard
    make codesize
"""

import os
import re
import subprocess
import sys
from collections import defaultdict


def get_obj_sections(obj_path):
    """Return {section: size} for an object file, or empty dict on failure."""
    try:
        r = subprocess.run(
            ["objdump", "-h", obj_path],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    sections = defaultdict(int)
    for line in r.stdout.splitlines():
        m = re.match(r"\s+\d+\s+(\S+)\s+([0-9a-f]+)\s", line)
        if not m:
            continue
        name, size = m.group(1), int(m.group(2), 16)
        if size == 0:
            continue
        if name.startswith(".text"):
            sections["text"] += size
        elif name.startswith(".rodata"):
            sections["rodata"] += size
        elif name.startswith(".data.rel.ro"):
            sections["data.rel.ro"] += size
        elif name.startswith(".data"):
            sections["data"] += size
        elif name.startswith(".bss"):
            sections["bss"] += size
    return dict(sections)


def scan_emlearn(build_dir):
    """Find emlearn .o files and return {module_name: {section: size}}."""
    results = {}
    for root, _dirs, files in os.walk(build_dir):
        for f in files:
            if not f.endswith(".o") or not f.startswith(".") and "emlearn" not in root:
                continue
            rel = os.path.relpath(root, build_dir)
            if not rel.startswith("emlearn_"):
                continue
            # e.g. "emlearn_logreg" -> "logreg"
            name = rel.replace("emlearn_", "")
            path = os.path.join(root, f)
            sections = get_obj_sections(path)
            if sections:
                results[name] = sections
    return results


COLS = ["text", "rodata", "data.rel.ro", "data", "bss"]


def print_report(modules):
    """Print emlearn size report sorted by code size."""
    ranked = sorted(
        modules.items(),
        key=lambda x: x[1].get("text", 0),
        reverse=True,
    )
    if not ranked:
        return

    name_w = max(len("module"), *(len(n) for n in modules))
    col_w = 9

    hdr = f"{'module':<{name_w}}"
    for c in COLS:
        hdr += f"  {c:>{col_w}}"
    hdr += f"  {'total':>{col_w}}"
    print(hdr)
    print("-" * len(hdr))

    totals = defaultdict(int)
    for name, sections in ranked:
        total = 0
        row = f"{name:<{name_w}}"
        for c in COLS:
            v = sections.get(c, 0)
            totals[c] += v
            total += v
            row += f"  {v:>{col_w},}"
        row += f"  {total:>{col_w},}"
        print(row)

    print("-" * len(hdr))
    total_all = sum(totals.values())
    row = f"{'total':<{name_w}}"
    for c in COLS:
        row += f"  {totals[c]:>{col_w},}"
    row += f"  {total_all:>{col_w},}"
    print(row)


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    build_dir = sys.argv[1]
    if not os.path.isdir(build_dir):
        print(f"Error: '{build_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    modules = scan_emlearn(build_dir)
    if not modules:
        print(f"No emlearn object files found in '{build_dir}'", file=sys.stderr)
        sys.exit(1)

    print_report(modules)


if __name__ == "__main__":
    main()
