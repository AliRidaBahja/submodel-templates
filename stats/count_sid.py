#!/usr/bin/env python3
"""
Count semanticId prefixes with URL host-only namespaces.

Rules:
- URL IDs: collapse http/https and count by host only (e.g., "admin-shell.io").
- IRDI-like: leading r'^(\d{4}-\d+)'  → e.g., "0173-1" from "0173-1#02-AAO129#002".
- IEC CDD-like: leading r'^(\d{4}/\d+)' → e.g., "0112/2" from "0112/2///61360_4#...".
- Suppress 'OTHER' (anything not matching the above).

Input: JSON array with objects containing "semanticId".
Output: text table (default) or JSON (--json / -o).
"""

import argparse, json, sys, re
from urllib.parse import urlparse
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

re_irdi_dash = re.compile(r'^(\d{4}-\d+)')
re_iec_cdd  = re.compile(r'^(\d{4}/\d+)')

def extract_prefix(identifier: str) -> Tuple[str, str]:
    """
    Returns (kind, prefix). kind in {"URL","IRDI","IEC","OTHER"}.
    URL prefix is host only (scheme-collapsed), e.g., "admin-shell.io".
    """
    if not isinstance(identifier, str):
        return ("OTHER", "")
    s = identifier.strip()
    if not s:
        return ("OTHER", "")

    m = re_irdi_dash.match(s)
    if m:
        return ("IRDI", m.group(1))

    m = re_iec_cdd.match(s)
    if m:
        return ("IEC", m.group(1))

    pr = urlparse(s)
    if pr.netloc:  # any scheme with netloc; http/https collapsed by using host only
        return ("URL", pr.netloc.lower())

    return ("OTHER", "")

def load_items(path: str) -> List[Dict[str, Any]]:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def count_prefixes(items: List[Dict[str, Any]], include_other: bool) -> Tuple[Counter, Dict[str, Counter]]:
    total = Counter()
    by_kind: Dict[str, Counter] = defaultdict(Counter)
    for it in items:
        sid = it.get("semanticId", "")
        kind, pref = extract_prefix(sid)
        if kind == "OTHER" and not include_other:
            continue
        if not pref:
            continue
        total[pref] += 1
        by_kind[kind][pref] += 1
    return total, by_kind

def print_table(total: Counter, by_kind: Dict[str, Counter]) -> None:
    width = max((len(k) for k in total), default=6)
    print(f"{'PREFIX'.ljust(width)}  COUNT")
    for k, v in sorted(total.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{k.ljust(width)}  {v}")

    if by_kind.get("URL"):
        print("\n# URL namespaces (host-only)")
        uw = max((len(k) for k in by_kind["URL"]), default=6)
        for k, v in sorted(by_kind["URL"].items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{k.ljust(uw)}  {v}")

def to_json(total: Counter, by_kind: Dict[str, Counter]) -> Dict[str, Any]:
    return {
        "total": dict(sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_kind": {k: dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
                    for k, c in sorted(by_kind.items())}
    }

def parse_args():
    ap = argparse.ArgumentParser(description="Count semanticId prefixes with URL host-only namespaces.")
    ap.add_argument("input", help="Path to JSON array (use '-' for stdin).")
    ap.add_argument("--include-other", action="store_true", help="Include 'OTHER' category.")
    ap.add_argument("--json", action="store_true", help="Print JSON to stdout instead of a text table.")
    ap.add_argument("-o", "--output", help="Write JSON to this file.")
    return ap.parse_args()

def main():
    args = parse_args()
    items = load_items(args.input)
    total, by_kind = count_prefixes(items, include_other=args.include_other)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(to_json(total, by_kind), f, indent=2, ensure_ascii=False)
        return

    if args.json:
        print(json.dumps(to_json(total, by_kind), indent=2, ensure_ascii=False))
    else:
        print_table(total, by_kind)

if __name__ == "__main__":
    main()
