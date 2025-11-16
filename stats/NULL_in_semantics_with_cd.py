#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any, Dict


def count_semantic_ids_in_node(node: Any, counters: Dict[str, int]) -> None:
    if isinstance(node, dict):
        if "_semantic" in node:
            sem_list = node.get("_semantic") or []
            cd_map = node.get("_concept_description") or {}
            if not isinstance(cd_map, dict):
                cd_map = {}

            for uri in sem_list:
                counters["total"] += 1
                if uri not in cd_map or cd_map[uri] is None:
                    counters["null"] += 1

        for v in node.values():
            count_semantic_ids_in_node(v, counters)

    elif isinstance(node, list):
        for item in node:
            count_semantic_ids_in_node(item, counters)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <path-to-summary-json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    submodels = data.get("submodels", {})

    print("\n=== SEMANTIC ID COVERAGE REPORT ===\n")

    for name, sm in submodels.items():
        if not isinstance(sm, dict):
            continue

        counters = {"total": 0, "null": 0}
        count_semantic_ids_in_node(sm, counters)

        filled = counters["total"] - counters["null"]

        print(f"Submodel: {name}")
        print(f"  Total semanticIds:            {counters['total']}")
        print(f"  With concept description:     {filled}")
        print(f"  Missing / null description:   {counters['null']}")
        print("-" * 50)

    print()
    

if __name__ == "__main__":
    main()
