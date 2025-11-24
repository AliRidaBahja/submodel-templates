#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------- repo traversal ----------

def pick_json_in_dir(d: Path) -> Optional[Path]:
    js = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    if not js:
        return None
    templ = [p for p in js if "template" in p.name.lower()]
    return templ[0] if templ else js[0]


def list_numeric_children(d: Path) -> List[Path]:
    return [p for p in d.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)]


def descend_versions_and_pick_json(root: Path) -> List[Path]:
    picked: List[Path] = []

    for cur, dirs, _ in os.walk(root):
        curp = Path(cur)

        # ---------------------------------------------------------
        # NEW LOGIC: do NOT check for JSONs at the root directory
        # ---------------------------------------------------------
        if curp != root:
            chosen = pick_json_in_dir(curp)
            if chosen:
                picked.append(chosen)
                dirs[:] = []       # stop descending from this subtree
                continue

        # numeric version logic stays untouched
        numeric = list_numeric_children(curp)
        if numeric:
            newest = max(numeric, key=lambda p: int(p.name))
            dirs[:] = [d for d in dirs if (curp / d).resolve() == newest.resolve()]

    return picked


# ---------- AAS helpers ----------

def _extract_ref_value(ref: Any) -> Optional[str]:
    """Extract first key.value from an ExternalReference-like dict."""
    if not isinstance(ref, dict):
        return None
    keys = ref.get("keys", [])
    if isinstance(keys, list) and keys:
        k0 = keys[0]
        if isinstance(k0, dict):
            v = k0.get("value")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _extract_all_ref_values(ref: Any) -> List[str]:
    """Extract ALL key.value strings from an ExternalReference-like dict."""
    values: List[str] = []
    if not isinstance(ref, dict):
        return values
    keys = ref.get("keys", [])
    if not isinstance(keys, list):
        return values
    for k in keys:
        if not isinstance(k, dict):
            continue
        v = k.get("value")
        if isinstance(v, str) and v.strip():
            values.append(v.strip())
    return values


def collect_semantic_ids(
    node: Any,
    out: List[Tuple[str, str, str]],
    path: str = ""
) -> None:
    """
    Collect ALL semanticId, semanticIdListElement, and supplementalSemanticIds
    occurrences in the tree.

    For nodes without idShort, use "<no-idShort>" for the primary semanticId.
    For semanticIdListElement, suffix owner with " [LIST-ELEMENT]".
    For supplementalSemanticIds, use owner '|_>' (same entity as the semanticId row above).
    Also store the absolute JSON path so that get_by_path(root, path) can locate the node.
    """
    if isinstance(node, dict):
        base_idshort = node.get("idShort") if isinstance(node.get("idShort"), str) else "<no-idShort>"

        # semanticId of this node
        sid = node.get("semanticId")
        if isinstance(sid, dict):
            val = _extract_ref_value(sid)
            if val is not None:
                out.append((base_idshort, val, path))

        # semanticIdListElement of this node (SubmodelElementList)
        sid_le = node.get("semanticIdListElement")
        if isinstance(sid_le, dict):
            val_le = _extract_ref_value(sid_le)
            if val_le is not None:
                owner = f"{base_idshort} [LIST-ELEMENT]"
                out.append((owner, val_le, path))

        # supplementalSemanticIds (same entity as base_idshort, printed with arrow owner)
        supps = node.get("supplementalSemanticIds")
        if isinstance(supps, list):
            for ref in supps:
                for v in _extract_all_ref_values(ref):
                    out.append(("|_>", v, path))

        # recurse into dict children, tracking JSON path
        for k, v in node.items():
            subpath = f"{path}/{k}" if path else k
            collect_semantic_ids(v, out, subpath)

    elif isinstance(node, list):
        # recurse into list children, tracking index in path
        for idx, item in enumerate(node):
            subpath = f"{path}/{idx}" if path else str(idx)
            collect_semantic_ids(item, out, subpath)


def collect_concept_descriptions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collect all ConceptDescriptions as a LIST (do NOT collapse by id).

    Returns a list of dicts:
        {
          "id": str | None,
          "idShort": str | None,
          "isCaseOf": [iri_or_irdi, ...]
        }
    """
    cds_list: List[Dict[str, Any]] = []
    arr = data.get("conceptDescriptions", [])
    if not isinstance(arr, list):
        return cds_list

    for cd in arr:
        if not isinstance(cd, dict):
            continue
        if cd.get("modelType") != "ConceptDescription":
            continue

        cid = cd.get("id")
        if isinstance(cid, str):
            cid = cid.strip()
        else:
            cid = None

        cid_short = cd.get("idShort") if isinstance(cd.get("idShort"), str) else None

        # extract isCaseOf entries
        is_case_vals: List[str] = []
        iscase = cd.get("isCaseOf", [])
        if isinstance(iscase, list):
            for ref in iscase:
                if not isinstance(ref, dict):
                    continue
                keys = ref.get("keys", [])
                if isinstance(keys, list):
                    for k in keys:
                        if not isinstance(k, dict):
                            continue
                        v = k.get("value")
                        if isinstance(v, str):
                            is_case_vals.append(v.strip())

        cds_list.append({
            "id": cid,
            "idShort": cid_short,
            "isCaseOf": is_case_vals,
        })

    return cds_list


def match_concepts(semantic_id: str, cds_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return ALL ConceptDescriptions whose id == semantic_id."""
    return [cd for cd in cds_list if cd.get("id") == semantic_id]


def pretty_print_submodel(
    file_path: Path,
    submodel_name: str,
    sem_rows: List[Tuple[int, int, str, str, Optional[str], List[str], bool, str]],
    cd_rows: List[Tuple[int, int, Optional[str], str, List[str], bool]],
) -> None:
    """
    sem_rows:
      (global_sem_idx, local_sem_idx, owner_idShort, semanticId,
       cd_idShort_or_None, cd_isCaseOf_list, missing_flag, abs_path)

    cd_rows:
      (global_cd_idx, local_cd_idx, cd_idShort_or_None, cd_id,
       cd_isCaseOf_list, is_unlinked_flag)
    """

    # counts
    missing_sem = sum(1 for r in sem_rows if r[6])
    unlinked_cd = sum(1 for r in cd_rows if r[5])

    # ---------- semanticId table ----------
    if sem_rows:
        width_g = max(len(str(o[0])) for o in sem_rows)
        width_l = max(len(str(o[1])) for o in sem_rows)
        width_owner = max(len(o[2]) + (4 if o[6] else 0) for o in sem_rows)  # account for " (M)"
        width_sid = max(len(o[3]) for o in sem_rows)
        width_cd = max(len(o[4]) if o[4] else len("NO_CONCEPT") for o in sem_rows)
        width_is = max(len(", ".join(o[5])) if o[5] else len("-") for o in sem_rows)
        width_path = max(len(o[7]) for o in sem_rows)

        print()
        print("=" * 140)
        print(f"FILE: {file_path}")
        print(f"SUBMODEL: {submodel_name}")
        print("=" * 140)

        header = (
            f"{'G#'.rjust(width_g)}  "
            f"{'L#'.rjust(width_l)}  "
            f"{'idShort'.ljust(width_owner)}   "
            f"{'semanticId'.ljust(width_sid)}   "
            f"{'conceptIdShort'.ljust(width_cd)}   "
            f"{'cd_isCaseOf'.ljust(width_is)}   "
            f"{'absPath'.ljust(width_path)}"
        )
        print(header)
        print("-" * 140)

        for g_idx, l_idx, owner, sid, cd_short, iscase, missing, abs_path in sem_rows:
            mark = " (M)" if missing else ""
            owner_marked = f"{owner}{mark}"
            cd = cd_short if cd_short else "NO_CONCEPT"
            is_str = ", ".join(iscase) if iscase else "-"

            print(
                f"{str(g_idx).rjust(width_g)}  "
                f"{str(l_idx).rjust(width_l)}  "
                f"{owner_marked.ljust(width_owner)}   "
                f"{sid.ljust(width_sid)}   "
                f"{cd.ljust(width_cd)}   "
                f"{is_str.ljust(width_is)}   "
                f"{abs_path.ljust(width_path)}"
            )

    # ---------- ConceptDescription table ----------
    if cd_rows:
        width_g2 = max(len(str(o[0])) for o in cd_rows)
        width_l2 = max(len(str(o[1])) for o in cd_rows)
        width_cshort = max(
            len((o[2] if o[2] else "<no-idShort>") + (" (U)" if o[5] else ""))
            for o in cd_rows
        )
        width_cid = max(len(o[3]) for o in cd_rows)
        width_is2 = max(len(", ".join(o[4])) if o[4] else len("-") for o in cd_rows)

        print()
        print("CONCEPT DESCRIPTIONS (U = unlinked w.r.t this submodel)")
        print("-" * 140)

        header2 = (
            f"{'G#'.rjust(width_g2)}  "
            f"{'L#'.rjust(width_l2)}  "
            f"{'cd_idShort'.ljust(width_cshort)}   "
            f"{'cd_id'.ljust(width_cid)}   "
            f"{'cd_isCaseOf'.ljust(width_is2)}"
        )
        print(header2)
        print("-" * 140)

        for g_idx, l_idx, cd_short, cid, iscase, is_unlinked in cd_rows:
            base_short = cd_short if cd_short else "<no-idShort>"
            mark = " (U)" if is_unlinked else ""
            short_str = base_short + mark
            is_str = ", ".join(iscase) if iscase else "-"

            print(
                f"{str(g_idx).rjust(width_g2)}  "
                f"{str(l_idx).rjust(width_l2)}  "
                f"{short_str.ljust(width_cshort)}   "
                f"{cid.ljust(width_cid)}   "
                f"{is_str.ljust(width_is2)}"
            )

    # ---------- summary ----------
    print()
    print(f"SUMMARY: missing semanticIds (M): {missing_sem}, unlinked ConceptDescriptions (U): {unlinked_cd}")
    print("=" * 140)


def get_target_si(path: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
    from pathlib import Path
    import json, sys
    
    # ---------------------------------------------
    # Resolve which JSON files to process
    # ---------------------------------------------
    if path is not None:
        jf = Path(path).resolve()
        if (not jf.exists()) or jf.suffix.lower() != ".json":
            return {}
        json_files = [jf]

    else:
        # mimic main(): traverse repository
        root = Path(__file__).resolve().parent
        json_files = descend_versions_and_pick_json(root)
        if not json_files:
            return {}

    # ---------------------------------------------
    # Process files
    # ---------------------------------------------
    missing_summary: Dict[str, List[Dict[str, str]]] = {}

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        concepts = collect_concept_descriptions(data)
        submodels = data.get("submodels", [])
        if not isinstance(submodels, list):
            continue

        file_missing: List[Dict[str, str]] = []

        for sm in submodels:
            if not isinstance(sm, dict):
                continue

            semantic_pairs: List[Tuple[str, str, str]] = []
            collect_semantic_ids(sm, semantic_pairs)

            # generate semantic rows (local only)
            sem_rows = []
            global_sem_counter = 1
            local_sem_idx = 1

            for owner, sid, abs_path in semantic_pairs:
                matches = match_concepts(sid, concepts)

                if matches:
                    for cd in matches:
                        sem_rows.append(
                            (
                                global_sem_counter,
                                local_sem_idx,
                                owner,
                                sid,
                                cd.get("idShort"),
                                cd.get("isCaseOf") or [],
                                False,
                                abs_path,
                            )
                        )
                        global_sem_counter += 1
                        local_sem_idx += 1
                else:
                    sem_rows.append(
                        (
                            global_sem_counter,
                            local_sem_idx,
                            owner,
                            sid,
                            None,
                            [],
                            True,       # missing
                            abs_path,
                        )
                    )
                    global_sem_counter += 1
                    local_sem_idx += 1

            # collect missing
            for _, _, owner, sid, _, _, missing, abs_path in sem_rows:
                if missing:
                    file_missing.append({
                        "idShort": owner,
                        "semanticId": sid,
                        "absPath": abs_path,
                    })

        if file_missing:
            missing_summary[str(jf)] = file_missing

    return missing_summary


if __name__ == "__main__":
    import sys
    from pathlib import Path

    args = sys.argv[1:]

    # determine json files
    if len(args) == 1:
        jf = Path(args[0]).resolve()
        if not jf.exists() or jf.suffix.lower() != ".json":
            sys.exit()
        json_files = [jf]

    elif len(args) == 0:
        root = Path.cwd()
        json_files = descend_versions_and_pick_json(root)
        if not json_files:
            sys.exit()

    else:
        sys.exit()

    missing_summary: Dict[str, List[Dict[str, str]]] = {}
    global_sem_counter = 1
    global_cd_counter = 1

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        concepts = collect_concept_descriptions(data)
        submodels = data.get("submodels", [])
        if not isinstance(submodels, list):
            continue

        file_missing: List[Dict[str, str]] = []

        for sm in submodels:
            if not isinstance(sm, dict):
                continue

            semantic_pairs: List[Tuple[str, str, str]] = []
            collect_semantic_ids(sm, semantic_pairs)

            sem_rows = []
            local_sem_idx = 1

            for owner, sid, abs_path in semantic_pairs:
                matches = match_concepts(sid, concepts)

                if matches:
                    for cd in matches:
                        sem_rows.append(
                            (
                                global_sem_counter,
                                local_sem_idx,
                                owner,
                                sid,
                                cd.get("idShort"),
                                cd.get("isCaseOf") or [],
                                False,
                                abs_path,
                            )
                        )
                        global_sem_counter += 1
                        local_sem_idx += 1
                else:
                    sem_rows.append(
                        (
                            global_sem_counter,
                            local_sem_idx,
                            owner,
                            sid,
                            None,
                            [],
                            True,
                            abs_path,
                        )
                    )
                    global_sem_counter += 1
                    local_sem_idx += 1

            for _, _, owner, sid, _, _, missing, abs_path in sem_rows:
                if missing:
                    file_missing.append(
                        {
                            "idShort": owner,
                            "semanticId": sid,
                            "absPath": abs_path,
                        }
                    )

        if file_missing:
            missing_summary[str(jf)] = file_missing

    if missing_summary:
        Path("missing_semantic_ids.json").write_text(
            json.dumps(missing_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
