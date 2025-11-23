import json
import sys
from typing import Any, Dict, List, Optional


# -------------------- basic helpers -------------------- #

def get_node(root: Dict[str, Any], path: str) -> Any:
    """
    Follow a JSON path like 'submodels/0/submodelElements/0/value/3/value/0/value/0'
    and return the referenced node.
    """
    node = root
    for part in path.strip("/").split("/"):
        node = node[int(part)] if part.isdigit() else node[part]
    return node


def extract_descriptions(node: Dict[str, Any]) -> Dict[str, str]:
    """
    Collect language→text from 'description' arrays.
    """
    descs: Dict[str, str] = {}
    for d in node.get("description", []):
        lang = d.get("language")
        text = d.get("text")
        if lang and text:
            descs[lang] = text
    return descs


# -------------------- reference / semantic helpers -------------------- #

def extract_ref_uris(ref_obj: Any) -> List[str]:
    """
    Return all key.value URIs from a Reference-like object, or [] if none.
    Works for semanticId, semanticIdListElement, supplementalSemanticIds[*], etc.
    """
    if not isinstance(ref_obj, dict):
        return []
    uris: List[str] = []
    for k in ref_obj.get("keys") or []:
        v = k.get("value")
        if v:
            uris.append(v)
    return uris


def extract_semantic_summary(node: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Summarize all semantic references for an element into:
      - primary: semanticId URIs
      - listElement: semanticIdListElement URIs
      - supplemental: all URIs from supplementalSemanticIds
    """
    primary = extract_ref_uris(node.get("semanticId"))
    list_elem = extract_ref_uris(node.get("semanticIdListElement"))

    supplemental: List[str] = []
    for sup in node.get("supplementalSemanticIds", []):
        supplemental.extend(extract_ref_uris(sup))

    return {
        "primary": primary,
        "listElement": list_elem,
        "supplemental": supplemental,
    }


def extract_qualifiers(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract qualifiers, including their own semantic URIs (qualifier.semanticId).
    """
    qs: List[Dict[str, Any]] = []
    for q in node.get("qualifiers", []):
        qs.append({
            "type": q.get("type"),
            "kind": q.get("kind"),
            "valueType": q.get("valueType"),
            "value": q.get("value"),
            "semantic_uris": extract_ref_uris(q.get("semanticId")),
        })
    return qs


# -------------------- ConceptDescription helpers -------------------- #

def build_cd_index(root: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build an index: ConceptDescription.id → ConceptDescription object.
    """
    idx: Dict[str, Dict[str, Any]] = {}
    for cd in root.get("conceptDescriptions", []):
        cid = cd.get("id")
        if cid:
            idx[cid] = cd
    return idx


def summarize_cd(cd: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Compact summary of a ConceptDescription.
    """
    if cd is None:
        return None
    return {
        "id": cd.get("id"),
        "idShort": cd.get("idShort"),
        "description": extract_descriptions(cd),
    }


# -------------------- parents + siblings with semantics/CDs -------------------- #

def get_parents_with_cd(
    root: Dict[str, Any],
    full_path: str,
    cd_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return all parent nodes from root down to the parent of the target.
    For each dict parent, attach its semantic summary and matching ConceptDescription (from primary[0]).
    """
    parts = full_path.strip("/").split("/")
    parents: List[Dict[str, Any]] = []

    for i in range(len(parts) - 1):
        prefix = "/".join(parts[:i + 1])
        node = get_node(root, prefix)
        if not isinstance(node, dict):
            continue

        sem = extract_semantic_summary(node)
        primary = sem["primary"][0] if sem["primary"] else None
        cd = summarize_cd(cd_index.get(primary))

        parents.append({
            "path": prefix,
            "idShort": node.get("idShort"),
            "modelType": node.get("modelType"),
            "semantic": sem,
            "conceptDescription": cd,
        })
    return parents


def get_siblings_with_cd(
    root: Dict[str, Any],
    full_path: str,
    cd_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return siblings of the target node (same parent list), each with semantic summary and matching CD.
    """
    parts = full_path.strip("/").split("/")
    if not parts:
        return []
    leaf_idx_str = parts[-1]
    parent_path = "/".join(parts[:-1])
    parent = get_node(root, parent_path)

    siblings: List[Dict[str, Any]] = []
    if not isinstance(parent, dict):
        return siblings

    for key in ("value", "statements"):
        lst = parent.get(key)
        if not isinstance(lst, list):
            continue
        for idx, child in enumerate(lst):
            if str(idx) == leaf_idx_str:
                continue
            if not isinstance(child, dict):
                continue

            sem = extract_semantic_summary(child)
            primary = sem["primary"][0] if sem["primary"] else None
            cd = summarize_cd(cd_index.get(primary))

            siblings.append({
                "index": idx,
                "idShort": child.get("idShort"),
                "modelType": child.get("modelType"),
                "semantic": sem,
                "conceptDescription": cd,
            })
    return siblings


# -------------------- context object -------------------- #

def build_context_object(root: Dict[str, Any], full_path: str) -> Dict[str, Any]:
    """
    Build full context object for a given element path:
      - submodel
      - parent_chain
      - target (with semantic + qualifiers + CD)
      - siblings (with semantic + CD)
      - path
    """
    cd_index = build_cd_index(root)
    target = get_node(root, full_path)
    if not isinstance(target, dict):
        raise TypeError(f"Target at path '{full_path}' is not an object")

    target_sem = extract_semantic_summary(target)
    target_primary = target_sem["primary"][0] if target_sem["primary"] else None
    target_cd = summarize_cd(cd_index.get(target_primary))

    parents = get_parents_with_cd(root, full_path, cd_index)
    siblings = get_siblings_with_cd(root, full_path, cd_index)

    # pick submodel (first parent with modelType == "Submodel")
    submodel_block = None
    for p in parents:
        if p.get("modelType") == "Submodel":
            sem = p.get("semantic") or {"primary": [], "listElement": [], "supplemental": []}
            primary = sem["primary"][0] if sem["primary"] else None
            sub_cd = summarize_cd(cd_index.get(primary))
            submodel_block = {
                "idShort": p.get("idShort"),
                "semantic": sem,
                "conceptDescription": sub_cd,
            }
            break

    # parent_chain = all parents except submodel
    parent_chain: List[Dict[str, Any]] = []
    for p in parents:
        if p.get("modelType") == "Submodel":
            continue
        parent_chain.append({
            "idShort": p.get("idShort"),
            "modelType": p.get("modelType"),
            "semantic": p.get("semantic"),
            "conceptDescription": p.get("conceptDescription"),
            "path": p.get("path"),
        })

    ctx = {
        "submodel": submodel_block,
        "parent_chain": parent_chain,
        "target": {
            "idShort": target.get("idShort"),
            "modelType": target.get("modelType"),
            "valueType": target.get("valueType"),
            "description": extract_descriptions(target),
            "semantic": target_sem,
            "qualifiers": extract_qualifiers(target),
            "conceptDescription": target_cd,
        },
        "siblings": siblings,
        "path": full_path,
    }
    return ctx


# -------------------- main -------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python build_context.py <aas_json_path> <path_or_relpath_inside_first_submodel>",
            file=sys.stderr,
        )
        sys.exit(1)

    json_path = sys.argv[1]
    user_path = sys.argv[2].strip().lstrip("/")

    # if user didn't include 'submodels/' prefix, assume first submodel
    if not user_path.startswith("submodels/"):
        full_path = "submodels/0/" + user_path
    else:
        full_path = user_path

    with open(json_path, "r", encoding="utf-8") as f:
        aas_json = json.load(f)

    context = build_context_object(aas_json, full_path)
    print(json.dumps(context, indent=2, ensure_ascii=False))
