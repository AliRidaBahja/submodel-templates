import json

# ---------- low-level helpers ----------

def get_node(root, path):
    """
    Follow a JSON path like 'submodels/0/submodelElements/0/value/3/value/0/value/0'
    and return the referenced node.
    """
    node = root
    
    for part in path.strip("/").split("/"):
        if part.isdigit():
            node = node[int(part)]
        else:
            node = node[part]
    return node


def extract_semantic_id_value(semantic_id_obj):
    """
    AAS semanticId is an ExternalReference/ModelReference with keys.
    We take the first key.value as the 'local' ID (points to ConceptDescription.id).
    """
    if not isinstance(semantic_id_obj, dict):
        return None
    keys = semantic_id_obj.get("keys") or []
    if not keys:
        return None
    return keys[0].get("value")


def extract_descriptions(node):
    """
    Collect language→text from 'description' arrays (both for elements and conceptDescriptions).
    """
    descs = {}
    for d in node.get("description", []):
        lang = d.get("language")
        text = d.get("text")
        if lang and text:
            descs[lang] = text
    return descs


# ---------- concept description index ----------

def build_cd_index(root):
    """
    Build an index: ConceptDescription.id → ConceptDescription object.
    """
    idx = {}
    for cd in root.get("conceptDescriptions", []):
        cid = cd.get("id")
        if cid:
            idx[cid] = cd
    return idx


def summarize_concept_description(cd):
    """
    Return a compact summary of a ConceptDescription (id, idShort, descriptions).
    """
    if cd is None:
        return None
    return {
        "id": cd.get("id"),
        "idShort": cd.get("idShort"),
        "description": extract_descriptions(cd)
    }


# ---------- parents with CDs ----------

def get_parents(root, full_path, cd_index):
    """
    Return all parent nodes from root down to the parent of the target.
    For dict parents, also attach matching ConceptDescription (if any).
    """
    parts = full_path.strip("/").split("/")

    parents = []

    # exclude the leaf itself (last segment)
    for i in range(len(parts) - 1):
        prefix = "/".join(parts[:i + 1])
        node = get_node(root, prefix)

        if isinstance(node, dict):
            sid_value = extract_semantic_id_value(node.get("semanticId"))
            cd = summarize_concept_description(cd_index.get(sid_value))

            parents.append({
                "path": prefix,
                "kind": "dict",
                "idShort": node.get("idShort"),
                "modelType": node.get("modelType"),
                "semanticId_value": sid_value,
                "conceptDescription": cd
            })

        elif isinstance(node, list):
            parents.append({
                "path": prefix,
                "kind": "list",
                "idShort": None,
                "modelType": None,
                "semanticId_value": None,
                "conceptDescription": None
            })

        else:
            parents.append({
                "path": prefix,
                "kind": type(node).__name__,
                "idShort": None,
                "modelType": None,
                "semanticId_value": None,
                "conceptDescription": None
            })

    return parents


# ---------- demo: ConditionName only ----------

if __name__ == "__main__":
    json_path = r"C:\Users\aliba\Paper\submodel-templates\published\Predictive Maintenance\1\0\IDTA 02048_Template_PredictiveMaintenance.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ConditionName (M) from your TXT:
    # ... submodelElements/0/value/3/value/0/value/0
    rel_path = "submodelElements/0/value/3/value/0/value/0"
    full_path = "submodels/0/" + rel_path

    # target node
    target = get_node(data, full_path)

    # conceptDescription index
    cd_index = build_cd_index(data)

    print("TARGET NODE (raw):")
    print(json.dumps(target, indent=2, ensure_ascii=False))

    print("\nPARENTS WITH CONCEPT DESCRIPTIONS:")
    parents = get_parents(data, full_path, cd_index)
    print(json.dumps(parents, indent=2, ensure_ascii=False))
