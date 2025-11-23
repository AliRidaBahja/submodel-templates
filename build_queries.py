import json
import re
import sys
from typing import Any, Dict, List

STOPWORDS = {
    "the", "of", "for", "and", "a", "an", "to", "in", "is", "are", "which",
    "this", "that", "with", "as", "by", "on", "or", "be", "it",
    "describing",
}

# tokens that are domain-weak in almost any technical context
WEAK_DOMAIN_TOKENS = {
    "indicator", "asset", "property", "value",
    "name", "condition", "element", "data", "information",
}

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 2]
    return tokens


def humanize_idshort(s: str) -> str:
    if not s:
        return ""
    s = s.replace("_", " ").replace("-", " ")
    # split only between lower/digit and upper, so "RUL" stays "RUL"
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    return " ".join(s.split())

def compress(text: str) -> str:
    return " ".join((text or "").split())


def build_queries(context: Dict[str, Any]) -> List[str]:
    """
    Build natural-language and keyword queries from a context object produced by build_context.py.
    """
    t = context["target"]
    parents = context.get("parent_chain", [])

    # ---------- target labels / descriptions ----------
    raw_idshort = new_func(t)
    target_label_human = humanize_idshort(raw_idshort)

    cd = t.get("conceptDescription") or {}
    cd_desc = cd.get("description") or {}
    cd_desc_en = (cd_desc.get("en") or "").strip()

    t_desc = t.get("description") or {}
    t_desc_en = (t_desc.get("en") or "").strip()

    cd_idshort = (cd.get("idShort") or "").strip()
    if cd_idshort:
        target_label = humanize_idshort(cd_idshort)
    else:
        target_label = target_label_human

    # ---------- parent chain → domain phrase ----------
    queries: List[str] = []

    # --- collect parent labels ---
    parent_labels_nl: List[str] = []
    for p in parents:
        ps_raw = (p.get("idShort") or "").strip()
        if ps_raw:
            parent_labels_nl.append(humanize_idshort(ps_raw))

    # --- base text: target label + definition/description ---
    base_text_parts: List[str] = []
    if target_label:
        base_text_parts.append(target_label)
    if cd_desc_en:
        base_text_parts.append(cd_desc_en)
    elif t_desc_en:
        base_text_parts.append(t_desc_en)
    base_text = " ".join(base_text_parts)

    # --- parent text ---
    parent_text = " ".join(parent_labels_nl)

    # --- tokens ---
    base_tokens = tokenize(base_text)
    parent_tokens = tokenize(parent_text)

    # de-duplicate base tokens (preserve order)
    seen = set()
    base_unique: List[str] = []
    for t in base_tokens:
        if t not in seen:
            seen.add(t)
            base_unique.append(t)

    parent_unique: List[str] = []
    for t in parent_tokens:
        if t not in seen:
            seen.add(t)
            parent_unique.append(t)

    # separate strong vs weak tokens
    strong = [t for t in base_unique if t not in WEAK_DOMAIN_TOKENS]
    weak = [t for t in base_unique if t in WEAK_DOMAIN_TOKENS]

    # Q1: only strong tokens, short
    MAX_TOKENS = 6
    q1_tokens = strong[:MAX_TOKENS]
    if not q1_tokens:
        # fallback: use weak if strong list is empty
        q1_tokens = base_unique[:MAX_TOKENS]

    if q1_tokens:
        q1 = " ".join(q1_tokens)
        queries.append(q1)

    # Q2: strong + parent tokens
    context_tokens: List[str] = []
    for t in parent_unique:
        if t not in context_tokens and t not in WEAK_DOMAIN_TOKENS:
            context_tokens.append(t)

    if context_tokens:
        q2_tokens: List[str] = []
        for t in strong + context_tokens:
            if t not in q2_tokens:
                q2_tokens.append(t)
            if len(q2_tokens) >= MAX_TOKENS:
                break
        if not q2_tokens:
            q2_tokens = (base_unique + parent_unique)[:MAX_TOKENS]

        q2 = " ".join(q2_tokens)
        if q2 and q2 not in queries:
            queries.append(q2)

    return queries



def new_func(t):
    raw_idshort = (t.get("idShort") or "").strip()
    return raw_idshort


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_queries.py <context_json_path>", file=sys.stderr)
        sys.exit(1)

    ctx_path = sys.argv[1]

    def load_context(path: str) -> Dict[str, Any]:
        """Load JSON allowing UTF-8/UTF-16 BOM inputs."""
        with open(path, "rb") as f:
            data = f.read()

        for enc in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
            try:
                return json.loads(data.decode(enc))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

        raise UnicodeDecodeError(
            "utf-8/utf-16 variants", b"", 0, 0, "Unable to decode input file"
        )

    context = load_context(ctx_path)

    queries = build_queries(context)

    for i, q in enumerate(queries, start=1):
        print(f"{i}. {q}")

