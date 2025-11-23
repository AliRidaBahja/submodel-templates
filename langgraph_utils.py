from __future__ import annotations
import json
import sys
import requests
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv(".env")  # expects OPENAI_API_KEY / OPENAI_BASE_URL

UA = "AAS-Wikidata-Linker/0.1 (mailto:you@example.com)"


class PipelineState(TypedDict, total=False):
    context: Dict[str, Any]

    # query + history
    query: str
    query_history: List[str]

    # raw wikidata hits
    wikidata_hits: List[Dict[str, Any]]

    # evaluator decision
    eval_action: str        # select | refine | stop
    eval_reason: str
    eval_suggested_query: str

    # selected entities (if eval_action == "select")
    selected_entities: List[Dict[str, Any]]

    # loop control
    iteration: int
    max_iterations: int

    # debug / notes
    agent_notes: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


# ----------------------------
# HELPERS
# ----------------------------

def _normalize_query(q: str, max_tokens: int = 4) -> str:
    cleaned = " ".join(str(q).replace("|", " ").replace(",", " ").split())
    return " ".join(cleaned.split()[:max_tokens])


# ----------------------------
# QUERY GENERATION NODE
# ----------------------------

def node_generate_query(state: PipelineState) -> PipelineState:
    ctx = state["context"]
    prior_queries = state.get("query_history") or []
    feedback = {
        "action": state.get("eval_action"),
        "reason": state.get("eval_reason"),
        "suggested_query": state.get("eval_suggested_query"),
    }

    # If evaluator gave a suggested_query, adopt it directly (authoritative)
    suggested = feedback.get("suggested_query") or ""
    print("debug: ", suggested)
    if suggested:
        query = _normalize_query(suggested, max_tokens=4)
        history = prior_queries + [query]
        state["query"] = query
        state["query_history"] = history
        state["eval_action"] = None
        state["eval_reason"] = None
        state["eval_suggested_query"] = None
        print(f"[generate_query] iteration={state.get('iteration', 0)} (from suggested) query='{query}'")
        return state

    # Otherwise, let the LLM propose a new query from context (target + submodel)
    system_msg = SystemMessage(
        content=(
            "You output ONE Wikidata text-search query.\n"
            "- 2 to 4 English words, lowercase, spaces only. No quotes/JSON/bullets/explanations.\n"
            "- Use ONLY:\n"
            "    context['target'] (idShort, description, conceptDescription)\n"
            "    context['submodel'] (idShort, conceptDescription)\n"
            "- Ignore parents, siblings, qualifiers, paths, and raw URIs.\n"
            "- Prefer concrete domain nouns (e.g. remaining useful life, boundary, maintenance, prediction).\n"
            "- Avoid meta-words: name, id, code, value, property, indicator, data, information, description, type.\n"
            "- If nothing meaningful can be formed, return an empty string."
        )
    )
    # Full context is passed, but instructions restrict what to use
    human_msg = HumanMessage(
        content=(
            "Build a short query for this target + submodel.\n"
            f"Context JSON:\n{json.dumps(ctx, ensure_ascii=False)}\n"
            f"Previous queries: {prior_queries}\n"
            f"Evaluation feedback: {json.dumps(feedback, ensure_ascii=False)}"
        )
    )
    resp = llm.invoke([system_msg, human_msg])
    raw = (resp.content or "").strip()
    query = _normalize_query(raw, max_tokens=4)

    history = prior_queries + [query]
    state["query"] = query
    state["query_history"] = history
    state["eval_action"] = None
    state["eval_reason"] = None
    state["eval_suggested_query"] = None

    print(f"[generate_query] iteration={state.get('iteration', 0)} query='{query}'")
    return state


# ----------------------------
# WIKIDATA SEARCH NODE (wbsearchentities)
# ----------------------------

def node_search_wikidata(state: PipelineState) -> PipelineState:
    q = (state.get("query") or "").strip()
    if not q:
        print("[search_wikidata] empty query; keep prior hits and skip")
        return state

    url = "https://www.wikidata.org/w/api.php"
    headers = {"User-Agent": UA}
    params = {
        "action": "wbsearchentities",
        "search": q,
        "language": "en",
        "limit": "50",
        "format": "json",
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        hits = data.get("search", [])
        state["wikidata_hits"] = hits
        print(f"[search_wikidata] query='{q}' hits={len(hits)}")
    except Exception as e:
        state["wikidata_hits"] = [{"error": str(e)}]
        print(f"[search_wikidata] error: {e}")
    return state


# ----------------------------
# EVALUATION / SELECTION NODE
# ----------------------------

def node_agent_evaluate(state: PipelineState) -> PipelineState:
    ctx = state.get("context") or {}
    query = state.get("query") or ""
    hits = state.get("wikidata_hits") or []
    iteration = state.get("iteration", 0)
    history = state.get("query_history") or []

    system_msg = SystemMessage(
        content=(
            "You are the evaluation agent.\n"
            "You must decide whether to SELECT Wikidata entities as semantic candidates,\n"
            "REFINE the query, or STOP the search.\n\n"
            "Return STRICT JSON only:\n"
            "{\n"
            '  "action": "select" | "refine" | "stop",\n'
            '  "reason": "short explanation",\n'
            '  "suggested_query": "new short query if action=refine, else empty",\n'
            '  "candidates": [\n'
            '     { "id": "Q12345", "label": "label", "description": "desc", "score": 0.0, "rank": 1 }\n'
            "  ]\n"
            "}\n\n"
            "- Use the full AAS context to understand the target concept.\n"
            "- Hits come from Wikidata wbsearchentities: each has id, label, description.\n"
            "- You are allowed to SELECT only ontology-level concepts (properties, quantities, domain notions),\n"
            "  not scientific articles, journal articles, papers, people, or organizations.\n"
            "- If a hit's description contains terms like 'scholarly article', 'scientific article', 'journal article',\n"
            "  or 'academic paper', treat it as low priority and usually not a good candidate.\n"
            "- IMPORTANT RULES ABOUT ACTION:\n"
            "  * If hits list is EMPTY and iteration < 2, you MUST choose 'refine' (not 'stop') and propose a new 2–4 word query.\n"
            "  * You may only choose 'stop' if you have already seen at least 2 different queries in the query history,\n"
            "    and further refinement is very unlikely to find a good match.\n"
            "  * If there are hits but none are in the right domain, prefer 'refine' with a better query.\n"
            "  * SELECT only if there is at least one strong, non-article candidate; then fill 'candidates'.\n"
            "- score: 0.0–1.0, higher = better; for SELECT, scores should be >= 0.5.\n"
            "- rank: 1, 2, 3... after sorting candidates by score.\n"
        )
    )

    human_msg = HumanMessage(
        content=(
            "Evaluate the current query and hits.\n"
            "If you select, fill 'candidates' with top 3–5 good ones.\n"
            "If you refine, fill 'suggested_query' with a new short query.\n"
            "If you stop, leave suggested_query empty.\n\n"
            f"Iteration: {iteration}\n"
            f"Query history: {history}\n\n"
            f"Target context:\n{json.dumps(ctx, ensure_ascii=False)}\n\n"
            f"Current query: {query}\n\n"
            f"Wikidata hits:\n{json.dumps(hits, ensure_ascii=False)}"
        )
    )

    resp = llm.invoke([system_msg, human_msg])
    raw = (resp.content or "").strip()

    # Robust parsing
    try:
        parsed = json.loads(raw)
    except Exception:
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        if "{" in cleaned and "}" in cleaned:
            candidate = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = {
                    "action": "stop",
                    "reason": f"Invalid JSON: {raw}",
                    "suggested_query": "",
                    "candidates": [],
                }
        else:
            parsed = {
                "action": "stop",
                "reason": f"Invalid JSON: {raw}",
                "suggested_query": "",
                "candidates": [],
            }

    action = (parsed.get("action") or "stop").lower()
    if action not in ("select", "refine", "stop"):
        action = "stop"

    reason = parsed.get("reason") or ""
    suggested_raw = parsed.get("suggested_query") or ""
    candidates_raw = parsed.get("candidates") or []

    # HARD GUARD: don't stop on first try with empty hits
    hits_empty = not hits or (isinstance(hits, list) and len(hits) == 0)
    if action == "stop" and hits_empty and iteration < 1:
        action = "refine"
        if not suggested_raw:
            reason = (reason or "") + " | overridden to refine: empty hits on first iteration."

    # Normalize suggested_query
    cleaned_suggested = ""
    if suggested_raw:
        cleaned_suggested = _normalize_query(suggested_raw, max_tokens=4)

    selected_entities: List[Dict[str, Any]] = []

    if action == "select":
        # ensure candidates are sane; store id/label/description/score/rank
        for c in candidates_raw:
            cid = c.get("id")
            if not cid:
                continue

            label = c.get("label", "") or ""
            desc = c.get("description", "") or ""
            score_val = float(c.get("score", 0.0))

            desc_low = desc.lower()
            if any(
                term in desc_low
                for term in [
                    "scholarly article",
                    "scientific article",
                    "journal article",
                    "academic paper",
                ]
            ):
                continue

            if score_val < 0.5:
                continue

            selected_entities.append(
                {
                    "id": cid,
                    "label": label,
                    "description": desc,
                    "score": score_val,
                    "rank": int(c.get("rank", 0)),
                }
            )

        # if nothing survives filtering, downgrade action to refine
        if not selected_entities:
            action = "refine"
            reason = (reason or "") + " | no non-article candidates above score threshold; refine query."

    if action == "refine":
        history = state.get("query_history") or []
        current_q = state.get("query") or ""
        if (
            not cleaned_suggested
            or cleaned_suggested == current_q
            or cleaned_suggested in history
        ):
            action = "stop"
            reason = (reason or "") + " | stopping: no new query to try."
            cleaned_suggested = ""
        else:
            state["iteration"] = state.get("iteration", 0) + 1

    state["eval_action"] = action
    state["eval_reason"] = reason
    state["eval_suggested_query"] = cleaned_suggested
    state["selected_entities"] = selected_entities

    print(
        f"[agent_evaluate] action={action} reason='{reason}' "
        f"suggested='{cleaned_suggested}' selected={len(selected_entities)}"
    )
    return state


# ----------------------------
# ROUTER
# ----------------------------

def router_after_eval(state: PipelineState) -> str:
    action = (state.get("eval_action") or "stop").lower()
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if action == "refine" and iteration < max_iterations:
        print(f"[router] refine -> iteration {iteration}")
        return "refine"  # mapped to generate_query
    print(f"[router] done with action={action}, iteration={iteration}")
    return "done"        # mapped to agent_postprocess


# ----------------------------
# POSTPROCESS NODE
# ----------------------------

def node_agent_postprocess(state: PipelineState) -> PipelineState:
    q = state.get("query") or ""
    hits = state.get("wikidata_hits") or []
    action = state.get("eval_action")
    reason = state.get("eval_reason")
    selected = state.get("selected_entities") or []

    state["agent_notes"] = (
        f"Final action={action}, query='{q}', hits={len(hits)}, "
        f"selected={len(selected)}, reason='{reason}'"
    )
    return state


# ----------------------------
# GRAPH BUILD
# ----------------------------

def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("generate_query", node_generate_query)
    graph.add_node("search_wikidata", node_search_wikidata)
    graph.add_node("agent_evaluate", node_agent_evaluate)
    graph.add_node("agent_postprocess", node_agent_postprocess)

    graph.set_entry_point("generate_query")
    graph.add_edge("generate_query", "search_wikidata")
    graph.add_edge("search_wikidata", "agent_evaluate")
    graph.add_conditional_edges(
        "agent_evaluate",
        router_after_eval,
        {
            "refine": "generate_query",
            "done": "agent_postprocess",
        },
    )
    graph.add_edge("agent_postprocess", END)
    return graph.compile()


# ----------------------------
# CONTEXT LOADER
# ----------------------------

def load_context(path: str):
    with open(path, "rb") as f:
        data = f.read()
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return json.loads(data.decode(enc))
        except Exception:
            continue
    raise UnicodeDecodeError(
        "utf-8/utf-16 variants", b"", 0, 0, "Unable to decode input file"
    )


# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python langgraph_utils.py <context_json_path>", file=sys.stderr)
        sys.exit(1)

    context = load_context(sys.argv[1])

    app = build_graph()
    initial_state: PipelineState = {
        "context": context,
        "iteration": 0,
        "max_iterations": 3,
        "query_history": [],
    }
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    print("Query history:")
    for i, q in enumerate(final_state.get("query_history") or [], 1):
        print(f"{i}. {q}")

    print("\nLast query:", final_state.get("query"))
    print("Eval action:", final_state.get("eval_action"))
    print("Eval reason:", final_state.get("eval_reason"))
    print("Eval suggested:", final_state.get("eval_suggested_query"))

    print("\nSelected entities:")
    selected = final_state.get("selected_entities") or []
    if not selected:
        print("None.")
    else:
        for c in selected:
            print(
                f"- {c.get('id')} | {c.get('label')} | "
                f"{c.get('description')} | score={c.get('score')} rank={c.get('rank')}"
            )

    print("\nWikidata hits (top 5):")
    hits_out = final_state.get("wikidata_hits") or []
    if not hits_out:
        print("No hits.")
    else:
        for h in hits_out[:5]:
            print(
                "-",
                h.get("id"),
                "|",
                h.get("label"),
                "|",
                h.get("description"),
            )

    print("\nAgent notes:")
    print(final_state.get("agent_notes"))
