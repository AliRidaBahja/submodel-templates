import requests
import json
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import sys
load_dotenv()

# force UTF-8 output so labels from Wikidata don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

UA = "AAS-Wikidata-Linker/0.1 (mailto:you@example.com)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

class PipelineState(TypedDict, total=False):
    # context.json
    context: Dict[str, Any]

    # wbsearchentities
    query: str
    search_type: str
    wikidata_search_hits: List[Dict[str, Any]]

    # SPARQL
    sparql_query: str
    wikidata_sparql_results: List[Dict[str, Any]]

    # Linked Data Interface
    wikidata_entities: Dict[str, Any]


def node_prepare_search_input(state: PipelineState) -> PipelineState:
    ctx = state.get("context", {})
    target = ctx.get("target", {})
    parents = ctx.get("parents", [])

    id_short = (target.get("idShort") or "").replace("_", " ").strip()
    desc_en = (target.get("description_en") or "").strip()

    parent_desc = ""
    if parents:
        p0 = parents[0]
        parent_desc = (p0.get("description_en") or "").strip()

    parts = [id_short, desc_en, parent_desc]
    query = " ".join(p for p in parts if p)
    state["query"] = (query or id_short)[:250]

    model_type = (target.get("modelType") or "").lower()
    state["search_type"] = "property" if model_type == "property" else "item"

    return state


# def node_search_wikimedia(state: PipelineState) -> PipelineState:
#     query = (state.get("query") or "").strip()
#     if not query:
#         state["wikidata_search_hits"] = []
#         return state

#     search_type = state.get("search_type", "item")

#     params = {
#         "action": "wbsearchentities",
#         "format": "json",
#         "language": "en",
#         "search": query,
#         "type": search_type,
#         "limit": 10,
#     }

#     try:
#         resp = SESSION.get("https://www.wikidata.org/w/api.php", params=params, timeout=10)
#         resp.raise_for_status()
#         data = resp.json()
#         state["wikidata_search_hits"] = data.get("search", [])
#     except Exception:
#         state["wikidata_search_hits"] = []

#     return state

def node_search_wikimedia(state: PipelineState) -> PipelineState:
    query = (state.get("query") or "").strip()
    if not query:
        state["wikidata_search_hits"] = []
        return state

    def _call(search_type: str) -> List[Dict[str, Any]]:
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": query,
            "type": search_type,
            "limit": 10,
        }
        try:
            resp = SESSION.get("https://www.wikidata.org/w/api.php", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("search", [])
        except Exception:
            return []

    # first try with whatever is in state (usually "item" or "property")
    search_type = state.get("search_type", "item")
    hits = _call(search_type)

    # fallback: if property got 0 hits, retry as item
    if not hits and search_type == "property":
        hits = _call("item")
        state["search_type"] = "item"

    state["wikidata_search_hits"] = hits
    return state

def node_run_wikidata_sparql(state: PipelineState) -> PipelineState:
    sparql = (state.get("sparql_query") or "").strip()
    if not sparql:
        state["wikidata_sparql_results"] = []
        return state

    params = {"query": sparql, "format": "json"}

    try:
        resp = SESSION.get(
            "https://query.wikidata.org/sparql",
            params=params,
            headers={"Accept": "application/sparql-results+json"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        state["wikidata_sparql_results"] = data.get("results", {}).get("bindings", [])
    except Exception:
        state["wikidata_sparql_results"] = []

    return state


def _extract_candidate_ids(state: PipelineState) -> List[str]:
    ids: List[str] = []

    for hit in state.get("wikidata_search_hits", []):
        id_ = hit.get("id")
        if id_:
            ids.append(id_)

    for row in state.get("wikidata_sparql_results", []):
        for v in row.values():
            val = v.get("value", "")
            if "www.wikidata.org/entity/" in val:
                ids.append(val.rsplit("/", 1)[-1])

    seen = set()
    out: List[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def node_fetch_wikidata_entities(state: PipelineState) -> PipelineState:
    candidate_ids = _extract_candidate_ids(state)
    entities: Dict[str, Any] = {}

    for id_ in candidate_ids:
        try:
            url = f"https://www.wikidata.org/wiki/Special:EntityData/{id_}.json"
            resp = SESSION.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            ent = data.get("entities", {}).get(id_)
            if ent is not None:
                entities[id_] = ent
        except Exception:
            continue

    state["wikidata_entities"] = entities
    return state

# ----- AI query generator node -----

def node_generate_query_ai(state: PipelineState) -> PipelineState:
    ctx = state.get("context", {})
    target = ctx.get("target", {})
    submodel = ctx.get("submodel") or {}

    system = SystemMessage(
        content=(
            "you generate a wikidata search query for wbsearchentities. input: json with 'target' and optional 'submodel'. "
            "goal: produce a 1–3 word query that matches how wikidata labels entities. "
            "rules: base the query only on target.idshort and target.description.en. "
            "strip aas-internal noise: asset, assets, interface, interfaces, description, template, schema, dataset, datasets, "
            "ai, security, configuration, collection, model, models, metadata, affordance. "
            "do not include the word 'property'. lowercase. no punctuation except spaces. "
            "output a short label-like phrase, never a sentence. "
            "return json only: { \"query\": \"<short label phrase>\", \"search_type\": \"item\" or \"property\" }. "
            "search_type = \"property\" only if target.modeltype == \"property\"; otherwise \"item\"."
        )
    )

    human = HumanMessage(
        content=json.dumps(
            {
                "target": target,
                "submodel": submodel,
            },
            ensure_ascii=False,
        )
    )

    resp = llm.invoke([system, human])

    try:
        data = json.loads(resp.content)
    except Exception:
        # fallback: very dumb query from target.idShort
        id_short = (target.get("idShort") or "").replace("_", " ")
        state["query"] = id_short[:250]
        model_type = (target.get("modelType") or "").lower()
        state["search_type"] = "property" if model_type == "property" else "item"
        return state

    query = (data.get("query") or "").strip()
    if not query:
        id_short = (target.get("idShort") or "").replace("_", " ")
        query = id_short

    state["query"] = query[:250]
    st = (data.get("search_type") or "").lower()
    if st not in ("item", "property"):
        model_type = (target.get("modelType") or "").lower()
        st = "property" if model_type == "property" else "item"
    state["search_type"] = st

    return state


graph = StateGraph(PipelineState)
graph.add_node("prepare_search_input", node_prepare_search_input)
graph.add_node("search_wikimedia", node_search_wikimedia)
graph.add_node("run_wikidata_sparql", node_run_wikidata_sparql)
graph.add_node("fetch_wikidata_entities", node_fetch_wikidata_entities)
graph.add_node("generate_query", node_generate_query_ai)

# wire the pipeline:
# generate_query  -> search_wikimedia -> fetch_wikidata_entities
graph.add_edge("generate_query", "search_wikimedia")
graph.add_edge("search_wikimedia", "fetch_wikidata_entities")

graph.set_entry_point("generate_query")
graph.set_finish_point("fetch_wikidata_entities")

app = graph.compile()

if __name__ == "__main__":
    from published.target_si_to_json import get_target_si
    from build_context import call_context
    import sys 

    # for aas_path,target_list in get_target_si("C:\\Users\\aliba\\Paper\\submodel-templates\\published\\Wireless Communication\\1\\0\\IDTA 02022-1-0_Template_Wireless Communication.json").items():
    for aas_path,target_list in get_target_si().items():

        if aas_path == "C:\\Users\\aliba\\Paper\\submodel-templates\\published\\MTP\\1\\0\\IDTA 02001-1-0_Subomdel_MTPv1.0-rc2-with-documentation.json":
            continue

        for target in target_list:

            print("(!) Target is ", target["idShort"])
            try:
                context = call_context(aas_path, target["absPath"])
                
                # run full pipeline: generate_query -> search_wikimedia -> fetch_wikidata_entities
                result = app.invoke({"context": context})

                print("query:", result.get("query"))
                print("search_type:", result.get("search_type"))
                print("num_hits:", len(result.get("wikidata_search_hits", [])))
                entities = result.get("wikidata_entities", {})
                print("entities_fetched:", len(entities))

                for eid, ent in entities.items():
                    label = (
                        ent.get("labels", {})
                        .get("en", {})
                        .get("value")
                    )
                    if not label:
                        labels = ent.get("labels", {})
                        if labels:
                            label = next(iter(labels.values())).get("value")
                    print(f"  {eid}: {label}")
                
            except Exception as e:
                print("BAD PATH:", target["absPath"])
                raise
            
