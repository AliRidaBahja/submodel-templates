import requests

def wikidata_search(term, lang="en", limit=1):
    """Return Wikidata entity candidates for a term."""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": term,
        "language": lang,
        "uselang": lang,
        "format": "json",
        "limit": limit
    }
    r = requests.get(url, params=params, headers={"User-Agent": "AAS-Linker/0.1"})
    r.raise_for_status()
    data = r.json().get("search", [])
    print("DEBUG: ", data)
    return [
        {
            "id": e["id"],
            "label": e.get("label"),
            "description": e.get("description"),
            "url": f"https://www.wikidata.org/entity/{e['id']}"
        }
        for e in data
    ]

if __name__ == "__main__":
    semantic_id = "carbon footprint"
    results = wikidata_search(semantic_id)
    for r in results:
        print(f"{r['id']}: {r['label']} — {r['description']}\n  {r['url']}")
