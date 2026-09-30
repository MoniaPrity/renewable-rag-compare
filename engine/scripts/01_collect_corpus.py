"""
Collects a renewable-energy text corpus from Wikipedia ONLY, using a
deterministic category-tree crawl.

WHY THIS CHANGED (v2 - dataset rebuild):
  arXiv was dropped entirely. Every contamination cluster ever found in
  this project (dark energy/particle physics, agricultural remote sensing,
  heliophysics) came from arXiv's keyword search matching ambiguous terms
  ("energy", "grid", "solar", "biomass") inside unrelated papers'
  abstracts. Wikipedia was never the source of contamination - only the
  size of the hand-curated title list (~20 pages) was limiting.

  This version replaces the hand-curated list with a BFS crawl of the
  Wikipedia category tree, starting from a small set of unambiguous
  renewable-energy root categories (Category:Renewable energy,
  Category:Solar power, etc.) and recursing down to MAX_DEPTH. This is
  topic-bounded BY CONSTRUCTION - there is no keyword-match step that can
  accidentally pull in an off-topic document - so it produces a much
  larger corpus without reintroducing arXiv-style contamination.

  Two safety nets are still applied, same spirit as the old collector:
    - domain_filters.text_is_off_topic() on every fetched article's
      title+body, in case a category branch still drifts off-topic.
    - the SAME check applied to subcategory NAMES before recursing into
      them, so the crawl doesn't even walk into a branch like
      "Solar phenomena" hanging off a solar-power category tree.

Run from the engine/ folder:
    python scripts/01_collect_corpus.py

Output: engine/data/raw/wikipedia_*.json
"""
import json
import time
import sys
import socket
from pathlib import Path

socket.setdefaulttimeout(15)  # stops any hung HTTP call instead of waiting forever

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `app` imports work
from app.config import RAW_DIR
from app.domain_filters import text_is_off_topic

import requests
import wikipedia

API_URL = "https://en.wikipedia.org/w/api.php"

# Root categories to crawl from. Deliberately narrow and unambiguous - no
# root like "Energy" or "Physics" that could branch into off-topic
# territory. Add more here if you want a bigger corpus later.
ROOT_CATEGORIES = [
    "Renewable energy",
    "Solar power",
    "Wind power",
    "Hydroelectricity",
    "Geothermal energy",
    "Bioenergy",
    "Tidal power",
    "Wave power",
    "Sustainable energy",
    "Energy storage",
]

MAX_DEPTH = 2                 # how many category levels deep to recurse
MAX_PAGES_PER_CATEGORY = 40   # cap per category call
MAX_TOTAL_PAGES = 300         # hard cap on the whole crawl
MIN_ARTICLE_CHARS = 500       # skip stub pages - not enough content to be useful


def _api_get(params: dict) -> dict:
    params = {**params, "format": "json"}
    resp = requests.get(
        API_URL, params=params,
        headers={"User-Agent": "cse499b-corpus-collector/1.0 (student research project)"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _category_members(category: str, cmtype: str, limit: int = MAX_PAGES_PER_CATEGORY) -> list[str]:
    """cmtype: 'page' or 'subcat'. Returns member titles (Category: prefix
    stripped for subcats)."""
    members = []
    cmcontinue = None
    while len(members) < limit:
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Category:{category}", "cmtype": cmtype,
            "cmlimit": min(limit - len(members), 500),
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        try:
            data = _api_get(params)
        except Exception as e:
            print(f"    [warn] category API call failed for '{category}': {e}")
            break
        batch = data.get("query", {}).get("categorymembers", [])
        for item in batch:
            title = item["title"]
            if cmtype == "subcat":
                title = title.replace("Category:", "", 1)
            members.append(title)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return members


def discover_titles() -> list[str]:
    """BFS the category tree from ROOT_CATEGORIES down to MAX_DEPTH,
    collecting article titles. Skips recursing into any subcategory whose
    NAME trips the off-topic denylist, so contamination is stopped before
    a single page is even fetched."""
    seen_categories: set[str] = set()
    seen_titles: set[str] = set()
    frontier = [(cat, 0) for cat in ROOT_CATEGORIES]

    while frontier and len(seen_titles) < MAX_TOTAL_PAGES:
        category, depth = frontier.pop(0)
        key = category.lower()
        if key in seen_categories:
            continue
        seen_categories.add(key)

        print(f"  crawling category '{category}' (depth {depth})")
        pages = _category_members(category, cmtype="page")
        for title in pages:
            if title.lower() in seen_titles:
                continue
            seen_titles.add(title)
            if len(seen_titles) >= MAX_TOTAL_PAGES:
                break

        if depth < MAX_DEPTH:
            subcats = _category_members(category, cmtype="subcat", limit=20)
            for subcat in subcats:
                if text_is_off_topic(subcat):
                    print(f"    [skip subcategory - off-topic name] {subcat}")
                    continue
                if subcat.lower() not in seen_categories:
                    frontier.append((subcat, depth + 1))
        time.sleep(0.3)  # be polite to the API

    return list(seen_titles)


def _fetch_page_and_content(title: str, retries: int = 3, backoff: float = 2.0):
    """wikipedia.page() and page.content each make their own API call, and
    Wikipedia occasionally returns an empty/invalid body (rate limiting or
    a transient hiccup). Retry both together with backoff. Disambiguation
    and missing-page errors are deterministic - not worth retrying."""
    last_err = None
    for attempt in range(retries):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            text = page.content
            return page.title, text
        except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
            raise
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_err


def collect_wikipedia():
    wikipedia.set_lang("en")
    print("Discovering article titles from category tree...")
    titles = discover_titles()
    print(f"  found {len(titles)} candidate articles across {len(ROOT_CATEGORIES)} root categories\n")

    saved, skipped_offtopic, skipped_error = 0, 0, 0
    for title in titles:
        out_path = RAW_DIR / f"wikipedia_{title.replace(' ', '_').replace('/', '_')}.json"
        if out_path.exists():
            print(f"  skip (exists): {title}")
            saved += 1
            continue
        try:
            page_title, text = _fetch_page_and_content(title)
        except Exception as e:
            print(f"  [warn] could not fetch '{title}': {e}")
            skipped_error += 1
            continue

        if len(text) < MIN_ARTICLE_CHARS:
            print(f"  [skip - too short] {page_title} ({len(text)} chars)")
            continue

        # Safety net: category names are curated, but a page's actual body
        # can still occasionally drift (e.g. a borderline article that got
        # tagged into a renewable-energy category by editors for a loose
        # reason). Same check as the old collector.
        if text_is_off_topic(f"{page_title} {text}"):
            print(f"  [skip - off-topic] {page_title}")
            skipped_offtopic += 1
            continue

        out_path.write_text(
            json.dumps({"source": f"wikipedia:{page_title}", "title": page_title, "text": text}),
            encoding="utf-8",
        )
        print(f"  saved: {page_title} ({len(text)} chars)")
        saved += 1
        time.sleep(1.0)  # be polite to the API

    print(f"\nDone. {saved} articles saved, {skipped_offtopic} skipped as off-topic, "
          f"{skipped_error} failed to fetch.")


if __name__ == "__main__":
    print("Collecting Wikipedia articles via category-tree crawl (arXiv removed)...")
    collect_wikipedia()
    print("\nDone. Raw files are in engine/data/raw/")
