"""Product -> active ingredients, via NLM RxNorm (deterministic).

    python -m library.rxnorm --ndc 23155-512-30
    python -m library.rxnorm --name "acetaminophen 300 mg codeine 30 mg tablet"
    python -m library.rxnorm --name "..." --llm-fallback      (only used if RxNorm has nothing)

RxNorm is the algorithmic path and always runs first. The LLM fallback is
probabilistic, runs only when RxNorm returns no ingredients and the flag is
given, and its answer is always marked source="llm", unverified.

Responses are cached under data/cache/rxnorm/ (NLM asks clients to cache).
Note: NLM's separate drug-interaction API was discontinued on 2 Jan 2024;
only RxNorm's terminology services are used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

BASE = "https://rxnav.nlm.nih.gov/REST"
CACHE = Path("data/cache/rxnorm")


@dataclass
class Ingredients:
    source: str  # "rxnorm" | "llm" | "none"
    rxcui: Optional[str] = None
    concept_name: Optional[str] = None
    ingredients: list[str] = field(default_factory=list)  # IN names, e.g. "acetaminophen"
    precise: list[str] = field(default_factory=list)  # PIN names, e.g. "codeine phosphate"
    verified: bool = False  # RxNorm results are authoritative data; LLM results never are
    note: str = ""


def _get(path: str, params: dict, cache: Path = CACHE) -> dict:
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    f = cache / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    cache.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"_url": url, **data}), encoding="utf-8")
    return data


def _ndc11(ndc: str) -> str:
    """'23155-512-30' (5-3-2) -> '23155051230'. Pads each segment to 5-4-2."""
    parts = ndc.split("-")
    if len(parts) != 3:
        return ndc.replace("-", "")
    a, b, c = parts
    return a.zfill(5) + b.zfill(4) + c.zfill(2)


def rxcui_for(ndc: Optional[str] = None, name: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    if ndc:
        s = _get("ndcstatus.json", {"ndc": _ndc11(ndc)}).get("ndcStatus") or {}
        if s.get("rxcui"):
            return s["rxcui"], s.get("conceptName")
    if name:
        cands = (_get("approximateTerm.json", {"term": name, "maxEntries": 1})
                 .get("approximateGroup", {}).get("candidate") or [])
        if cands:
            return cands[0]["rxcui"], cands[0].get("name")
    return None, None


def ingredients_for(ndc: Optional[str] = None, name: Optional[str] = None,
                    llm_fallback: Optional[Callable[[str], list[str]]] = None) -> Ingredients:
    rxcui, concept = rxcui_for(ndc, name)
    if rxcui:
        groups = _get(f"rxcui/{rxcui}/related.json", {"tty": "IN PIN"}).get("relatedGroup", {}).get("conceptGroup") or []
        names = {g["tty"]: [c["name"] for c in g.get("conceptProperties", [])] for g in groups}
        if names.get("IN"):
            return Ingredients("rxnorm", rxcui, concept, sorted(names["IN"]), sorted(names.get("PIN", [])),
                               verified=True, note="from RxNorm")
    if llm_fallback is not None:
        guess = llm_fallback(name or ndc or "")
        return Ingredients("llm", None, None, sorted(guess), [], verified=False,
                           note="RxNorm had no match; ingredients GUESSED by a language model. Verify.")
    return Ingredients("none", rxcui, concept, note="RxNorm had no ingredient data for this product.")


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="library.rxnorm")
    ap.add_argument("--ndc")
    ap.add_argument("--name")
    ap.add_argument("--llm-fallback", action="store_true",
                    help="if RxNorm has nothing, ask a language model (result marked unverified)")
    args = ap.parse_args(argv)
    fallback = None
    if args.llm_fallback:
        from .llm import ask_ingredients
        fallback = ask_ingredients
    print(json.dumps(asdict(ingredients_for(args.ndc, args.name, fallback)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
