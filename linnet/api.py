"""HTTP API for the web UI.

    .venv/Scripts/python -m uvicorn linnet.api:app --port 8000

Privacy by construction:
  - patient data arrives in the request body, is solved in memory, and is
    returned; it is never written to disk or logged (access logs show paths only);
  - the language model is never called from here. It only ever reads public
    FDA labels (library/extract.py), offline, before any patient exists.

Endpoints:
  GET  /api/products              library products, for the intake picker
  GET  /api/patients              saved patients (demo and NHANES), for the switcher
  GET  /api/patients/tiers        tier of each saved patient (cached), for the switcher's dots
  GET  /api/patients/{name}
  POST /api/check                 patient JSON -> tier report (+ greedy comparison)
  GET  /api/library               sign-off items + extracted rows (metadata view)
  POST /api/library/verify        {"id": ..., "verified": true|false}
  GET  /api/review/queue          cards to review, one at a time (+ any peer opinion)
  POST /api/review/decide         approve / reject / flag one card; saved at once
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from library.load import load
from library import review_queue as rq
from library.store import LocalJsonStore

from . import isolated
from .patient import Patient

DATA = Path(os.environ.get("LINNET_DATA", "data"))
WEB = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="LINNET", docs_url="/api/docs", openapi_url="/api/openapi.json")
store = LocalJsonStore(DATA)


def _lib():
    # Loaded per request: the library is small, and sign-offs take effect at once.
    return load(DATA)


@app.get("/api/products")
def products():
    lib = _lib()
    return [{"name": p.name, "display_name": p.display_name,
             "ingredients": {k: f"{v.amount} {v.unit}" for k, v in p.ingredients.items()},
             "as_needed_max": p.otc_max_doses.count if p.otc_max_doses else None,
             "dose_note": p.dose_note, "verified": p.verified, "url": p.url,
             "unit_name": p.unit_name, "units_per_dose": float(p.units_per_dose),
             "label_max_units_per_dose": float(p.label_max_units_per_dose) if p.label_max_units_per_dose is not None else None}
            for p in sorted(lib.products.values(), key=lambda p: p.display_name)]


@app.get("/api/patients")
def patients():
    out = []
    for f in sorted((DATA / "patients").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        group = "From NHANES 2017-2020 (real routine and medications)" if f.stem.startswith("nhanes-") else "Demo patients"
        out.append({"id": f.stem, "name": d["name"], "group": group})
    out.sort(key=lambda p: (p["group"] != "Demo patients", p["id"]))
    return out


_tiers = {"key": None, "outcomes": {}}
_tiers_lock = threading.Lock()


def _data_key() -> tuple:
    files = [*(DATA / "ingredients").glob("*.json"), *(DATA / "products").glob("*.json"),
             *(DATA / "patients").glob("*.json"), DATA / "groups.json"]
    return (str(DATA.resolve()), *sorted((f.name, f.stat().st_mtime_ns) for f in files if f.exists()))


@app.get("/api/patients/tiers")
def patient_tiers():
    """Tier of each saved patient, for the dots in the switcher. Cached until a library or
    patient file changes (a review decision changes them)."""
    with _tiers_lock:
        key = _data_key()
        if _tiers["key"] != key:
            pts = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted((DATA / "patients").glob("*.json"))}
            _tiers["outcomes"] = isolated.outcomes(pts, DATA)
            _tiers["key"] = key
        return _tiers["outcomes"]


@app.get("/api/patients/{pid}")
def patient(pid: str):
    f = DATA / "patients" / f"{pid}.json"
    if not f.is_file() or f.parent.resolve() != (DATA / "patients").resolve():
        raise HTTPException(404, "no such demo patient")
    return json.loads(f.read_text(encoding="utf-8"))


@app.post("/api/check")
def check(body: dict, compare: bool = True):
    try:
        p = Patient.model_validate(body)
    except ValidationError as e:
        raise HTTPException(422, [{"where": ".".join(map(str, x["loc"])), "problem": x["msg"]} for x in e.errors()]) from e
    lib = _lib()
    unknown = [it.product for it in p.items if it.product not in lib.products]
    if unknown:
        raise HTTPException(422, [{"where": "items", "problem": f"not in the library: {u}"} for u in unknown])
    try:
        # Fresh worker process per check: identical results every run, and a hard time limit.
        return isolated.check(p.model_dump(mode="json"), DATA, compare)
    except isolated.CheckTimeout as e:
        raise HTTPException(504, [{"where": "solver", "problem": str(e)}]) from e


def _extracted():
    tables = {k: [] for k in ("dosing", "limits", "administration", "memberships", "unaccounted", "rejected")}
    for f in sorted((DATA / "extracted").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        base = {"ingredient": d["ingredient"], "model": d["model"], "label_url": d.get("label_url", "")}
        for k, src in (("dosing", "dosing"), ("limits", "limits"), ("administration", "administration"),
                       ("memberships", "memberships"), ("unaccounted", "unaccounted_for"), ("rejected", "rejected")):
            tables[k] += [{**base, **r} for r in d.get(src, [])]
    tables["unaccounted"].sort(key=lambda u: (u.get("priority", 9), u["ingredient"]))
    return tables


@app.get("/api/library")
def library():
    return {"items": store.items(), "extracted": _extracted()}


@app.get("/api/review/queue")
def review_queue():
    from library.peer import opinions
    ops = opinions(DATA)  # a second reviewer's opinion: shown, never applied
    return [{**c, "peer": ops.get(c["id"])} for c in rq.queue(DATA)]


class Decision(BaseModel):
    id: str
    decision: str  # "approved" | "rejected" | "flagged"
    by: str
    role: str = ""
    note: str = ""


@app.post("/api/review/decide")
def review_decide(d: Decision):
    # Written to disk (atomically) before this returns: stopping the server never loses a decision.
    try:
        return rq.decide(DATA, d.id, d.decision, d.by, d.role, d.note)
    except KeyError as e:
        raise HTTPException(404, [{"where": "id", "problem": f"no review item {d.id} (already decided?)"}]) from e
    except ValueError as e:
        raise HTTPException(422, [{"where": "decision", "problem": str(e)}]) from e


class Verify(BaseModel):
    id: str
    verified: bool


@app.post("/api/library/verify")
def verify(v: Verify):
    try:
        return store.set_verified(v.id, v.verified)
    except KeyError as e:
        raise HTTPException(404, f"no library item {v.id}") from e


if WEB.is_dir():
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")
