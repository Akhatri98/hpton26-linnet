"""HTTP API for the web UI.

    .venv/Scripts/python -m uvicorn linnet.api:app --port 8000

Privacy by construction:
  - patient data arrives in the request body, is solved in memory, and is
    returned; it is never written to disk or logged (access logs show paths only);
  - the language model is never called from here. It only ever reads public
    FDA labels (library/extract.py), offline, before any patient exists.

Endpoints:
  GET  /api/products              library products, for the intake picker
  GET  /api/patients              demo (constructed) patients
  GET  /api/patients/{name}
  POST /api/check                 patient JSON -> tier report (+ greedy comparison)
  GET  /api/library               sign-off items + extracted rows (metadata view)
  POST /api/library/verify        {"id": ..., "verified": true|false}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from library.load import load
from library.store import LocalJsonStore

from .baseline import run_greedy
from .patient import Patient
from .serialize import report_json
from .tiers import run

DATA = Path(os.environ.get("LINNET_DATA", "data"))
WEB = Path(__file__).resolve().parent.parent / "web"
TIMEOUT_MS = 10_000

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
             "dose_note": p.dose_note, "verified": p.verified, "url": p.url}
            for p in sorted(lib.products.values(), key=lambda p: p.display_name)]


@app.get("/api/patients")
def patients():
    out = []
    for f in sorted((DATA / "patients").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out.append({"id": f.stem, "name": d["name"]})
    return out


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
        raise HTTPException(422, [{"where": ".".join(map(str, x["loc"])), "problem": x["msg"]} for x in e.errors()])
    lib = _lib()
    unknown = [it.product for it in p.items if it.product not in lib.products]
    if unknown:
        raise HTTPException(422, [{"where": "items", "problem": f"not in the library: {u}"} for u in unknown])
    report = run(lib, p, timeout_ms=TIMEOUT_MS)
    g = run_greedy(lib, p) if compare else None
    return report_json(report, lib, g)


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


class Verify(BaseModel):
    id: str
    verified: bool


@app.post("/api/library/verify")
def verify(v: Verify):
    try:
        return store.set_verified(v.id, v.verified)
    except KeyError:
        raise HTTPException(404, f"no library item {v.id}")


if WEB.is_dir():
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")
