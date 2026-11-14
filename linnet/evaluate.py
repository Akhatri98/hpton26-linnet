"""Solver evaluation.  [deterministic, no model calls]

    python -m linnet.evaluate [--n 500] [--seed 1] [--bf-max-doses 8]

Generates random constructed regimens from the library (seeded, reproducible)
and measures four things:

  1. Agreement with exhaustive search. On a 15-minute grid, a backtracking
     search tries every placement. For tier 0 (solve A) and tier 2 (solve B) it
     must agree with Z3: if the grid finds a valid schedule where Z3 said
     "none", that is a bug. (Z3 works on exact times, so it may find off-grid
     schedules the grid misses; those are counted, not errors.)
  2. Greedy vs LINNET. The one-dose-at-a-time heuristic (baseline.py) against
     solve A. A "false alarm" is the heuristic giving up on a regimen that
     LINNET schedules as the patient lives now.
  3. Conflict minimality. Every conflict shown is re-checked with each rule
     removed in turn; each removal must make it satisfiable.
  4. Speed: wall-clock time per full check (all tiers).

Every schedule LINNET returns is also re-checked by the independent verifier
inside tiers.run (it raises if one ever breaks a rule).

Writes data/eval/solver.json and the false-alarm patients to data/eval/false_alarms/.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path

from library.load import build_regimen, load
from solver.encode import build, check
from solver.model import DAY
from solver.verify import violations

from .baseline import STEP, run_greedy
from .conflicts import is_count
from .patient import Patient, fmt_hhmm
from .tiers import run

# Products and the dose patterns a prescriber might plausibly write for them.
# (use, counts, every_hours options). Constructed, for stress-testing only.
MENU = {
    "nimodipine-30mg-cap": ("rx", [6], [None, 4]),
    "colestipol-1g-tab": ("rx", [1, 2], [None]),
    "levothyroxine-100mcg": ("rx", [1], [None]),
    "ciprofloxacin-500mg": ("rx", [2], [None, 12]),
    "apap-codeine-300-30": ("rx", [2, 3, 4], [None]),
    "tums-es-750": ("prn", [1, 2, 3], [None]),
    "acetaminophen-500-otc": ("prn", [1, 2, 3], [None]),
}


def _q15(h: float) -> str:
    m = int(round(h * 4)) * 15 % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def random_patient(rng: random.Random, i: int, routines: list | None = None) -> dict:
    names = rng.sample(sorted(MENU), rng.randint(2, 4))
    items = []
    for n in names:
        use, counts, every = MENU[n]
        c = rng.choice(counts)
        if use == "rx":
            it = {"product": n, "prescribed_count": c}
            e = rng.choice(every)
            if e:
                it["every_hours"] = e
        else:
            it = {"product": n, "reported_count": c}
        items.append(it)
    if routines:  # a real person's day (NHANES), drugs still drawn at random
        r = rng.choice(routines)
        return {"name": f"random regimen {i} on a real NHANES routine", "constructed": True, "items": items,
                "routine": {"meals": r["meals"], "sleep": {"start": r["bed"], "end": r["wake"]}}}
    wake = rng.uniform(5.5, 8.5)
    bed = wake + rng.uniform(15, 17)
    return {
        "name": f"random regimen {i}", "constructed": True, "items": items,
        "routine": {"meals": {"breakfast": _q15(wake + rng.uniform(0.25, 1.5)),
                              "lunch": _q15(rng.uniform(11.5, 13.5)),
                              "dinner": _q15(rng.uniform(17.5, 19.5))},
                    "sleep": {"start": _q15(bed), "end": _q15(wake)}},
    }


# --- exhaustive grid search -------------------------------------------------------------

class Budget(Exception):
    pass


def grid_search(regimen, counts: dict[str, int], meals, slots: list[Fraction], budget: int = 200_000):
    """Backtracking over grid slots. Returns a valid schedule, None if none exists
    on the grid, or raises Budget if the search was too large to finish."""
    prods = sorted(regimen.products, key=lambda p: (-counts[p.name], p.name))
    order = sorted(slots)
    placed = {p.name: [] for p in regimen.products}
    steps = [0]

    def place(pi: int, k: int, start: int):
        if pi == len(prods):
            return dict(placed) if not violations(regimen, placed, meals) else None
        p = prods[pi]
        if k == counts[p.name]:
            return place(pi + 1, 0, 0)
        for i in range(start, len(order)):
            steps[0] += 1
            if steps[0] > budget:
                raise Budget()
            placed[p.name] = placed[p.name] + [order[i]]
            if not violations(regimen, placed, meals, partial=True):
                got = place(pi, k + 1, i + 1)
                if got:
                    return got
            placed[p.name] = placed[p.name][:-1]
        return None

    return place(0, 0, 0)


def awake_slots(bed: Fraction, wake: Fraction) -> list[Fraction]:
    n = int(((bed - wake) % DAY) / STEP)
    return [(wake + STEP * i) % DAY for i in range(n + 1)]


# --- evaluation -------------------------------------------------------------------------

def evaluate(n: int, seed: int, bf_max: int, data: Path, routines: list | None = None) -> dict:
    lib = load(data)
    rng = random.Random(seed)
    out = {"n": n, "seed": seed, "routines": "NHANES 2017-18 (real)" if routines else "random", "grid_minutes": 15, "tiers": {}, "times_ms": [],
           "greedy": {"ok": 0, "gave_up": 0, "false_alarms": 0, "greedy_ok_linnet_not_tier0": 0},
           "grid": {"checked": 0, "agree": 0, "off_grid_only": 0, "too_large": 0, "BUGS": []},
           "conflicts": {"checked": 0, "minimal": 0, "not_minimal": []},
           "unfinished": []}  # (case, solves) that hit the per-solve time limit: counted apart, never as bugs
    fa_dir = data / "eval" / ("false_alarms_nhanes" if routines else "false_alarms")
    fa_dir.mkdir(parents=True, exist_ok=True)
    for old in fa_dir.glob("*.json"):
        old.unlink()

    for i in range(n):
        raw = random_patient(rng, i, routines)
        p = Patient.model_validate(raw)
        t0 = time.perf_counter()
        rep = run(lib, p)
        out["times_ms"].append((time.perf_counter() - t0) * 1000)
        out["tiers"][rep.outcome] = out["tiers"].get(rep.outcome, 0) + 1
        if rep.unfinished:
            out["unfinished"].append({"case": i, "solves": rep.unfinished, "outcome": rep.outcome})

        # 2. greedy vs LINNET (same question as solve A)
        g = run_greedy(lib, p)
        if g.ok:
            out["greedy"]["ok"] += 1
            if rep.outcome != "tier0" and not rep.unfinished:  # greedy's schedule is valid, so LINNET missed one: a bug
                out["greedy"]["greedy_ok_linnet_not_tier0"] += 1
        else:
            out["greedy"]["gave_up"] += 1
            if rep.outcome == "tier0":
                out["greedy"]["false_alarms"] += 1
                (fa_dir / f"false-alarm-{i:03d}.json").write_text(json.dumps(raw, indent=1) + "\n", encoding="utf-8")

        # 1. exhaustive grid search, for solve A (tier 0 question) and solve B (tier 2 question)
        bed, wake = p.sleep_window()
        built_a = build_regimen(lib, p.library_items(), "strict")
        reported = p.reported_counts()
        counts_a = {pr.name: (pr.max_doses if pr.prescribed else reported[pr.name]) for pr in built_a.regimen.products}
        questions = [("A", built_a.regimen, counts_a, p.meal_times(), awake_slots(bed, wake), rep.outcome == "tier0")]
        if rep.outcome == "tier2":
            reg_b = rep.built.regimen
            counts_b = {pr.name: pr.max_doses for pr in reg_b.products}
            questions.append(("B", reg_b, counts_b, {}, [STEP * k for k in range(int(DAY / STEP))], False))
        for name, reg, counts, meals, slots, z3_sat in questions:
            if sum(counts.values()) > bf_max:
                continue
            out["grid"]["checked"] += 1
            try:
                found = grid_search(reg, counts, meals, slots)
            except Budget:
                out["grid"]["too_large"] += 1
                continue
            if rep.unfinished and bool(found) != z3_sat:
                continue  # a solve timed out: not an answer, so not a disagreement (counted in "unfinished")
            if found and not z3_sat:
                out["grid"]["BUGS"].append({"case": i, "solve": name, "grid_schedule":
                                            {k: [fmt_hhmm(t) for t in v] for k, v in found.items()}})
            elif z3_sat and not found:
                out["grid"]["off_grid_only"] += 1
            else:
                out["grid"]["agree"] += 1

        # 3. conflict minimality
        if rep.conflicts:
            enc = build(rep.built.regimen)
            for c in rep.conflicts:
                out["conflicts"]["checked"] += 1
                ok = all(check(enc, only=[x for x in c.rules if x != r]).status == "sat" for r in c.rules)
                if ok:
                    out["conflicts"]["minimal"] += 1
                else:
                    out["conflicts"]["not_minimal"].append({"case": i, "rules": c.rules})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}", file=sys.stderr)

    ts = sorted(out["times_ms"])
    out["speed_ms"] = {"median": round(statistics.median(ts), 1), "p95": round(ts[int(0.95 * (len(ts) - 1))], 1),
                       "max": round(ts[-1], 1)}
    del out["times_ms"]
    return out


def summary(r: dict) -> str:
    g, gr, c = r["greedy"], r["grid"], r["conflicts"]
    tier0 = r["tiers"].get("tier0", 0)
    lines = [
        f"SOLVER EVALUATION  ({r['n']} random constructed regimens, seed {r['seed']}, routines: {r['routines']})",
        "",
        "Tiers:            " + ", ".join(f"{k} {v}" for k, v in sorted(r["tiers"].items())),
        "",
        "Greedy heuristic vs LINNET (same question: schedulable as they live now?)",
        f"  greedy gave up on      {g['gave_up']} of {r['n']}",
        f"  FALSE ALARMS           {g['false_alarms']}  (greedy gave up; LINNET found a schedule)"
        + (f"  = {100 * g['false_alarms'] / tier0:.0f}% of the {tier0} schedulable regimens" if tier0 else ""),
        f"  greedy ok, LINNET not  {g['greedy_ok_linnet_not_tier0']}  (must be 0)",
        "",
        f"Exhaustive {r['grid_minutes']}-minute grid search vs Z3",
        f"  checked {gr['checked']}  |  agree {gr['agree']}  |  Z3 found an off-grid schedule only {gr['off_grid_only']}"
        f"  |  too large to search {gr['too_large']}",
        f"  BUGS (grid found a schedule Z3 said does not exist): {len(gr['BUGS'])}  (must be 0)",
        "",
        f"Conflict minimality: {c['minimal']} of {c['checked']} conflicts minimal (every rule needed)",
        "",
        f"Speed per full check: median {r['speed_ms']['median']} ms, p95 {r['speed_ms']['p95']} ms, max {r['speed_ms']['max']} ms",
        f"Checks where a solve hit its time limit: {len(r['unfinished'])}"
        + (f"  {r['unfinished']}" if r["unfinished"] else ""),
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="linnet.evaluate")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bf-max-doses", type=int, default=8)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--nhanes-routines", action="store_true", help="use real NHANES sleep/meal times (data/nhanes/routines.json)")
    a = ap.parse_args(argv)
    routines = json.loads((a.data / "nhanes" / "routines.json").read_text(encoding="utf-8")) if a.nhanes_routines else None
    r = evaluate(a.n, a.seed, a.bf_max_doses, a.data, routines)
    (a.data / "eval").mkdir(parents=True, exist_ok=True)
    (a.data / "eval" / ("solver-nhanes.json" if routines else "solver.json")).write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    print(summary(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
