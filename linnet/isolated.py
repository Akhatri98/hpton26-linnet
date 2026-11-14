"""Run one timing check in a fresh worker process, with a hard wall-clock limit.

Why: Z3 keeps global state across solves in one process. After hundreds of
solves the same regimen can run far slower, hit per-solve timeouts, and so
return fewer tier-1 options than it does from a clean start (measured with the
C++ core: 0.04 s fresh vs over 40 s late in a 500-regimen sweep). A fresh process
per check makes results identical run to run, and lets a stuck check be killed
outright.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

LIMIT_S = 30  # whole check, all tiers, including conflict shrinking


def _work(patient: dict, data: str, compare: bool) -> dict:
    from library.load import load

    from .baseline import run_greedy
    from .patient import Patient
    from .serialize import report_json
    from .tiers import run

    lib = load(Path(data))
    p = Patient.model_validate(patient)
    report = run(lib, p)
    return report_json(report, lib, run_greedy(lib, p) if compare else None)


class CheckTimeout(Exception):
    pass


_pool: ProcessPoolExecutor | None = None


def _new_pool() -> ProcessPoolExecutor:
    # One task per worker: every check starts from a clean Z3. The next worker is
    # started as soon as the previous one exits, so its start-up overlaps idle time.
    return ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1, mp_context=mp.get_context("spawn"))


def check(patient: dict, data: Path, compare: bool = True, limit_s: float = LIMIT_S) -> dict:
    global _pool
    if _pool is None:
        _pool = _new_pool()
    fut = _pool.submit(_work, patient, str(data), compare)
    try:
        return fut.result(timeout=limit_s)
    except FutureTimeout:
        # Kill the stuck worker; the next check gets a fresh pool.
        for proc in list(getattr(_pool, "_processes", {}).values()):
            proc.kill()
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
        raise CheckTimeout(f"the check did not finish within {limit_s:g} s") from None


def _outcome(patient: dict, data: str) -> str:
    from library.load import load

    from .patient import Patient
    from .tiers import run

    return run(load(Path(data)), Patient.model_validate(patient)).outcome


def outcomes(patients: dict[str, dict], data: Path, workers: int = 4, limit_s: float = LIMIT_S) -> dict[str, str]:
    """Tier outcome of each saved patient, each in its own fresh process (as a single check
    would be); 'unknown' if one does not finish in time."""
    pool = ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1, mp_context=mp.get_context("spawn"))
    out = {}
    try:
        futs = {pid: pool.submit(_outcome, p, str(data)) for pid, p in patients.items()}
        for pid, fut in futs.items():
            try:
                out[pid] = fut.result(timeout=limit_s)
            except Exception:  # timeout or a crashed worker: say so, never guess
                out[pid] = "unknown"
    finally:
        for proc in list(getattr(pool, "_processes", {}).values()):
            if proc.is_alive():
                proc.kill()
        pool.shutdown(wait=False, cancel_futures=True)
    return out
