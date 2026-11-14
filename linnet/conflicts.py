"""M6: turn an unsat core into small, separate, explainable conflicts.

Z3's unsat core is a set of rules that cannot hold together, but it is not
guaranteed minimal, and it shows only the first contradiction found.

shrink():          deletion-based minimisation. Drop one rule at a time and
                   re-check; keep it only if dropping it makes the rest
                   satisfiable. The result is a minimal conflicting set (no
                   rule can be removed), though not necessarily the smallest.
find_conflicts():  after finding one minimal conflict, set its label rules
                   aside and solve again to surface the next, independent one.
                   Dose counts are never set aside (they come from the
                   prescription, not a label). Capped, and each check is
                   timeout-bounded.
"""

from __future__ import annotations

from dataclasses import dataclass

from solver.encode import Encoding, check

COUNT_PREFIXES = ("count::", "max::", "reported::", "rx::")  # prescription facts, never set aside


def is_count(rule_id: str) -> bool:
    return rule_id.startswith(COUNT_PREFIXES)


@dataclass
class Conflict:
    rules: list[str]  # solver rule ids, minimal
    proven_minimal: bool  # False if a re-check timed out during shrinking


def shrink(enc: Encoding, core: list[str]) -> Conflict:
    core = list(core)
    proven = True
    i = 0
    while i < len(core):
        trial = core[:i] + core[i + 1:]
        r = check(enc, only=trial)
        if r.status == "unsat":
            # Still conflicting without core[i]: drop it, and narrow to the new core.
            keep = set(r.core)
            core = [c for c in trial if c in keep]
            # do not advance i: core[i] is now a different rule
        else:
            if r.status == "unknown":
                proven = False
            i += 1
    return Conflict(core, proven)


def find_conflicts(enc: Encoding, first_core: list[str], limit: int = 3) -> list[Conflict]:
    conflicts: list[Conflict] = []
    saved = set(enc.disabled)
    core = first_core
    try:
        while core and len(conflicts) < limit:
            c = shrink(enc, core)
            conflicts.append(c)
            label = [r for r in c.rules if not is_count(r)]
            if not label:
                break
            enc.disabled |= set(label)
            r = check(enc)
            if r.status != "unsat":
                break
            core = r.core
    finally:
        enc.disabled = saved
    return conflicts
