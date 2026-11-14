"""ctypes binding to the native solver core (linnet_solver.cpp).

The encoding and every Z3 call live in C++; this module only moves plain data
across: names as UTF-8, hours and amounts as exact "n/d" rationals. Python's
`z3` module is never imported. The `z3-solver` wheel is used only for the
libz3 binary and headers it ships.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, byref, c_char_p, c_int, c_uint, c_void_p
from fractions import Fraction
from typing import Optional
from collections.abc import Iterable

STATUS = ("sat", "unsat", "unknown")

_S, _H, _I = c_char_p, c_void_p, c_int
_SIGNATURES = {
    "ln_error": ([], _S),
    "ln_new": ([_I, c_uint], _H),
    "ln_free": ([_H], None),
    "ln_product": ([_H, _S, _I, _I, _I], _I),
    "ln_ingredient": ([_H, _S, _S, _S], _I),
    "ln_meal_fixed": ([_H, _S, _S], _I),
    "ln_meal_var": ([_H, _S], _I),
    "ln_min_interval": ([_H, _S, _S, _S], _I),
    "ln_ingredient_interval": ([_H, _S, _S, _S], _I),
    "ln_separation": ([_H, _S, _S, _S, _S, _S], _I),
    "ln_ingredient_ceiling": ([_H, _S, _S, _S], _I),
    "ln_window": ([_H, _S, _S, POINTER(_S), POINTER(_S), _I], _I),
    "ln_meal_gap": ([_H, _S, _S, _S, _S, _S], _I),
    "ln_meal_window": ([_H, _S, _S, _S, _S, _S], _I),
    "ln_sleep": ([_H, _S, _S, _S, _S], _I),
    "ln_meal_shift": ([_H, _S, _S, _S, _S], _I),
    "ln_reported_cap": ([_H, _S, _I, _S], _I),
    "ln_minimize_cost": ([_H], _I),
    "ln_minimize_times": ([_H], _I),
    "ln_maximize_count": ([_H, _S], _I),
    "ln_block_superset": ([_H, _S], _I),
    "ln_tracked_count": ([_H], _I),
    "ln_tracked_id": ([_H, _I], _S),
    "ln_check": ([_H, POINTER(_S), _I], _I),
    "ln_core_size": ([_H], _I),
    "ln_core_id": ([_H, _I], _S),
    "ln_reason": ([_H], _S),
    "ln_dose_time": ([_H, _S, _I, POINTER(_S)], _I),
    "ln_meal_count": ([_H], _I),
    "ln_meal_name": ([_H, _I], _S),
    "ln_meal_value": ([_H, _I], _S),
    "ln_change_count": ([_H], _I),
    "ln_change_name": ([_H, _I], _S),
    "ln_change_value": ([_H, _I], _S),
    "ln_cost_value": ([_H], _S),
    "ln_sexpr": ([_H], _S),
}

_lib: Optional[ctypes.CDLL] = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        from . import build as _build  # imported here so `python -m solver.native.build` runs cleanly

        if _build.stale():
            _build.build()
        z3lib = _build.z3_dir() / "lib"
        if sys.platform == "win32":
            os.add_dll_directory(str(z3lib))  # libz3.dll's own runtime DLLs sit beside it
        ctypes.CDLL(str(z3lib / _build.LIBZ3))  # loaded first, so the core binds to this libz3
        lib = ctypes.CDLL(str(_build.LIBRARY))
        for name, (args, res) in _SIGNATURES.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, res
        _lib = lib
    return _lib


def _b(s: Optional[str]) -> Optional[bytes]:
    return None if s is None else s.encode("utf-8")


def _h(x) -> Optional[bytes]:
    """Hours or an amount, as an exact rational."""
    if x is None:
        return None
    x = Fraction(x)
    return f"{x.numerator}/{x.denominator}".encode()


def _array(items: Iterable[bytes]):
    items = list(items)
    return (c_char_p * len(items))(*items), len(items)


class Native:
    """One encoding, on its own Z3 solver (or optimizer)."""

    def __init__(self, optimize: bool = False, timeout_ms: int = 10_000):
        self._lib = _load()
        self._h = self._lib.ln_new(int(optimize), int(timeout_ms))
        if not self._h:
            raise RuntimeError(self._lib.ln_error().decode("utf-8", "replace"))

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.ln_free(self._h)
            self._h = None

    def _ok(self, rc: int) -> int:
        if rc < 0:
            raise ValueError(self._lib.ln_error().decode("utf-8", "replace"))
        return rc

    # --- structure -------------------------------------------------------------------
    def product(self, name: str, max_doses: int, prescribed: bool, count: Optional[int] = None) -> None:
        self._ok(self._lib.ln_product(self._h, _b(name), max_doses, int(prescribed), -1 if count is None else count))

    def ingredient(self, product: str, ingredient: str, amount) -> None:
        self._ok(self._lib.ln_ingredient(self._h, _b(product), _b(ingredient), _h(amount)))

    def meal_fixed(self, name: str, time) -> None:
        self._ok(self._lib.ln_meal_fixed(self._h, _b(name), _h(time)))

    def meal_var(self, name: str) -> None:
        self._ok(self._lib.ln_meal_var(self._h, _b(name)))

    # --- label rules (rule_id None: hard, untracked) -----------------------------------
    def min_interval(self, rule_id, product, hours) -> None:
        self._ok(self._lib.ln_min_interval(self._h, _b(rule_id), _b(product), _h(hours)))

    def ingredient_interval(self, rule_id, ingredient, hours) -> None:
        self._ok(self._lib.ln_ingredient_interval(self._h, _b(rule_id), _b(ingredient), _h(hours)))

    def separation(self, rule_id, a, b, before, after) -> None:
        self._ok(self._lib.ln_separation(self._h, _b(rule_id), _b(a), _b(b), _h(before), _h(after)))

    def ingredient_ceiling(self, rule_id, ingredient, max_amount) -> None:
        self._ok(self._lib.ln_ingredient_ceiling(self._h, _b(rule_id), _b(ingredient), _h(max_amount)))

    def window(self, rule_id, product, spans) -> None:
        starts, n = _array(_h(s) for s, _ in spans)
        ends, _ = _array(_h(e) for _, e in spans)
        self._ok(self._lib.ln_window(self._h, _b(rule_id), _b(product), starts, ends, n))

    def meal_gap(self, rule_id, product, before, after, meal=None) -> None:
        self._ok(self._lib.ln_meal_gap(self._h, _b(rule_id), _b(product), _h(before), _h(after), _b(meal)))

    def meal_window(self, rule_id, product, meal, earliest, latest) -> None:
        self._ok(self._lib.ln_meal_window(self._h, _b(rule_id), _b(product), _b(meal), _h(earliest), _h(latest)))

    # --- routine and tier-1 terms ------------------------------------------------------
    def sleep(self, wake, awake_len, max_shift=None, cost_per_hour=None) -> None:
        """Doses only while awake (`routine::sleep@<product>`). With max_shift
        and cost_per_hour, waking earlier / bed later become bounded variables."""
        self._ok(self._lib.ln_sleep(self._h, _h(wake), _h(awake_len), _h(max_shift), _h(cost_per_hour)))

    def meal_shift(self, meal, original, max_shift, cost_per_hour) -> None:
        """A variable meal may move up to max_shift hours from `original`."""
        self._ok(self._lib.ln_meal_shift(self._h, _b(meal), _h(original), _h(max_shift), _h(cost_per_hour)))

    def reported_cap(self, product, reported: int, cost_per_dose) -> None:
        """Take at most the reported number of as-needed doses; fewer costs."""
        self._ok(self._lib.ln_reported_cap(self._h, _b(product), int(reported), _h(cost_per_dose)))

    # --- objectives (lexicographic, in the order added) --------------------------------
    def minimize_cost(self) -> None:
        """Total cost of the meal shifts, sleep shift and dropped doses declared so far."""
        self._ok(self._lib.ln_minimize_cost(self._h))

    def minimize_times(self) -> None:
        self._ok(self._lib.ln_minimize_times(self._h))

    def maximize_count(self, product) -> None:
        self._ok(self._lib.ln_maximize_count(self._h, _b(product)))

    def block_superset(self, eps) -> tuple[int, dict[str, Fraction], Fraction]:
        """After a sat tier-1 check: read the model's changes, then exclude
        models that repeat all of them. Returns (how many changes were blocked,
        the size of every possible change, total cost). The sizes are keyed
        `dev::<meal>`, `sleep::wake_earlier`, `sleep::bed_later`,
        `dropped::<product>`. 0 blocked means nothing changed and nothing was added."""
        blocked = self._ok(self._lib.ln_block_superset(self._h, _h(eps)))
        sizes = {self._lib.ln_change_name(self._h, i).decode(): Fraction(self._lib.ln_change_value(self._h, i).decode())
                 for i in range(self._lib.ln_change_count(self._h))}
        return blocked, sizes, Fraction(self._lib.ln_cost_value(self._h).decode())

    # --- solving -------------------------------------------------------------------------
    def tracked(self) -> list[str]:
        return [self._lib.ln_tracked_id(self._h, i).decode() for i in range(self._lib.ln_tracked_count(self._h))]

    def check(self, ids: Iterable[str]) -> str:
        arr, n = _array(_b(i) for i in ids)
        return STATUS[self._ok(self._lib.ln_check(self._h, arr, n))]

    def unsat_core(self) -> list[str]:
        return [self._lib.ln_core_id(self._h, i).decode() for i in range(self._lib.ln_core_size(self._h))]

    def reason_unknown(self) -> str:
        return self._lib.ln_reason(self._h).decode("utf-8", "replace")

    def dose_times(self, product: str, max_doses: int) -> list[Fraction]:
        out, t = [], c_char_p()
        for k in range(max_doses):
            if self._ok(self._lib.ln_dose_time(self._h, _b(product), k, byref(t))):
                out.append(Fraction(t.value.decode()))
        return out

    def sexpr(self) -> str:
        """The whole encoding as SMT-LIB text, objectives included. For debugging."""
        text = self._lib.ln_sexpr(self._h)
        if text is None:
            self._ok(-1)
        return text.decode("utf-8", "replace")

    def meal_times(self) -> dict[str, Fraction]:
        """Every meal's time in the last model (fixed meals included)."""
        out = {}
        for i in range(self._lib.ln_meal_count(self._h)):
            v = self._lib.ln_meal_value(self._h, i)
            if v is None:
                self._ok(-1)
            out[self._lib.ln_meal_name(self._h, i).decode()] = Fraction(v.decode())
        return out
