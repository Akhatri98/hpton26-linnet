"""Command line: python -m linnet data/patients/<patient>.json"""

import argparse
import sys
from pathlib import Path

from library.load import load

from .patient import load_patient
from .render import render
from .tiers import run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="linnet", description=__doc__)
    ap.add_argument("patient", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--timeout-ms", type=int, default=10_000)
    ap.add_argument("--compare-baseline", action="store_true",
                    help="also run a greedy heuristic scheduler and compare verdicts")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    lib = load(args.data)
    patient = load_patient(args.patient)
    report = run(lib, patient, timeout_ms=args.timeout_ms)
    print(render(report, lib))
    if args.compare_baseline:
        from .baseline import run_greedy
        from .render import render_baseline
        print(render_baseline(run_greedy(lib, patient), report, lib))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
