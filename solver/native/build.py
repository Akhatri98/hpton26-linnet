"""Build the native solver core (linnet_solver.cpp) into a shared library.

    python -m solver.native.build

Needs a C++17 compiler: g++ or clang++ on PATH, $CXX, or MSYS2's g++ on
Windows. Z3's headers and libz3 come from the `z3-solver` wheel already in
requirements.txt, so there is nothing else to install. `solver.native` calls
this by itself when the library is missing or older than the source.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "linnet_solver.cpp"
LIBRARY = HERE / ("linnet_solver" + {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so"))
LIBZ3 = {"win32": "libz3.dll", "darwin": "libz3.dylib"}.get(sys.platform, "libz3.so")


def z3_dir() -> Path:
    """The installed z3-solver wheel, located without importing it."""
    spec = importlib.util.find_spec("z3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("z3-solver is not installed (pip install -r requirements.txt)")
    return Path(next(iter(spec.submodule_search_locations)))


def _compiler() -> str:
    candidates = [os.environ.get("CXX"), shutil.which("g++"), shutil.which("clang++"),
                  r"C:\msys64\ucrt64\bin\g++.exe", r"C:\msys64\mingw64\bin\g++.exe"]
    for c in candidates:
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    raise RuntimeError("no C++ compiler found: install g++ (MSYS2 on Windows) or set CXX")


def stale() -> bool:
    return not LIBRARY.exists() or LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime


def build() -> Path:
    z3 = z3_dir()
    cxx = _compiler()
    tmp = LIBRARY.with_name(f"{LIBRARY.stem}.{os.getpid()}{LIBRARY.suffix}")
    cmd = [cxx, "-std=c++17", "-O2", "-shared", "-fPIC", "-I", str(z3 / "include"), str(SOURCE), "-o", str(tmp)]
    if sys.platform == "win32":
        # MinGW links straight against the DLL; -static folds its own runtime in,
        # so the result depends on libz3.dll and nothing from MSYS2.
        cmd += [str(z3 / "lib" / LIBZ3), "-static"]
    else:
        cmd += ["-L", str(z3 / "lib"), "-lz3", f"-Wl,-rpath,{z3 / 'lib'}"]
    env = dict(os.environ)
    if Path(cxx).parent != Path("."):  # g++ finds its own back end through PATH
        env["PATH"] = str(Path(cxx).parent) + os.pathsep + env.get("PATH", "")
    done = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"building {LIBRARY.name} failed:\n{done.stderr}")
    os.replace(tmp, LIBRARY)  # atomic: another process never sees a half-written library
    return LIBRARY


if __name__ == "__main__":
    print(build())
