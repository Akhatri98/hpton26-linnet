"""Where sign-offs are written.  [deterministic]

The web UI and the CLI both go through a Store. LocalJsonStore edits the
JSON files in data/ (so every approval shows up in `git diff`). A hosted
store (e.g. Supabase) would implement the same two methods.

Only library facts live here. Patient data is never stored: checks run in
memory and the result goes back to the browser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .review import _get, _items


class Store(Protocol):
    def items(self) -> list[dict]: ...
    def set_verified(self, item_id: str, value: bool) -> dict: ...


class LocalJsonStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def items(self) -> list[dict]:
        out = []
        for iid, path, ptr, ok, lines in _items(self.root):
            out.append({"id": iid, "kind": iid.split(":", 1)[0] if ":" in iid else "rule",
                        "verified": bool(ok), "file": path.relative_to(self.root).as_posix(), "lines": lines,
                        "record": _get(json.loads(path.read_text(encoding="utf-8")), ptr)})
        return out

    def set_verified(self, item_id: str, value: bool) -> dict:
        for iid, path, ptr, _, _ in _items(self.root):
            if iid == item_id:
                doc = json.loads(path.read_text(encoding="utf-8"))
                _get(doc, ptr)["verified"] = bool(value)
                path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return {"id": iid, "verified": bool(value), "file": path.relative_to(self.root).as_posix()}
        raise KeyError(item_id)
