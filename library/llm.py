"""The ONLY probabilistic component in LINNET: a language-model call.

Everything a model returns is treated as an untrusted draft: it passes through
deterministic checks (library.extract) and a human sign-off before the solver
ever sees it.

Backends:
  openrouter   HTTPS call to OpenRouter. Needs OPENROUTER_API_KEY in the
               environment or in a .env file (gitignored). Model is a setting.
  files        Writes the prompt to a file and reads the answer from another.
               Used to run any external model by hand, or a subagent, without
               an API key.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("LINNET_MODEL", "")


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (environment or .env)")
    return key


def complete(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    if not model:
        raise RuntimeError("no model given (pass --model or set LINNET_MODEL)")
    body = json.dumps({
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(OPENROUTER_URL, data=body, headers={
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "X-Title": "LINNET",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"]


def ask_ingredients(product: str, model: str = DEFAULT_MODEL) -> list[str]:
    """LLM fallback for RxNorm misses. Returns lowercase ingredient names."""
    prompt = ("List the active ingredients of this medication product as JSON "
              '{"ingredients": ["name", ...]} using generic names in lowercase, nothing else.\n'
              f"Product: {product}")
    out = json.loads(complete(prompt, model))
    return [str(x).lower() for x in out.get("ingredients", [])]
