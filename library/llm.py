"""The ONLY probabilistic component in LINNET: a language-model call.

Everything a model returns is treated as an untrusted draft: it passes through
deterministic checks (library.extract) and a human sign-off before the solver
ever sees it. The model only ever sees public FDA label text, never patient data.

Backends (chosen by the model name):
  claude-*     Anthropic API via the official SDK. Needs ANTHROPIC_API_KEY in
               the environment or in .env (gitignored).
  anything     OpenRouter (e.g. "meta-llama/llama-3.1-8b-instruct"). Needs
  else         OPENROUTER_API_KEY.
  files        (not here) library.extract prepare/ingest: write the prompt to a
               file, read the answer from another. Any model, no key.

Every API call prints its token use and cost, and adds it to SPENT.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("LINNET_MODEL", "claude-haiku-5-5")

# $ per 1M tokens (input, output), Anthropic first-party prices; for the cost printout only.
PRICES = {"claude-haiku-5-5": (0.10, 0.50)}
SPENT = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}

SYSTEM = ("You extract facts from drug labels into JSON. Reply with exactly one JSON object and nothing else: "
          "no prose, no markdown fences.")


def _key(name: str) -> str:
    key = os.environ.get(name)
    if not key and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip().removeprefix("export ").strip() == name:
                key = v.strip().strip('"').strip("'")
    if not key:
        raise RuntimeError(f"{name} not set (environment or .env)")
    return key


def _json_text(text: str) -> str:
    """Strip a markdown fence if the model added one anyway; the caller json.loads it."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def _anthropic(prompt: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY"))
    r = client.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "low"},  # extraction from given text; keeps thinking (and cost) small
        messages=[{"role": "user", "content": prompt}],
    )
    u = r.usage
    pin, pout = PRICES.get(model, (0.0, 0.0))
    usd = (u.input_tokens * pin + u.output_tokens * pout) / 1e6
    SPENT["calls"] += 1
    SPENT["input_tokens"] += u.input_tokens
    SPENT["output_tokens"] += u.output_tokens
    SPENT["usd"] += usd
    print(f"    [{model}] in {u.input_tokens} / out {u.output_tokens} tokens  ~${usd:.4f}  "
          f"(session total ~${SPENT['usd']:.4f})")
    if r.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    if r.stop_reason == "max_tokens":
        raise RuntimeError("answer cut off at max_tokens")
    text = "".join(b.text for b in r.content if b.type == "text")
    return _json_text(text)


def _openrouter(prompt: str, model: str, temperature: float = 0.0) -> str:
    body = json.dumps({
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(OPENROUTER_URL, data=body, headers={
        "Authorization": f"Bearer {_key('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "X-Title": "LINNET",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return _json_text(data["choices"][0]["message"]["content"])


def complete(prompt: str, model: str = DEFAULT_MODEL) -> str:
    if not model:
        raise RuntimeError("no model given (pass --model or set LINNET_MODEL)")
    if model.startswith("claude-"):
        return _anthropic(prompt, model)
    return _openrouter(prompt, model)


def ask_ingredients(product: str, model: str = DEFAULT_MODEL) -> list[str]:
    """LLM fallback for RxNorm misses. Returns lowercase ingredient names."""
    prompt = ("List the active ingredients of this medication product as JSON "
              '{"ingredients": ["name", ...]} using generic names in lowercase, nothing else.\n'
              f"Product: {product}")
    out = json.loads(complete(prompt, model))
    return [str(x).lower() for x in out.get("ingredients", [])]
