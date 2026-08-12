"""The llm_retry state cache — one JSON per asset, the pass's memory of its two LLM calls.

An LLM is not deterministic, but the pass contract requires ``run()`` to be: the cache is what
squares that circle, exactly as ``vlm_regions`` does it. The RAW validated LLM response is stored
the first (and only) time each round is called; every later run re-derives its candidates from the
cached response, a pure function of (cached answer, asset geometry). Consequences:

* ``--check-idempotent`` holds despite a nondeterministic annotator;
* transport failure RAISES and is NEVER cached — "the LLM proposed nothing" must not become a
  stored finding (mirrors ``vlm_regions.prompt.ask_regions``);
* the cache key is (mesh_sha1, prompt version, pass version): a changed mesh or prompt invalidates
  the whole cache, which RE-ARMS the retry — the contract's "re-check if the mesh changes"
  (docs/trajPipeline/llm-retry.md). Deleting the file re-arms it by hand.

Layout: ``assets/objects/grasps/_passes/llm_retry/_cache/<asset>.json``. The ``_cache`` directory
sits INSIDE the pass's sidecar directory but is invisible to the harness — ``base.sidecar_assets``
globs ``*.json`` non-recursively, so nothing here can be mistaken for a sidecar.

Per round the cache holds: the request digest (provenance — what was actually sent), the model
that answered, the raw validated answer, and the authoring-failure list derived from it at call
time (kept so the round-B feedback can report drops without recomputing them from a response that
might have been superseded). Round B additionally records the feedback digest its request was
computed from.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..base import PASSES_DIR

CACHE_VERSION = 1
CACHE_DIR = PASSES_DIR / "llm_retry" / "_cache"


def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def cache_key(mesh_sha1: str, prompt_version: int, pass_version: int) -> str:
    """What makes a stored response still valid. The MODEL is deliberately absent (as in
    ``vlm_regions.regions.cache_key``): a new model on an unchanged mesh is not a reason to spend
    a bounded retry round."""
    return f"{CACHE_VERSION}|{mesh_sha1}|{prompt_version}|{pass_version}"


def load_state(name: str, mesh_sha1: str, prompt_version: int, pass_version: int) -> dict | None:
    """The cached state for one asset, or None when absent or STALE.

    A stale cache (different mesh, prompt, or pass version) reads as "no cache": the next run
    re-opens round A, which is the contract's re-arm-on-change behavior. The stale file is left in
    place until that run overwrites it, so nothing is lost if the staleness was itself a mistake."""
    p = cache_path(name)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("cache_key") != cache_key(mesh_sha1, prompt_version, pass_version):
        return None
    if not isinstance(data.get("rounds"), dict):
        return None
    return data


def new_state(name: str, mesh_sha1: str, prompt_version: int, pass_version: int) -> dict:
    return {
        "_comment": [
            f"llm_retry state cache for catalog object {name!r} — the raw LLM responses the pass's",
            "candidates are deterministically re-derived from (docs/trajPipeline/llm-retry.md).",
            "GENERATED. Deleting this file RE-ARMS the retry (both rounds) for this asset.",
        ],
        "cache_version": CACHE_VERSION,
        "cache_key": cache_key(mesh_sha1, prompt_version, pass_version),
        "asset": name,
        "mesh_sha1": mesh_sha1,
        "prompt_version": int(prompt_version),
        "pass_version": int(pass_version),
        "rounds": {},
    }


def save_state(state: dict) -> Path:
    out = cache_path(state["asset"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    return out


def request_digest(content: list) -> str:
    """Stable digest of one multimodal request's content blocks (text + base64 images).

    Provenance, not a lookup key: it records what was actually sent so a cached answer can be
    audited against the request that produced it."""
    h = hashlib.sha1()
    for block in content:
        if block.get("type") == "text":
            h.update(b"text|")
            h.update(block["text"].encode())
        elif block.get("type") == "image":
            h.update(b"image|")
            h.update(block["source"]["data"].encode())
        h.update(b"\n")
    return h.hexdigest()
