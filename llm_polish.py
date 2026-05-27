"""
LLM text polishing — post-process Whisper output via GitHub Models API.

Uses the OpenAI-compatible chat completions endpoint at
https://models.inference.ai.azure.com with a GitHub token.

The LLM fixes transcription artefacts (wrong words, grammar, punctuation)
WITHOUT changing factual content, names, numbers, or meaning.
"""
import json
import logging
import os
import subprocess
import urllib.error
import urllib.request
from typing import NamedTuple

log = logging.getLogger("freewispr")

API_URL = "https://models.inference.ai.azure.com/chat/completions"

# Models available via GitHub Models (tested and working)
AVAILABLE_MODELS = {
    "gpt-4.1-nano":  "GPT-4.1 Nano — snabbast (~1s)",
    "gpt-4.1-mini":  "GPT-4.1 Mini — bra balans (~1.3s)",
    "gpt-4o":        "GPT-4o — hogst kvalitet (~1.6s)",
    "gpt-4o-mini":   "GPT-4o Mini — aldre, snabb (~1.5s)",
}

DEFAULT_MODEL = "gpt-4.1-nano"


def resolve_api_key(api_key: str = "") -> str:
    """Return explicit key, env token, or GitHub CLI auth token.

    This lets the desktop app use the same machine-level GitHub auth that
    developer tools such as opencode/gh already use, without forcing the user
    to find and paste a separate token into settings.
    """
    explicit = (api_key or "").strip()
    if explicit:
        return explicit
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _text_meta(text: str) -> str:
    words = len(text.split())
    return f"chars={len(text)}, words={words}"

_SYSTEM_PROMPT = (
    "Du ar en svensk textkorrigerare for dikterad text (speech-to-text). "
    "Korrigera BARA uppenbara transkriptionsfel: felhorda ord, grammatik, "
    "interpunktion och stavning. "
    "ANDRA INTE: innehall, artal, namn, siffror, fakta eller meningens betydelse. "
    "Om texten redan ar korrekt, returnera den EXAKT som den ar. "
    "Returnera BARA den korrigerade texten, inget annat."
)


class PolishResult(NamedTuple):
    """Result of a polish operation."""
    text: str
    model: str
    latency_ms: int
    changed: bool


def _call_api(api_key: str, model: str, user_text: str,
              timeout_sec: float = 8.0) -> dict:
    """Make a raw API call. Returns the parsed JSON response."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0,
        "max_tokens": max(200, len(user_text) * 2),
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def polish(text: str, api_key: str, model: str = DEFAULT_MODEL) -> PolishResult:
    """Send text through LLM for polishing. Returns PolishResult.

    On any error, returns the original text unchanged (never blocks dictation).
    """
    import time

    text = text.strip()
    resolved_key = resolve_api_key(api_key)
    if not text or not resolved_key:
        return PolishResult(text=text, model=model, latency_ms=0, changed=False)

    t0 = time.perf_counter()
    try:
        data = _call_api(resolved_key, model, text)
        result = data["choices"][0]["message"]["content"].strip()
        latency = int((time.perf_counter() - t0) * 1000)

        # Safety: if LLM returns something wildly different length-wise,
        # it probably hallucinated — use original.
        #
        # Length floor: 30 % of the original IS too strict on short
        # utterances ("Klockan ringer" -> "Ringer." is a legitimate
        # 3->1 word polish). Allow shrink down to either 30 % of
        # the original OR an absolute floor of 20 chars, whichever
        # is more permissive.
        too_long = len(result) > len(text) * 3
        too_short = len(result) < len(text) * 0.3 and len(result) < 20
        if too_long or too_short:
            log.warning("LLM-svar avviker for mycket i langd (%d vs %d), "
                        "anvander original", len(result), len(text))
            return PolishResult(text=text, model=model, latency_ms=latency,
                                changed=False)

        changed = result != text
        if changed:
            log.info("LLM-polerad (%s, %dms, in=%s, out=%s)",
                     model, latency, _text_meta(text), _text_meta(result))
        else:
            log.info("LLM: ingen andring behoves (%s, %dms)", model, latency)

        return PolishResult(text=result, model=model, latency_ms=latency,
                            changed=changed)

    except urllib.error.HTTPError as e:
        latency = int((time.perf_counter() - t0) * 1000)
        try:
            e.read()
        except Exception:
            pass
        log.warning("LLM HTTP %d (%dms, model=%s)", e.code, latency, model)
        return PolishResult(text=text, model=model, latency_ms=latency,
                            changed=False)

    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        log.warning("LLM-fel (%dms): %s", latency, e)
        return PolishResult(text=text, model=model, latency_ms=latency,
                            changed=False)


def test_connection(api_key: str, model: str = DEFAULT_MODEL) -> tuple[bool, str]:
    """Test the API connection. Returns (success, message).

    Sends a known Swedish sentence and verifies the response.
    """
    import time

    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        return False, "Ingen GitHub-auth hittades (ange nyckel, GITHUB_TOKEN/GH_TOKEN eller logga in med gh auth login)"

    test_input = "Det har ar ett test av dikteringsfunktionen."

    t0 = time.perf_counter()
    try:
        data = _call_api(resolved_key, model, test_input, timeout_sec=10.0)
        latency = int((time.perf_counter() - t0) * 1000)

        result = data["choices"][0]["message"]["content"].strip()
        used_model = data.get("model", model)
        tokens_in = data.get("usage", {}).get("prompt_tokens", "?")
        tokens_out = data.get("usage", {}).get("completion_tokens", "?")

        return True, (
            f"OK! Modell: {used_model}\n"
            f"Svar: \"{result}\"\n"
            f"Tid: {latency}ms | Tokens: {tokens_in} in, {tokens_out} ut"
        )

    except urllib.error.HTTPError as e:
        latency = int((time.perf_counter() - t0) * 1000)
        if e.code == 401:
            return False, f"Ogiltig API-nyckel (HTTP 401, {latency}ms)"
        if e.code == 403:
            return False, f"Atkomst nekad (HTTP 403, {latency}ms)"
        if e.code == 404:
            return False, f"Modell '{model}' finns inte (HTTP 404, {latency}ms)"
        if e.code == 429:
            return False, f"Rate limit — for manga anrop (HTTP 429, {latency}ms)"
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        return False, f"HTTP {e.code} ({latency}ms): {body}"

    except urllib.error.URLError as e:
        latency = int((time.perf_counter() - t0) * 1000)
        return False, f"Natverksfel ({latency}ms): {e.reason}"

    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        return False, f"Fel ({latency}ms): {e}"
