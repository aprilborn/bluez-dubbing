from __future__ import annotations

import contextlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from shutil import which
from typing import Dict, List, Tuple

import httpx

from common_schemas.models import ASRResponse, Segment, TranslateRequest
from common_schemas.service_utils import get_service_logger

# The draft translation is produced by the sibling M2M-100 plugin. We shell out
# to its runner exactly the way the translation service invokes any worker
# (stdin JSON -> stdout JSON) so we don't have to duplicate torch/transformers
# into this plugin's virtualenv.
_HERE = Path(__file__).resolve().parent
_MODELS_DIR = _HERE.parent
_UV_BIN = which("uv")

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
# NOTE: gemma3:12b-it-qat was requested but is not present on this host; the
# installed equivalent is gemma4:12b-it-qat. Override via config/llm_polish.yaml.
_DEFAULT_OLLAMA_MODEL = "gemma4:12b-it-qat"

_SYSTEM_PROMPT = (
    "You are a professional subtitle localization editor. You receive a full "
    "list of subtitle segments for one video, each with the original text and a "
    "rough machine-translated draft. You improve the draft translations as a "
    "single conversation: fix cross-segment pronoun and reference consistency, "
    "remove overly literal or awkward phrasing, and make every line natural and "
    "idiomatic in the target language while preserving the exact meaning. You "
    "MUST NOT merge, split, reorder, add or drop segments. Return one improved "
    "line per input segment, keyed by its id. Respond with JSON only."
)


def _draft_worker_cmd(runner: Path) -> Tuple[str, ...]:
    """Mirror runner_api._format_cmd: prefer `uv run`, fall back to the venv."""
    if _UV_BIN:
        return (_UV_BIN, "run", runner.name)
    venv_python = runner.parent / ".venv" / "bin" / "python"
    return (str(venv_python), str(runner))


def _get_draft(req: TranslateRequest, logger: logging.Logger) -> ASRResponse:
    extra = req.extra or {}
    draft_model_dir = extra.get("draft_model_dir", "facebook_m2m100Model")
    runner = _MODELS_DIR / draft_model_dir / "runner.py"
    if not runner.exists():
        raise RuntimeError(f"draft model runner not found: {runner}")

    # Build a clean request for the draft worker: same segments/langs, but only
    # the config keys that worker understands (avoid leaking our Ollama params).
    draft_extra: Dict[str, object] = {"log_level": extra.get("log_level", "INFO")}
    if extra.get("draft_model_name"):
        draft_extra["model_name"] = extra["draft_model_name"]
    draft_req = TranslateRequest(
        segments=req.segments,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        extra=draft_extra,
    )

    cmd = list(_draft_worker_cmd(runner))
    logger.info("Requesting draft translation from %s", draft_model_dir)
    draft_start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=draft_req.model_dump_json(),
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        cwd=runner.parent,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"draft worker failed ({proc.returncode}); see runner stderr for details.")
    out = (proc.stdout or "").strip()
    if not out:
        raise RuntimeError("draft worker produced no output")
    try:
        draft = ASRResponse(**json.loads(out))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON from draft worker: {exc}\nraw:\n{out}")
    logger.info(
        "Draft translation ready segments=%d in %.2fs",
        len(draft.segments),
        time.perf_counter() - draft_start,
    )
    return draft


def _build_llm_payload(req: TranslateRequest, draft: ASRResponse) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for idx, (src, drafted) in enumerate(zip(req.segments or [], draft.segments)):
        items.append(
            {
                "id": idx,
                "start": drafted.start,
                "end": drafted.end,
                "original": src.text,
                "draft": drafted.text,
            }
        )
    return items


# A strict JSON schema is far more reliable than format="json": it forces the
# model to emit well-formed {segments:[{id,text}]} instead of stopping early,
# truncating, or returning an empty string on longer batches.
_POLISH_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
            },
        }
    },
    "required": ["segments"],
}


def _chunks(items: List[Dict[str, object]], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _polish_chunk(
    chunk: List[Dict[str, object]],
    context_pairs: List[Dict[str, object]],
    req: TranslateRequest,
    http: httpx.Client,
    model: str,
    options: Dict[str, object],
    keep_alive: object,
    timeout: float,
) -> Dict[int, str]:
    context_block = ""
    if context_pairs:
        context_block = (
            "For continuity, here are the already-finalized translations of the "
            "immediately preceding segments. Do NOT re-output them; use them only "
            "to keep pronouns, references and terminology consistent:\n"
            f"{json.dumps(context_pairs, ensure_ascii=False)}\n\n"
        )

    user_prompt = (
        f"Target language (ISO code): {req.target_lang}\n"
        f"Source language (ISO code): {req.source_lang or 'auto'}\n\n"
        f"{context_block}"
        "Improve the following subtitle segments. Each has an integer `id`, the "
        "`original` source text and a machine-translated `draft`:\n\n"
        f"{json.dumps(chunk, ensure_ascii=False)}\n\n"
        "Return exactly one improved entry per id shown above, reusing the same ids.\n"
        "Rules:\n"
        "- Only improve the translated text; keep meaning faithful to `original`.\n"
        "- Fix pronouns/references so they stay consistent across segments.\n"
        "- Prefer natural, idiomatic phrasing over literal word-for-word.\n"
        "- Never merge, split, add or drop segments."
    )

    body = {
        "model": model,
        "stream": False,
        "format": _POLISH_SCHEMA,
        "keep_alive": keep_alive,
        "options": options,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = http.post("/api/chat", json=body, timeout=timeout)
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama returned an empty response")

    parsed = json.loads(content)  # schema-constrained, so this is valid JSON
    segments = parsed.get("segments") if isinstance(parsed, dict) else None
    if not isinstance(segments, list):
        raise RuntimeError(f"Ollama JSON missing 'segments' list; got {parsed!r}")

    out: Dict[int, str] = {}
    for entry in segments:
        if isinstance(entry, dict) and "id" in entry and "text" in entry:
            out[int(entry["id"])] = str(entry["text"])
    return out


def _call_ollama(items: List[Dict[str, object]], req: TranslateRequest, logger: logging.Logger) -> Dict[int, str]:
    """Polish every segment, chunk by chunk, always returning a text for each id.

    A 12B local model can't reliably emit one giant JSON array, so we split the
    batch into small chunks (passing recent finalized lines as read-only context
    to keep cross-segment consistency). Any segment the model omits or a whole
    chunk that errors out falls back to its draft translation, so segment count,
    order and timestamps are never disturbed — the pipeline degrades instead of
    hard-failing. Set strict_polish=true to raise on any fallback instead.
    """
    extra = req.extra or {}
    base_url = str(extra.get("ollama_url", _DEFAULT_OLLAMA_URL)).rstrip("/")
    model = extra.get("ollama_model", _DEFAULT_OLLAMA_MODEL)
    timeout = float(extra.get("ollama_timeout", 600))
    keep_alive = extra.get("ollama_keep_alive", 0)
    batch_size = int(extra.get("batch_size", 20)) or len(items)
    context_size = int(extra.get("context_size", 4))
    strict = bool(extra.get("strict_polish", False))
    options = {
        "temperature": float(extra.get("temperature", 0.2)),
        "num_ctx": int(extra.get("num_ctx", 8192)),
        "num_predict": int(extra.get("num_predict", 8192)),
    }

    logger.info(
        "Polishing %d segments via Ollama model=%s (batch_size=%d)",
        len(items),
        model,
        batch_size,
    )

    polished: Dict[int, str] = {}
    finalized_context: List[Dict[str, object]] = []
    fallbacks = 0

    with httpx.Client(base_url=base_url) as http:
        for chunk in _chunks(items, batch_size):
            ctx = finalized_context[-context_size:] if context_size else []
            try:
                got = _polish_chunk(chunk, ctx, req, http, model, options, keep_alive, timeout)
            except Exception as exc:  # noqa: BLE001 - degrade to draft, never fail the job
                logger.warning(
                    "polish chunk (ids %d-%d) failed (%s: %s); using draft for these.",
                    chunk[0]["id"],
                    chunk[-1]["id"],
                    type(exc).__name__,
                    exc,
                )
                got = {}

            for item in chunk:
                cid = int(item["id"])
                text = got.get(cid)
                if not text or not text.strip():
                    text = str(item["draft"])
                    fallbacks += 1
                    logger.warning("segment id=%d missing/empty from LLM; using draft.", cid)
                polished[cid] = text
                finalized_context.append({"original": item["original"], "text": text})

    if fallbacks:
        msg = f"{fallbacks}/{len(items)} segments fell back to the draft translation."
        if strict:
            raise RuntimeError("strict_polish enabled: " + msg)
        logger.warning(msg)

    return polished


def _translate(req: TranslateRequest, logger: logging.Logger) -> ASRResponse:
    run_start = time.perf_counter()
    segments = req.segments or []
    logger.info(
        "Starting llm_polish run segments=%d source=%s target=%s",
        len(segments),
        req.source_lang,
        req.target_lang,
    )

    out = ASRResponse(language=req.target_lang)
    if not segments:
        logger.info("No segments to translate; returning empty response.")
        return out

    draft = _get_draft(req, logger)

    # Hard invariant: the draft worker must return exactly the input segments.
    if len(draft.segments) != len(segments):
        raise RuntimeError(
            f"draft segment count mismatch: expected {len(segments)}, got {len(draft.segments)}"
        )

    items = _build_llm_payload(req, draft)
    polished = _call_ollama(items, req, logger)

    # CRITICAL: exactly one output line per input segment. Count, order,
    # timestamps, speaker and segment_id are taken verbatim from the ORIGINAL
    # request (the source of truth) — never from the LLM or the draft worker.
    # `polished` always covers every id (draft fallback fills any gap), but we
    # defensively fall back to the draft text here too so a segment can never
    # desync from its timestamp.
    for idx, src in enumerate(segments):
        text = polished.get(idx) or draft.segments[idx].text
        out.segments.append(
            Segment(
                start=src.start,
                end=src.end,
                text=text,
                speaker_id=src.speaker_id,
                lang=req.target_lang,
                segment_id=src.segment_id,
            )
        )

    logger.info(
        "Completed llm_polish run in %.2fs (segments=%d).",
        time.perf_counter() - run_start,
        len(out.segments),
    )
    return out


def _run_once() -> None:
    req = TranslateRequest(**json.loads(sys.stdin.read()))
    log_level = (req.extra or {}).get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level, logging.INFO)
    logger = get_service_logger("translation.llm_polish", log_level)

    with contextlib.redirect_stdout(sys.stderr):
        out = _translate(req, logger)

    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    _run_once()
