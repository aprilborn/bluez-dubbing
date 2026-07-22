from __future__ import annotations
import contextlib
import json
import logging
import random
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from silero_tts.silero_tts import SileroTTS

from common_schemas.models import SegmentAudioOut, TTSRequest, TTSResponse
from common_schemas.service_utils import get_service_logger

# Silero uses 'ua' for Ukrainian; the rest of the pipeline uses ISO 'uk'.
LANG_ALIASES = {"uk": "ua"}

# One SileroTTS instance per (silero_lang, model_id, sample_rate). Speaker is set
# per-segment on the cached instance, so it is not part of the cache key.
_MODEL_CACHE: Dict[Tuple[str, str, int], SileroTTS] = {}
_MODEL_LOCK = threading.Lock()


def _resolve_device(pref: str | None) -> str:
    if pref in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return pref


def _silero_lang(lang: str) -> str:
    return LANG_ALIASES.get(lang, lang)


def _resolve_model_id(system_lang: str, silero_lang: str, models_cfg: dict) -> str:
    return models_cfg.get(system_lang) or SileroTTS.get_latest_model(silero_lang)


def _resolve_sample_rate(silero_lang: str, model_id: str, requested: int, logger) -> int:
    available = SileroTTS.get_available_sample_rates_static(silero_lang, model_id)
    if requested in available:
        return requested
    fallback = max(available)
    logger.warning(
        "Sample rate %s unsupported for %s/%s; using %s instead.",
        requested,
        silero_lang,
        model_id,
        fallback,
    )
    return fallback


def _get_tts(
    silero_lang: str,
    model_id: str,
    sample_rate: int,
    device: str,
    put_accent: bool,
    put_yo: bool,
    num_threads: int,
    logger,
) -> SileroTTS:
    key = (silero_lang, model_id, sample_rate)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached:
            logger.debug("Using cached silero model=%s lang=%s sr=%s.", model_id, silero_lang, sample_rate)
            return cached

        load_start = time.perf_counter()
        tts = SileroTTS(
            model_id=model_id,
            language=silero_lang,
            sample_rate=sample_rate,
            device=device,
            put_accent=put_accent,
            put_yo=put_yo,
            num_threads=num_threads,
        )
        _MODEL_CACHE[key] = tts
        logger.info(
            "Loaded silero model=%s lang=%s sr=%s device=%s in %.2fs.",
            model_id,
            silero_lang,
            sample_rate,
            device,
            time.perf_counter() - load_start,
        )
        return tts


def _pick_speaker(tts: SileroTTS, speaker_id: str | None, system_lang: str, default_speaker: str | None) -> str:
    speakers = [s for s in tts.get_available_speakers() if s != "random"]
    if not speakers:  # some models expose only 'random'
        speakers = tts.get_available_speakers()

    base_default = default_speaker if default_speaker in speakers else speakers[0]
    if not speaker_id:
        return base_default

    # Stable per-speaker voice so a diarized speaker keeps one voice across segments.
    rng = random.Random(f"{speaker_id}-{system_lang}")
    return rng.choice(speakers)


def _synthesize(req: TTSRequest) -> TTSResponse:
    params = req.extra or {}
    general_cfg = params.get("general", {})
    models_cfg = params.get("models", {}) or {}
    speakers_cfg = params.get("speakers", {}) or {}
    # Optional single voice forced for the whole run (set from the UI speaker
    # dropdown). Applied only to languages whose model actually has that speaker.
    forced_speaker = params.get("forced_speaker") or None

    log_level = str(params.get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level, logging.INFO)
    logger = get_service_logger("tts.silero", log_level)

    device = _resolve_device(general_cfg.get("device", "auto"))
    requested_sr = int(general_cfg.get("sample_rate", 48000))
    put_accent = bool(general_cfg.get("put_accent", True))
    put_yo = bool(general_cfg.get("put_yo", True))
    num_threads = int(general_cfg.get("num_threads", 6))

    run_start = time.perf_counter()
    workspace_path = Path(req.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    workspace_root = workspace_path.resolve()

    out = TTSResponse()
    logger.info(
        "Starting silero synthesis segments=%d workspace=%s device=%s sr=%s",
        len(req.segments or []),
        req.workspace,
        device,
        requested_sr,
    )

    # Keep one voice per (speaker_id, lang) across the whole run.
    speaker_voice_map: Dict[Tuple[str | None, str], str] = {}

    for i, segment in enumerate(req.segments):
        seg_start = time.perf_counter()

        system_lang = segment.lang or req.language
        if not system_lang:
            raise RuntimeError(f"segment {i} has no language and request has no default language")
        silero_lang = _silero_lang(system_lang)

        model_id = _resolve_model_id(system_lang, silero_lang, models_cfg)
        sample_rate = _resolve_sample_rate(silero_lang, model_id, requested_sr, logger)
        tts = _get_tts(silero_lang, model_id, sample_rate, device, put_accent, put_yo, num_threads, logger)

        key = (segment.speaker_id, system_lang)
        speaker = speaker_voice_map.get(key)
        if not speaker:
            if forced_speaker and forced_speaker in tts.get_available_speakers():
                speaker = forced_speaker
            else:
                if forced_speaker:
                    logger.warning(
                        "Forced speaker '%s' not available for %s/%s; using auto selection.",
                        forced_speaker,
                        silero_lang,
                        model_id,
                    )
                speaker = _pick_speaker(tts, segment.speaker_id, system_lang, speakers_cfg.get(system_lang))
            speaker_voice_map[key] = speaker
        tts.speaker = speaker

        # Decide output path (support in-place re-synthesis for the review stage).
        if segment.legacy_audio_path:
            output_file = Path(segment.legacy_audio_path)
            output_file = output_file if output_file.is_absolute() else (workspace_path / output_file)
            output_file = output_file.resolve()
            try:
                output_file.relative_to(workspace_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"legacy_audio_path must reside inside workspace: {segment.legacy_audio_path}"
                ) from exc
            output_file = output_file.with_suffix(".wav")
        else:
            identifier = segment.segment_id or f"seg-{i}"
            output_file = workspace_path / f"{identifier}.wav"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        tts.tts(segment.text, str(output_file))

        out.segments.append(
            SegmentAudioOut(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                audio_prompt_url=segment.audio_prompt_url,
                audio_url=str(output_file),
                speaker_id=segment.speaker_id,
                lang=segment.lang,
                sample_rate=sample_rate,
                segment_id=segment.segment_id,
            )
        )
        logger.info(
            "Generated segment %d lang=%s model=%s speaker=%s duration=%.2fs",
            i,
            silero_lang,
            model_id,
            speaker,
            time.perf_counter() - seg_start,
        )

    logger.info(
        "Completed silero synthesis in %.2fs (segments=%d).",
        time.perf_counter() - run_start,
        len(out.segments),
    )
    return out


def _run_once():
    req = TTSRequest(**json.loads(sys.stdin.read()))
    with contextlib.redirect_stdout(sys.stderr):
        out = _synthesize(req)
    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    _run_once()
