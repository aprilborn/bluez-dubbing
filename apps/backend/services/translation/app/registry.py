from __future__ import annotations
from pathlib import Path
from common_schemas.service_utils import Worker, read_model_languages

BASE = Path(__file__).resolve().parents[1]  # service root

# Map multiple models for this service
WORKERS = {
        "deep_translator": Worker(
            venv_python=BASE/"models/deepTranslationModel/.venv/bin/python",
            runner=BASE/"models/deepTranslationModel/runner.py",
            languages=read_model_languages("deep_translator"),
        ),
        "facebook_m2m100": Worker(
            venv_python=BASE/"models/facebook_m2m100Model/.venv/bin/python",
            runner=BASE/"models/facebook_m2m100Model/runner.py",
            languages=read_model_languages("facebook_m2m100"),
        ),
        "llm_polish": Worker(
            venv_python=BASE/"models/llm_polish/.venv/bin/python",
            runner=BASE/"models/llm_polish/runner.py",
            languages=read_model_languages("llm_polish"),
        ),

    }

def _supports(worker, source_language: str | None, target_language: str | None) -> bool:
    if not worker.languages:
        return False
    if target_language and target_language not in worker.languages:
        return False
    # Source language is frequently unknown (auto-detect => None/empty); only
    # constrain on it when it's actually provided. Otherwise a blank source_lang
    # would wrongly reject the requested model and silently fall back to another.
    if source_language and source_language not in worker.languages:
        return False
    return True


def get_worker(model_key: str | None, source_language: str | None, target_language: str) -> tuple[Path, Path, str]:
    # Prefer the explicitly requested model when it can handle the language(s).
    selected_key = None
    if model_key in WORKERS and _supports(WORKERS[model_key], source_language, target_language):
        selected_key = model_key

    # Otherwise, pick the first model that supports the language(s).
    if selected_key is None:
        for k, w in WORKERS.items():
            if _supports(w, source_language, target_language):
                selected_key = k
                break

    # Fallback to the first model if none declare support (language unchanged)
    if selected_key is None:
        print(f"No model found supporting language={source_language} - {target_language}. Defaulting to first model.")
        selected_key = next(iter(WORKERS))

    w = WORKERS[selected_key]
    return w.venv_python, w.runner, selected_key
