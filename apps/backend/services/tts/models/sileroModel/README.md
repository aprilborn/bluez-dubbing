# Silero TTS worker

Wraps [`silero-tts`](https://pypi.org/project/silero-tts/) as a TTS worker for the
Bluez dubbing pipeline. Reads a `TTSRequest` JSON on stdin and writes a
`TTSResponse` JSON on stdout (same contract as the other TTS models).

Silero is a fixed-speaker, non-cloning model: each supported language ships a set
of built-in speakers. Diarized `speaker_id`s are mapped deterministically onto the
model's available speakers so a given speaker keeps a consistent voice across
segments. Model weights (`.pt`) are downloaded on first use into the installed
`silero_tts/silero_models/` directory and cached thereafter.

Configuration lives in `libs/common-schemas/config/silero.yaml`.
