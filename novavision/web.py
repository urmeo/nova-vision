from __future__ import annotations

import base64
import logging
import os
import random
import re
import threading
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from novavision.affect.analyzer import EmotionAnalysis
from novavision.pipeline import NovaVision, Settings, build_pipeline
from novavision.prompting import STYLE_PRESETS
from novavision.serving import (
    ConcurrencyGuard,
    RateLimiter,
    env_int,
    is_loopback_host,
    public_enabled,
    resolve_host,
    token_ok,
)

logger = logging.getLogger(__name__)

MIN_TEXT, MAX_TEXT = 3, 2000
_CONTROL_CHARS = re.compile(r"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]")
_ROOT = Path(__file__).resolve().parents[1]


def _analysis_payload(analysis: EmotionAnalysis) -> dict:
    ranked = sorted(analysis.scores.items(), key=lambda item: item[1], reverse=True)
    return {
        "primary_emotion": analysis.primary,
        "confidence": round(analysis.confidence * 100, 1),
        "valence": analysis.valence,
        "arousal": analysis.arousal,
        "emotions": [{"name": name, "score": round(score * 100, 1)} for name, score in ranked],
    }


def _valid_text(data) -> tuple[str, str | None]:
    raw = data.get("text", "") if isinstance(data, dict) else ""
    if not isinstance(raw, str):
        return "", f"Text must be {MIN_TEXT}-{MAX_TEXT} characters."

    text = _CONTROL_CHARS.sub(" ", raw).strip()
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return "", "Text must contain valid Unicode characters."
    if not MIN_TEXT <= len(text) <= MAX_TEXT:
        return text, f"Text must be {MIN_TEXT}-{MAX_TEXT} characters."
    return text, None


def _generation_options(data: dict) -> tuple[str, int]:
    style = data.get("style", "artistic")
    if not isinstance(style, str) or style.lower() not in STYLE_PRESETS:
        raise ValueError(f"Unknown style. Choose one of: {', '.join(STYLE_PRESETS)}.")

    raw_seed = data.get("seed")
    if raw_seed is None:
        seed = random.randint(0, 2**31 - 1)
    else:
        if isinstance(raw_seed, bool) or (
            isinstance(raw_seed, float) and not raw_seed.is_integer()
        ):
            raise ValueError("seed must be an integer.")
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError):
            raise ValueError("seed must be an integer.") from None
    if not -(2**63) <= seed < 2**63:
        raise ValueError("seed must fit in a signed 64-bit integer.")
    return style.lower(), seed


def _bearer_token() -> str | None:
    authorization = request.headers.get("Authorization", "")
    scheme, separator, credentials = authorization.partition(" ")
    if separator and scheme.lower() == "bearer":
        return credentials.strip()
    return request.headers.get("X-API-Token") or None


def create_app(*, pipeline: NovaVision | None = None) -> Flask:
    settings = Settings() if pipeline is None else None
    app = Flask(__name__, static_folder=str(_ROOT / "docs"), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024

    origins = [
        origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()
    ]
    if origins:
        from flask_cors import CORS

        CORS(app, origins=origins)

    rate_limiter = RateLimiter(env_int("NOVA_RATE_LIMIT", 30))
    generation_guard = ConcurrencyGuard(env_int("NOVA_MAX_CONCURRENCY", 2))
    pipeline_lock = threading.Lock()

    def get_pipeline() -> NovaVision:
        nonlocal pipeline
        if pipeline is None:
            with pipeline_lock:
                if pipeline is None:
                    pipeline = build_pipeline(settings)
        return pipeline

    def rate_limited():
        if not rate_limiter.allow(request.remote_addr or "unknown"):
            return jsonify({"error": "Rate limit exceeded. Try again shortly."}), 429
        return None

    @app.before_request
    def validate_host():
        if not public_enabled() and not is_loopback_host(request.headers.get("Host", "")):
            return jsonify({"error": "Invalid host header."}), 400

    @app.get("/")
    @app.get("/index.html")
    def index():
        html = (_ROOT / "docs" / "index.html").read_text(encoding="utf-8")
        return app.response_class(
            html.replace('<html lang="en" data-preview="true">', '<html lang="en">', 1),
            mimetype="text/html",
        )

    @app.get("/figma/<path:filename>")
    def figma_assets(filename):
        return send_from_directory(_ROOT / "figma", filename)

    @app.post("/api/analyze")
    def analyze():
        if (limited := rate_limited()) is not None:
            return limited
        if public_enabled() and not token_ok(_bearer_token()):
            return jsonify({"error": "Unauthorized."}), 401
        text, error = _valid_text(request.get_json(silent=True))
        if error:
            return jsonify({"error": error}), 400

        try:
            analysis = get_pipeline().analyzer.analyze(text)
            return jsonify({"success": True, **_analysis_payload(analysis)})
        except Exception:
            logger.exception("Analysis failed")
            return jsonify({"error": "Analysis failed."}), 500

    @app.post("/api/generate")
    def generate():
        if (limited := rate_limited()) is not None:
            return limited
        if not token_ok(_bearer_token()):
            return jsonify({"error": "Unauthorized."}), 401
        data = request.get_json(silent=True)
        data = data if isinstance(data, dict) else {}
        text, error = _valid_text(data)
        if error:
            return jsonify({"error": error}), 400
        try:
            style, seed = _generation_options(data)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        if not generation_guard.acquire():
            return jsonify({"error": "Server busy. Try again shortly."}), 429
        try:
            result = get_pipeline().auto_run(text, style=style, seed=seed)
            with BytesIO() as buffer:
                result.image.save(buffer, format="PNG")
                image = base64.b64encode(buffer.getvalue()).decode("ascii")
            return jsonify(
                {
                    "success": True,
                    **_analysis_payload(result.analysis),
                    "image": f"data:image/png;base64,{image}",
                    "prompt": result.prompt,
                    "original_text": text,
                    "style": style,
                    "seed": result.seed,
                    "backend": result.backend,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        except TimeoutError:
            logger.warning("Generation timed out")
            return jsonify({"error": "Generation timed out. Try again shortly."}), 504
        except Exception:
            logger.exception("Generation failed")
            return jsonify({"error": "Generation failed."}), 500
        finally:
            generation_guard.release()

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host = resolve_host()
    port = env_int("PORT", 8000)
    if host == "0.0.0.0":
        logger.warning(
            "Listening on 0.0.0.0. Set NOVA_API_TOKEN to enable public API requests. "
            "For deployment, run python -m gunicorn 'novavision.web:create_app()'."
        )
    create_app().run(host=host, port=port)


if __name__ == "__main__":
    main()
