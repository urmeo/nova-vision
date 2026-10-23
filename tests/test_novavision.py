import base64
from io import BytesIO
from unittest.mock import Mock

import pytest
from PIL import Image

from novavision.affect.analyzer import EmotionAnalyzer
from novavision.affect.lexicon import AffectLexicon
from novavision.generation.base import NullBackend
from novavision.pipeline import NovaVision
from novavision.web import create_app


@pytest.fixture
def pipeline():
    analyzer = EmotionAnalyzer(lexicon=AffectLexicon({"happy": (0.8, 0.6)}))
    analyzer._classifier = Mock(
        return_value=[
            [
                {"label": "joy", "score": 0.9},
                {"label": "neutral", "score": 0.1},
            ]
        ]
    )
    return NovaVision(backend=NullBackend(), analyzer=analyzer)


def test_text_to_image_pipeline_is_deterministic(pipeline):
    text = "A happy sunrise"
    result = pipeline.auto_run(text, style="nature", seed=17, width=32, height=24)
    repeated = pipeline.auto_run(text, style="nature", seed=17, width=32, height=24)

    assert result.analysis.primary == "joy"
    assert result.analysis.valence > 0
    assert (result.backend, result.tier, result.seed) == ("null", "affect", 17)
    assert text in result.prompt
    assert "joyful" in result.prompt
    assert "landscape photography" in result.prompt
    assert result.image.mode == "RGB"
    assert result.image.size == (32, 24)
    assert result.prompt == repeated.prompt
    assert result.image.tobytes() == repeated.image.tobytes()


def test_flask_analyzes_text_and_returns_a_png(pipeline, monkeypatch):
    monkeypatch.setenv("NOVA_PUBLIC", "0")
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setenv("NOVA_RATE_LIMIT", "30")
    monkeypatch.setenv("NOVA_MAX_CONCURRENCY", "2")
    monkeypatch.delenv("NOVA_API_TOKEN", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    client = create_app(pipeline=pipeline).test_client()
    text = "A happy sunrise"

    analysis = client.post("/api/analyze", json={"text": text})
    assert analysis.status_code == 200
    assert analysis.json["success"] is True
    assert analysis.json["primary_emotion"] == "joy"
    assert analysis.json["confidence"] == 90

    invalid = client.post("/api/generate", json={"text": "x"})
    assert invalid.status_code == 400
    assert invalid.json["error"]

    response = client.post("/api/generate", json={"text": text, "style": "nature", "seed": 17})
    assert response.status_code == 200
    data = response.json
    assert data["success"] is True
    assert (data["backend"], data["seed"], data["style"]) == ("null", 17, "nature")
    assert data["primary_emotion"] == "joy"
    assert data["original_text"] == text
    assert text in data["prompt"]
    assert data["image"].startswith("data:image/png;base64,")
    encoded = data["image"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
        image.load()
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (512, 512)

    monkeypatch.setenv("NOVA_PUBLIC", "1")
    for token in ("", "test-token"):
        monkeypatch.setenv("NOVA_API_TOKEN", token)
        protected = create_app(pipeline=pipeline).test_client()
        calls = pipeline.analyzer._classifier.call_count
        for endpoint in ("/api/analyze", "/api/generate"):
            for headers in ({}, {"X-API-Token": "wrong"}):
                denied = protected.post(endpoint, json={"text": text}, headers=headers)
                assert denied.status_code == 401
        assert pipeline.analyzer._classifier.call_count == calls
        if token:
            assert protected.post(
                "/api/analyze", json={"text": text}, headers={"X-API-Token": token}
            ).status_code == 200
            assert protected.post(
                "/api/generate", json={"text": text},
                headers={"Authorization": f"Bearer {token}"},
            ).status_code == 200

    monkeypatch.setenv("NOVA_PUBLIC", "0")
    monkeypatch.setenv("NOVA_TRUST_PROXY", "1")
    monkeypatch.setenv("NOVA_RATE_LIMIT", "1")
    limited = create_app(pipeline=pipeline).test_client()
    statuses = [
        limited.post(
            "/api/analyze", json={"text": text},
            headers={"X-Forwarded-For": f"198.51.100.{index}, 203.0.113.10"},
        ).status_code
        for index in (1, 2)
    ]
    assert statuses == [200, 429]
