import json

import pytest

pytest.importorskip("flask")

from PIL import Image  # noqa: E402

import server  # noqa: E402
from novavision.affect.analyzer import EmotionAnalysis  # noqa: E402
from novavision.pipeline import Result  # noqa: E402


class FakeAnalyzer:
    def analyze(self, text):
        return EmotionAnalysis("joy", 0.91, 0.8, 0.7, 1.0, {"joy": 0.91, "sadness": 0.09})


class FakePipeline:
    analyzer = FakeAnalyzer()
    backend_name = "null"

    def auto_run(self, text, style="artistic", seed=0):
        image = Image.new("RGB", (8, 8), (10, 20, 30))
        return Result(
            image, "prompt", self.analyzer.analyze(text), "affect", seed, self.backend_name
        )


@pytest.fixture
def client(monkeypatch):
    from novavision import serving

    monkeypatch.setattr(server, "_pipeline", FakePipeline())
    monkeypatch.setattr(server, "_rate_limiter", serving.RateLimiter(max_requests=1000))
    monkeypatch.setattr(server, "_gen_guard", serving.ConcurrencyGuard(max_concurrent=4))
    monkeypatch.delenv("NOVA_API_TOKEN", raising=False)
    return server.app.test_client()


def test_analyze_ok(client):
    resp = client.post("/api/analyze", json={"text": "i feel great"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["primary_emotion"] == "joy"
    assert data["emotions"][0]["name"] == "joy"


def test_analyze_short_text(client):
    assert client.post("/api/analyze", json={"text": "hi"}).status_code == 400


def test_non_string_text_is_rejected(client):
    for bad in (123, ["a", "b"], True, None, {"k": 1}):
        for route in ("/api/analyze", "/api/generate"):
            assert client.post(route, json={"text": bad}).status_code == 400


def test_generate_ok(client):
    resp = client.post(
        "/api/generate", json={"text": "i feel great", "style": "artistic", "seed": 5}
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["image"].startswith("data:image/png;base64,")
    assert data["seed"] == 5
    assert data["backend"] == "null"


@pytest.mark.parametrize("backend", ["null", "diffusers", "hf-api"])
def test_generate_identifies_the_backend_that_produced_the_result(client, backend):
    server._pipeline.backend_name = backend

    response = client.post("/api/generate", json={"text": "i feel great"})

    assert response.status_code == 200
    assert response.get_json()["backend"] == backend


def test_control_chars_stripped():
    text, err = server._valid_text({"text": "hello\x00\x07 there\x1f"})
    assert err is None
    assert all(c not in text for c in "\x00\x07\x1f")


def test_index_served(client):
    assert client.get("/").status_code == 200


@pytest.mark.parametrize(
    "name", ["pipeline.svg", "pipeline-mobile.svg", "pipeline.png", "pipeline-mobile.png"]
)
def test_pipeline_assets_served(client, name):
    response = client.get(f"/assets/{name}")

    assert response.status_code == 200
    if name.endswith(".svg"):
        assert response.mimetype == "image/svg+xml"
        assert response.data.lstrip().startswith(b"<svg")
    else:
        assert response.mimetype == "image/png"
        assert response.data.startswith(b"\x89PNG\r\n\x1a\n")


def test_asset_route_cannot_expose_repository_files(client):
    assert client.get("/assets/../../server.py").status_code == 404


def test_repo_root_not_exposed(client):
    for leak in ("/server.py", "/config.py", "/.env", "/outputs/results/results.json"):
        assert client.get(leak).status_code == 404


def test_rate_limit_returns_429(client):
    from novavision import serving

    server._rate_limiter = serving.RateLimiter(max_requests=1)
    assert client.post("/api/analyze", json={"text": "hello there"}).status_code == 200
    assert client.post("/api/analyze", json={"text": "hello again"}).status_code == 429


def test_rotating_xff_does_not_bypass_rate_limit(client, monkeypatch):
    monkeypatch.delenv("NOVA_TRUST_PROXY", raising=False)
    from novavision import serving

    server._rate_limiter = serving.RateLimiter(max_requests=1)
    a = client.post("/api/analyze", json={"text": "hello"}, headers={"X-Forwarded-For": "1.1.1.1"})
    b = client.post("/api/analyze", json={"text": "world"}, headers={"X-Forwarded-For": "2.2.2.2"})
    assert a.status_code == 200 and b.status_code == 429


def test_busy_returns_429(client):
    from novavision import serving

    server._gen_guard = serving.ConcurrencyGuard(max_concurrent=1)
    server._gen_guard.acquire()
    resp = client.post("/api/generate", json={"text": "i feel great"})
    assert resp.status_code == 429


def test_generate_requires_token_when_set(client, monkeypatch):
    monkeypatch.setenv("NOVA_API_TOKEN", "s3cret")
    assert client.post("/api/generate", json={"text": "i feel great"}).status_code == 401
    ok = client.post(
        "/api/generate",
        json={"text": "i feel great"},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert ok.status_code == 200


def test_generate_unknown_style_rejected(client):
    resp = client.post("/api/generate", json={"text": "i feel great", "style": "<script>"})
    assert resp.status_code == 400
    assert "style" in resp.get_json()["error"].lower()


def test_generate_seed_must_fit_64_bits(client):
    ok = client.post("/api/generate", json={"text": "i feel great", "seed": 2**63 - 1})
    assert ok.status_code == 200
    for bad in (2**63, -(2**63), 10**30):
        resp = client.post("/api/generate", json={"text": "i feel great", "seed": bad})
        assert resp.status_code == 400


def test_oversize_body_rejected(client):
    body = '{"text": "' + "x" * (33 * 1024) + '"}'
    resp = client.post("/api/analyze", data=body, content_type="application/json")
    assert resp.status_code == 413


def test_worst_case_legal_body_fits_under_cap(client):
    body = json.dumps({"text": "\U0001f600" * 2000, "style": "artistic", "seed": 1})
    resp = client.post("/api/generate", data=body, content_type="application/json")
    assert resp.status_code == 200


def test_nonstandard_json_seed_literals_rejected(client):
    for literal in ("Infinity", "-Infinity", "NaN", "true"):
        body = '{"text": "i feel great", "seed": ' + literal + "}"
        resp = client.post("/api/generate", data=body, content_type="application/json")
        assert resp.status_code == 400, literal


def test_fractional_seed_rejected(client):
    bad = client.post("/api/generate", json={"text": "i feel great", "seed": 2.9})
    assert bad.status_code == 400
    ok = client.post("/api/generate", json={"text": "i feel great", "seed": 2.0})
    assert ok.status_code == 200
    assert ok.get_json()["seed"] == 2


def test_non_dict_json_bodies_rejected(client):
    for body in ("[1, 2]", '"abc"', "123", "true"):
        for route in ("/api/analyze", "/api/generate"):
            resp = client.post(route, data=body, content_type="application/json")
            assert resp.status_code == 400, (route, body)
