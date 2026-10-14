import multiprocessing
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
from PIL import Image

from novavision.affect.analyzer import EmotionAnalysis
from novavision.config import Settings
from novavision.generation import hf_api_backend
from novavision.generation.hf_api_backend import HFApiBackend
from novavision.pipeline import NovaVision, build_pipeline


def stalled_or_successful_worker(
    connection, image_path, model_id, token, timeout, prompt, parameters
):
    if "stalled request" in prompt:
        time.sleep(60)
    Image.new("RGB", (8, 8)).save(image_path)
    connection.send(("ok", ""))
    connection.close()


def empty_worker(connection, *args):
    connection.close()


def reported_error_worker(connection, image_path, model_id, token, timeout, prompt, parameters):
    connection.send((prompt, "Reported failure"))
    connection.close()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("-inf"), float("nan"), True])
def test_hosted_timeout_rejects_unbounded_values(timeout):
    with pytest.raises(ValueError, match="finite and greater than zero"):
        HFApiBackend(token="test-token", timeout=timeout)


def test_hosted_client_has_a_default_and_configurable_timeout(monkeypatch, tmp_path):
    clients = []
    messages = []

    def client(**kwargs):
        clients.append(kwargs)
        return SimpleNamespace(text_to_image=lambda *args, **kwargs: Image.new("RGB", (8, 8)))

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(InferenceClient=client))
    connection = SimpleNamespace(send=messages.append, close=lambda: None)

    for timeout in (None, 3.5):
        backend = HFApiBackend(
            token="test-token", **({} if timeout is None else {"timeout": timeout})
        )
        hf_api_backend._generate_image(
            connection,
            tmp_path / "image.png",
            backend.model_id,
            backend.token,
            backend.timeout,
            "a quiet forest",
            {},
        )

    assert clients == [
        {"token": "test-token", "timeout": 120.0},
        {"token": "test-token", "timeout": 3.5},
    ]
    assert messages == [("ok", ""), ("ok", "")]


def test_hosted_timeout_does_not_mask_other_failures(monkeypatch, tmp_path):
    failure = ValueError("Unsupported model")
    messages = []

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(InferenceClient=lambda **kwargs: SimpleNamespace(text_to_image=fail)),
    )
    connection = SimpleNamespace(send=messages.append, close=lambda: None)
    hf_api_backend._generate_image(
        connection,
        tmp_path / "image.png",
        "test-model",
        "test-token",
        1,
        "a quiet forest",
        {},
    )
    assert messages == [("error", "ValueError: Unsupported model")]


def test_hosted_sdk_timeout_is_translated(monkeypatch, tmp_path):
    errors = pytest.importorskip("huggingface_hub.errors")
    failure = errors.InferenceTimeoutError("Request stalled")
    messages = []

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(InferenceClient=lambda **kwargs: SimpleNamespace(text_to_image=fail)),
    )
    connection = SimpleNamespace(send=messages.append, close=lambda: None)
    hf_api_backend._generate_image(
        connection,
        tmp_path / "image.png",
        "test-model",
        "test-token",
        1,
        "a quiet forest",
        {},
    )
    assert messages == [("timeout", "InferenceTimeoutError: Request stalled")]


@pytest.mark.parametrize("module, name", [("httpx", "ReadTimeout"), ("requests", "ReadTimeout")])
def test_hosted_transport_timeouts_are_recognized(module, name):
    dependency = pytest.importorskip(module)
    exception = getattr(dependency, name)("Request stalled")
    assert hf_api_backend._is_timeout(exception)


def test_stalled_hosted_http_request_times_out(monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    client = hub.InferenceClient(token="test-token", timeout=0.2)
    monkeypatch.setattr(hub, "InferenceClient", lambda **kwargs: client)
    release = threading.Event()
    entered = threading.Event()
    messages = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            entered.set()
            release.wait(5)

        def log_message(self, *args):
            pass

    http = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever)
    worker.start()
    try:
        hf_api_backend._generate_image(
            SimpleNamespace(send=messages.append, close=lambda: None),
            tmp_path / "image.png",
            f"http://127.0.0.1:{http.server_port}",
            "test-token",
            0.2,
            "a quiet forest",
            {},
        )
        assert entered.is_set()
        assert len(messages) == 1
        assert messages[0][0] == "timeout"
        assert not (tmp_path / "image.png").exists()
    finally:
        release.set()
        http.shutdown()
        worker.join(timeout=3)
        http.server_close()
    assert not worker.is_alive()


def test_generation_timeout_returns_504_and_releases_slot(monkeypatch, tmp_path):
    pytest.importorskip("flask")
    import server
    from novavision.serving import ConcurrencyGuard, RateLimiter

    monkeypatch.setattr(hf_api_backend, "_generate_image", stalled_or_successful_worker)
    monkeypatch.setattr(
        hf_api_backend,
        "TemporaryDirectory",
        lambda **kwargs: TemporaryDirectory(dir=tmp_path, **kwargs),
    )
    analysis = EmotionAnalysis("joy", 0.9, 0.8, 0.7, 1.0, {"joy": 0.9})
    backend = HFApiBackend(token="test-token", timeout=2)
    pipeline = NovaVision(backend=backend, analyzer=SimpleNamespace(analyze=lambda text: analysis))
    monkeypatch.setattr(server, "_pipeline", pipeline)
    monkeypatch.setattr(server, "_gen_guard", ConcurrencyGuard(max_concurrent=1))
    monkeypatch.setattr(server, "_rate_limiter", RateLimiter(max_requests=10))
    monkeypatch.delenv("NOVA_API_TOKEN", raising=False)

    client = server.app.test_client()
    children = {child.pid for child in multiprocessing.active_children()}
    first = client.post("/api/generate", json={"text": "stalled request", "seed": 1})
    assert {child.pid for child in multiprocessing.active_children()} == children
    backend.timeout = 10
    second = client.post("/api/generate", json={"text": "a quiet forest", "seed": 1})

    assert first.status_code == 504
    assert first.get_json() == {"error": "Generation timed out. Try again shortly."}
    assert second.status_code == 200
    assert second.get_json()["backend"] == "hf-api"
    assert {child.pid for child in multiprocessing.active_children()} == children
    assert not list(tmp_path.iterdir())


def test_hosted_worker_exit_without_result_is_reported(monkeypatch):
    monkeypatch.setattr(hf_api_backend, "_generate_image", empty_worker)
    backend = HFApiBackend(token="test-token", timeout=10)
    with pytest.raises(RuntimeError, match="exited without a result"):
        backend.generate("a quiet forest")


@pytest.mark.parametrize("status, exception", [("timeout", TimeoutError), ("error", RuntimeError)])
def test_hosted_worker_reports_timeout_separately_from_other_errors(monkeypatch, status, exception):
    monkeypatch.setattr(hf_api_backend, "_generate_image", reported_error_worker)
    backend = HFApiBackend(token="test-token", timeout=10)
    children = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(exception, match="Reported failure"):
        backend.generate(status)
    assert {child.pid for child in multiprocessing.active_children()} == children


def test_hosted_timeout_environment_reaches_backend(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "hf-api")
    monkeypatch.setenv("NOVA_HF_TIMEOUT", "7.5")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    settings = Settings()
    assert settings.hf_timeout == 7.5
    assert build_pipeline(settings).backend.timeout == 7.5


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "-inf"])
def test_hosted_timeout_environment_rejects_unbounded_values(monkeypatch, timeout):
    monkeypatch.setenv("NOVA_HF_TIMEOUT", timeout)
    with pytest.raises(ValueError):
        Settings()
