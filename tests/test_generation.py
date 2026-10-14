import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from PIL import Image

from novavision.generation import NullBackend, get_backend
from novavision.generation.diffusers_backend import DiffusersBackend
from novavision.generation.hf_api_backend import HFApiBackend


class RecordingClient:
    def __init__(self):
        self.parameters = {}

    def text_to_image(self, prompt, **parameters):
        self.parameters.update(parameters)
        return Image.new("RGB", (8, 8))


def test_factory():
    assert isinstance(get_backend("null"), NullBackend)
    with pytest.raises(ValueError):
        get_backend("nope")


def test_hf_api_ignores_device_kwarg(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy")
    backend = get_backend("hf-api", model_id="stabilityai/sd-turbo", device="cpu")
    assert backend.name == "hf-api"


def test_size_and_mode():
    image = NullBackend().generate("hi", width=64, height=48, seed=1)
    assert image.size == (64, 48)
    assert image.mode == "RGB"


def test_seed_is_deterministic():
    first = NullBackend().generate("hi", seed=7).tobytes()
    repeated = NullBackend().generate("hi", seed=7).tobytes()
    other_seed = NullBackend().generate("hi", seed=8).tobytes()
    assert first == repeated
    assert first != other_seed


def test_negative_seed_does_not_crash():
    image = NullBackend().generate("hi", width=8, height=8, seed=-1)
    assert image.size == (8, 8)


@pytest.mark.parametrize(
    ("model_id", "expected_negative_prompt"),
    [
        ("stabilityai/stable-diffusion-2-1", "blurry"),
        ("stabilityai/sd-turbo", None),
    ],
)
def test_hf_api_negative_prompt_depends_on_model(monkeypatch, model_id, expected_negative_prompt):
    monkeypatch.setenv("HF_TOKEN", "x")
    client = RecordingClient()
    backend = HFApiBackend(model_id=model_id)
    backend._client = client
    backend.generate("p", width=8, height=8, seed=3, negative_prompt="blurry")
    assert client.parameters["negative_prompt"] == expected_negative_prompt
    assert client.parameters["seed"] == 3


@pytest.mark.parametrize(
    ("device", "dtype"),
    [("cpu", "float32"), ("cuda", "float16"), ("cuda:0", "float16"), ("mps", "float16")],
)
def test_diffusers_loads_the_selected_precision(monkeypatch, device, dtype):
    loaded = {}

    class FakePipeline:
        def to(self, target):
            loaded["device"] = target
            return self

        def set_progress_bar_config(self, **kwargs):
            pass

    def from_pretrained(model_id, **kwargs):
        loaded.update(kwargs)
        return FakePipeline()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="float16", float32="float32"))
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(AutoPipelineForText2Image=SimpleNamespace(from_pretrained=from_pretrained)),
    )

    backend = DiffusersBackend(device=device)
    assert backend.dtype == dtype
    assert backend.pipe is backend.pipe
    assert loaded["torch_dtype"] == dtype
    assert loaded["device"] == device


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_diffusers_selects_available_hardware(monkeypatch, cuda_available, mps_available, expected):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda_available),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
        ),
    )

    assert DiffusersBackend().device == expected


def test_diffusers_serializes_requests_using_the_cached_pipeline(monkeypatch):
    class FakeGenerator:
        def __init__(self, **kwargs):
            pass

        def manual_seed(self, seed):
            return self

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Generator=FakeGenerator))
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    image = Image.new("RGB", (8, 8))

    def fake_pipe(**kwargs):
        if kwargs["prompt"] == "first":
            first_entered.set()
            assert release_first.wait(3), "test did not release the first request"
        else:
            second_entered.set()
        return SimpleNamespace(images=[image])

    backend = DiffusersBackend(device="cpu")
    backend._pipe = fake_pipe

    def second_request():
        second_attempted.set()
        return backend.generate("second", width=8, height=8)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(backend.generate, "first", width=8, height=8)
        try:
            assert first_entered.wait(3)
            second = executor.submit(second_request)
            assert second_attempted.wait(3)
            assert not second_entered.wait(0.1), "shared pipeline ran two requests at once"
        finally:
            release_first.set()
        assert first.result(timeout=3) is image
        assert second.result(timeout=3) is image
        assert second_entered.is_set()


def test_null_backend_prompt_sensitivity():
    first = NullBackend().generate("hi", seed=7).tobytes()
    other_prompt = NullBackend().generate("bye", seed=7).tobytes()
    assert first != other_prompt


def test_hf_api_default_size_matches_diffusers(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    client = RecordingClient()
    backend = HFApiBackend(model_id="stabilityai/sd-turbo")
    backend._client = client
    backend.generate("p", seed=1)
    assert (client.parameters["width"], client.parameters["height"]) == (512, 512)
