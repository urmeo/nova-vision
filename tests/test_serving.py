import pytest

from novavision import serving


def test_resolve_host_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("NOVA_PUBLIC", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    assert serving.resolve_host() == "127.0.0.1"
    assert serving.public_enabled() is False


def test_resolve_host_public_opt_in(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setenv("NOVA_PUBLIC", "1")
    assert serving.resolve_host() == "0.0.0.0"


def test_resolve_host_spaces_sandbox(monkeypatch):
    monkeypatch.delenv("NOVA_PUBLIC", raising=False)
    monkeypatch.setenv("SPACE_ID", "user/novavision")
    assert serving.resolve_host() == "0.0.0.0"


def test_token_ok_disabled_without_env(monkeypatch):
    monkeypatch.delenv("NOVA_API_TOKEN", raising=False)
    assert serving.token_ok(None) is True


def test_token_ok_enforced_when_set(monkeypatch):
    monkeypatch.setenv("NOVA_API_TOKEN", "s3cret")
    assert serving.token_ok("s3cret") is True
    assert serving.token_ok("wrong") is False
    assert serving.token_ok(None) is False


def test_token_ok_handles_non_ascii(monkeypatch):
    monkeypatch.setenv("NOVA_API_TOKEN", "clé-secrète")
    assert serving.token_ok("clé-secrète") is True
    assert serving.token_ok("wrong") is False


def test_rate_limiter_evicts_stale_keys():
    rl = serving.RateLimiter(max_requests=1, window_seconds=10, gc_threshold=2)

    for i in range(10):
        rl.allow(f"ip-{i}", now=0.0)
    rl.allow("trigger", now=100.0)
    assert len(rl._hits) <= 2


def test_rate_limiter_blocks_over_budget():
    rl = serving.RateLimiter(max_requests=2, window_seconds=60)
    assert rl.allow("ip", now=0.0) is True
    assert rl.allow("ip", now=1.0) is True
    assert rl.allow("ip", now=2.0) is False

    assert rl.allow("other", now=2.0) is True


def test_rate_limiter_window_slides():
    rl = serving.RateLimiter(max_requests=1, window_seconds=10)
    assert rl.allow("ip", now=0.0) is True
    assert rl.allow("ip", now=5.0) is False
    assert rl.allow("ip", now=11.0) is True


def test_concurrency_guard_caps_slots():
    guard = serving.ConcurrencyGuard(max_concurrent=1)
    assert guard.acquire() is True
    assert guard.acquire() is False
    guard.release()
    assert guard.acquire() is True


def test_env_int_reads_valid(monkeypatch):
    monkeypatch.setenv("NOVA_X", "42")
    assert serving.env_int("NOVA_X", 7) == 42


def test_env_int_unset_uses_default(monkeypatch):
    monkeypatch.delenv("NOVA_X", raising=False)
    assert serving.env_int("NOVA_X", 7) == 7


def test_env_int_malformed_fails_loudly(monkeypatch):
    monkeypatch.setenv("NOVA_X", "1O")
    with pytest.raises(ValueError, match="must be an integer"):
        serving.env_int("NOVA_X", 7)
