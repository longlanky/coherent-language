"""Tests for the CLI client's credential handling.

`transcribe` is an extension-less script, so it is loaded by path rather than
imported as a module.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "transcribe"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_loader(
        "transcribe_cli", importlib.machinery.SourceFileLoader("transcribe_cli", str(SCRIPT))
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_host_parsing_is_exact_not_substring(cli):
    assert cli.server_host("https://asr.ninefive.cc") == "asr.ninefive.cc"
    assert cli.server_host("http://127.0.0.1:8000") == "127.0.0.1"
    assert cli.server_host("http://[::1]:8000") == "::1"
    # A substring test would wrongly call this "local" and suppress the check.
    assert cli.server_host("https://localhost.evil.example") == "localhost.evil.example"


def test_local_hosts_skip_credentials(cli):
    assert "127.0.0.1" in cli.LOCAL_HOSTS
    assert "::1" in cli.LOCAL_HOSTS
    assert "localhost.evil.example" not in cli.LOCAL_HOSTS


def test_trusted_host_is_the_known_endpoint(cli):
    assert cli.TRUSTED_HOSTS == {"asr.ninefive.cc"}


def test_poll_loop_gives_up_instead_of_hanging(cli, monkeypatch, tmp_path):
    """A permanently 'queued' job (dead worker, lost job) must produce an
    error, not an endless stream of dots."""
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF" + b"\0" * 32)
    monkeypatch.setattr(
        sys, "argv",
        ["transcribe", str(audio), "--server", "http://127.0.0.1:8000"],
    )
    monkeypatch.setattr(cli, "MAX_WAIT_S", 0)
    monkeypatch.setattr(cli, "POLL_EVERY", 0)

    def fake_http_json(req):
        if req.get_method() == "POST":
            return {"job_id": "stuck-job", "status": "queued"}
        return {"status": "queued"}  # never progresses

    monkeypatch.setattr(cli, "http_json", fake_http_json)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "Timed out" in str(exc.value)


def test_local_server_is_not_sent_a_credential(cli, monkeypatch, tmp_path):
    """A credential in hand must still not be attached to a localhost request.

    All HTTP is stubbed: these tests must never touch the running service.
    """
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF" + b"\0" * 32)
    seen = []
    monkeypatch.setattr(
        sys, "argv",
        ["transcribe", str(audio), "--server", "http://127.0.0.1:8000"],
    )
    monkeypatch.setattr(cli, "get_cookie", lambda _args: "FAKE-JWT")
    monkeypatch.setattr(cli, "MAX_WAIT_S", 0)
    monkeypatch.setattr(cli, "POLL_EVERY", 0)

    def fake_http_json(req):
        seen.append(req)
        if req.get_method() == "POST":
            return {"job_id": "stub", "status": "queued"}
        return {"status": "queued"}

    monkeypatch.setattr(cli, "http_json", fake_http_json)
    with pytest.raises(SystemExit):
        cli.main()
    assert seen, "the client should have attempted a submit"
    assert all(not req.get_header("Cookie") for req in seen)


def test_untrusted_remote_host_refuses_the_credential(cli, monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF" + b"\0" * 32)
    monkeypatch.setattr(
        sys, "argv",
        ["transcribe", str(audio), "--server", "https://evil.example"],
    )
    monkeypatch.setattr(cli, "get_cookie", lambda _args: "FAKE-JWT")
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "Refusing to send" in str(exc.value)


def test_lookalike_hostname_is_not_treated_as_local(cli, monkeypatch, tmp_path):
    """'localhost.evil.example' must not pass the old substring check."""
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF" + b"\0" * 32)
    monkeypatch.setattr(
        sys, "argv",
        ["transcribe", str(audio), "--server", "https://localhost.evil.example"],
    )
    monkeypatch.setattr(cli, "get_cookie", lambda _args: "FAKE-JWT")
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "Refusing to send" in str(exc.value)
