"""Hosted deployment (Streamlit Community Cloud): no datasets, no backend.

The app must still run everything it can, fetch the small IoT-23 logs on
demand, and fall back cleanly where a dataset cannot be fetched."""
import io as stdio
import os

import pytest

from data import fetch
from data import load_iot23 as io
from data import load_nbaiot as nb

ZEEK = ("#separator \\x09\n#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\t"
        "service\tduration\torig_bytes\tresp_bytes\tconn_state\tlocal_orig\tlocal_resp\t"
        "missed_bytes\thistory\torig_pkts\torig_ip_bytes\tresp_pkts\tresp_ip_bytes\t"
        "tunnel_parents\tlabel\tdetailed-label\n"
        "1.0\tC1\t10.0.0.1\t1\t10.0.0.2\t23\ttcp\t-\t0.5\t10\t0\tS0\t-\t-\t0\tS\t2\t120\t0\t0\t-"
        "\tbenign\t-\n")


class _Resp(stdio.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def empty_iot23(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "DEFAULT_ROOT", str(tmp_path / "iot23"))
    io._CACHE.clear()
    yield tmp_path / "iot23"
    io._CACHE.clear()


def test_ensure_iot23_downloads_every_capture(empty_iot23, monkeypatch):
    urls = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return _Resp(ZEEK.encode())
    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    assert fetch.ensure_iot23()
    assert len(urls) == 3 and all(u.startswith("https://mcfp.felk.cvut.cz/") for u in urls)
    for cap in fetch.IOT23_CAPTURES:
        assert (empty_iot23 / cap / "conn.log.labeled").exists()
    assert not list(empty_iot23.rglob("*.part"))


def test_ensure_iot23_rejects_a_non_zeek_download(empty_iot23, monkeypatch):
    monkeypatch.setattr(fetch.urllib.request, "urlopen",
                        lambda url, timeout: _Resp(b"<html>not found</html>"))
    assert not fetch.ensure_iot23()
    assert not list(empty_iot23.rglob("conn.log.labeled"))


def test_ensure_iot23_survives_no_network(empty_iot23, monkeypatch):
    def offline(url, timeout):
        raise OSError("no network")
    monkeypatch.setattr(fetch.urllib.request, "urlopen", offline)
    assert fetch.ensure_iot23() is False


def test_ensure_iot23_is_a_no_op_when_present(monkeypatch):
    if not io.available():
        pytest.skip("IoT-23 not downloaded")
    monkeypatch.setattr(fetch.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("must not download"))
    assert fetch.ensure_iot23()


def test_fallback_feature_names_match_the_real_columns():
    from data.fake_nbaiot import feature_names
    if not os.path.exists(nb.paths()["benign"]):
        pytest.skip("N-BaIoT not downloaded")
    assert feature_names() == nb.feature_names()


def test_app_without_datasets_or_backend(tmp_path, monkeypatch):
    """Live re-run on N-BaIoT without the files warns and shows the precomputed run."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(nb, "DEFAULT_ROOT", str(tmp_path / "no_nbaiot"))
    monkeypatch.setenv("TRUST_IDS_BACKEND", "http://127.0.0.1:9")
    at = AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py"),
                           default_timeout=300).run()
    at.sidebar.toggle[0].set_value(True)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("not on this server" in w.value for w in at.sidebar.warning)
    assert any(t.label == "Live experiment" for t in at.tabs)
