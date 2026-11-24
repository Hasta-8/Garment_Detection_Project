import threading
import time

import pytest
from fastapi.testclient import TestClient
from helpers import FakeModel, encode, garment_image

import app.main as main
from app.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, max_upload_mb=1, busy_timeout_s=0.2, demo_dir=tmp_path)


@pytest.fixture
def make_client(settings):
    def make(model=None, **overrides):
        s = settings.model_copy(update=overrides)
        return TestClient(main.create_app(s, model or FakeModel()), raise_server_exceptions=False)

    return make


def post(client, data, path="/v1/analyze", name="g.jpg"):
    return client.post(path, files={"file": (name, data, "image/jpeg")})


def test_analyze_returns_the_contract(make_client):
    r = post(make_client(), encode(garment_image()))
    assert r.status_code == 200, r.text
    b = r.json()
    assert set(b["attributes"]) == {"category", "colour", "pattern", "sleeve", "gender"}
    cat = b["attributes"]["category"]
    assert cat["label"] == "t-shirt" and len(cat["alternatives"]) == 3
    assert cat["alternatives"][0] == {"label": cat["label"], "score": cat["score"]}
    assert b["title"] == "Men's black striped t-shirt, sleeveless"
    assert b["image"]["width"] == 256 and b["request_id"] == r.headers["x-request-id"]
    assert set(b["timings"]) == {"decode_ms", "model_ms", "total_ms"}


def test_legacy_path_and_top_k(make_client):
    r = post(make_client(top_k=5), encode(garment_image()), path="/analyze")
    assert r.status_code == 200 and len(r.json()["attributes"]["colour"]["alternatives"]) == 5


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        (b"not an image", 415, "unsupported_type"),
        (b"", 422, "empty_file"),
        (encode(garment_image(16, 16)), 422, "too_small"),
    ],
)
def test_bad_uploads_get_clear_errors(make_client, data, status, code):
    r = post(make_client(), data)
    assert (r.status_code, r.json()["error"]) == (status, code)
    assert set(r.json()) == {"error", "detail", "request_id"}


def test_oversized_upload_is_413(make_client):
    r = post(make_client(), b"\xff" * (1024 * 1024 + 10))
    assert (r.status_code, r.json()["error"]) == (413, "too_large")


def test_missing_file_is_422(make_client):
    r = make_client().post("/v1/analyze")
    assert (r.status_code, r.json()["error"]) == (422, "invalid_request")


def test_ready_is_503_until_the_model_loads(settings, monkeypatch):
    gate = threading.Event()

    def slow_load(_s):
        gate.wait(5)
        return FakeModel()

    monkeypatch.setattr(main, "load_model", slow_load)
    with TestClient(main.create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        r = client.get("/ready")
        assert (r.status_code, r.json()["error"]) == (503, "not_ready")
        assert post(client, encode(garment_image())).status_code == 503
        gate.set()
        for _ in range(50):
            if client.get("/ready").status_code == 200:
                break
            time.sleep(0.05)
        assert client.get("/ready").status_code == 200


def test_failed_loading_is_reported_without_details(settings, monkeypatch):
    def broken(_s):
        raise RuntimeError("secret path /home/x")

    monkeypatch.setattr(main, "load_model", broken)
    with TestClient(main.create_app(settings)) as client:
        for _ in range(50):
            r = client.get("/ready")
            if r.json().get("error") == "startup_failed":
                break
            time.sleep(0.05)
        assert r.json()["error"] == "startup_failed" and "/home" not in r.text


def test_unexpected_errors_hide_details(make_client):
    class Broken:
        def predict(self, image, top_k=3):
            raise RuntimeError("internal /secret/path")

    r = post(make_client(Broken()), encode(garment_image()))
    assert r.status_code == 500 and r.json()["error"] == "internal_error"
    assert "/secret" not in r.text


def test_root_redirects_to_docs_without_a_demo(make_client):
    r = make_client().get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/docs"
