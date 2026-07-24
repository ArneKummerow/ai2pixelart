import base64
import io
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

from ai2pixelart.corrupt import upscale
from ai2pixelart.webapp import (
    AppError,
    ViewerApp,
    clean_with_info,
    collect_sources,
    make_server,
)


@pytest.fixture
def workspace(tmp_path, sprite):
    """A folder with one pseudo-pixel-art image, as `view` would serve it."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    Image.fromarray(upscale(sprite, 3.0, interp="nearest")).save(ws / "a.png")
    (ws / "notes.txt").write_text("ignored")
    return ws


def test_collect_sources_filters_images(workspace):
    assert [p.name for p in collect_sources(workspace)] == ["a.png"]


def test_clean_with_info_validates_params(sprite):
    big = upscale(sprite, 3.0, interp="nearest")
    png, info = clean_with_info(big, {"max_colors": 8, "merge_de": 6.0, "denoise": False})
    assert info["palette_size"] <= 8
    assert info["input_colors"] >= info["palette_size"]
    assert Image.open(io.BytesIO(png)).size == (24, 24)

    with pytest.raises(AppError):
        clean_with_info(big, {"bogus": 1})
    with pytest.raises(AppError):
        clean_with_info(big, {"merge_de": 999})
    with pytest.raises(AppError):
        clean_with_info(big, {"max_colors": "not-a-number"})
    with pytest.raises(AppError):
        clean_with_info(big, {"granularity": 8})


def test_run_clean_caches_in_memory_only(workspace):
    app = ViewerApp(workspace)
    before = sorted(p.name for p in workspace.iterdir())

    first = app.run_clean("a.png", {"max_colors": 6})
    assert first["image"].startswith("data:image/png;base64,")
    assert first["info"]["palette_size"] <= 6
    assert len(app.cache) == 1
    assert app.run_clean("a.png", {"max_colors": 6}) is first  # cache hit
    assert app.run_clean("a.png", {"max_colors": 5}) is not first  # params in key

    # same params, changed file content -> recompute (key = content hash)
    img = np.array(Image.open(workspace / "a.png"))
    Image.fromarray(255 - img).save(workspace / "a.png")
    assert app.run_clean("a.png", {"max_colors": 6}) is not first

    # nothing precomputed or cached on disk, and a restart starts cold
    assert sorted(p.name for p in workspace.iterdir()) == before
    assert ViewerApp(workspace).cache == {}


def test_image_path_rejects_escapes(workspace):
    app = ViewerApp(workspace)
    for bad in ["../a.png", "notes.txt", ".hidden.png", "", "missing.png"]:
        with pytest.raises(AppError):
            app.image_path(bad)


def test_upload_sanitizes_and_never_overwrites(workspace, sprite):
    app = ViewerApp(workspace)
    buf = io.BytesIO()
    Image.fromarray(sprite).save(buf, format="PNG")
    data = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    assert app.save_upload("../we ird$.png", data)["name"] == "we_ird_.png"
    assert app.save_upload("we_ird_.png", data)["name"] == "we_ird_.png"  # same bytes: reuse
    assert app.save_upload("a.png", data)["name"] == "a-2.png"  # name clash: suffix

    with pytest.raises(AppError):
        app.save_upload("a.txt", data)
    with pytest.raises(AppError):
        app.save_upload("a.png", "not-base64!!")
    with pytest.raises(AppError):
        app.save_upload("a.png", base64.b64encode(b"not an image").decode())


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req).read())


def test_api_server_roundtrip(workspace):
    server = make_server(workspace, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert "workspace" in urllib.request.urlopen(f"{base}/").read().decode()

        listing = json.loads(urllib.request.urlopen(f"{base}/api/images").read())
        assert [i["name"] for i in listing["images"]] == ["a.png"]
        assert {p["key"] for p in listing["presets"]} >= {"classical", "classical16"}

        raw = urllib.request.urlopen(f"{base}/img/a.png").read()
        assert raw == (workspace / "a.png").read_bytes()
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{base}/img/../notes.txt")
        assert exc.value.code == 404

        payload = _post(f"{base}/api/clean", {"image": "a.png", "params": {"max_colors": 6}})
        assert payload["info"]["palette_size"] <= 6
        assert payload["image"].startswith("data:image/png;base64,")

        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(f"{base}/api/clean", {"image": "nope.png"})
        assert exc.value.code == 400

        up = _post(f"{base}/api/upload", {"name": "b.png", "data": _tiny_png_b64()})
        assert up["name"] == "b.png"
        names = [i["name"] for i in json.loads(urllib.request.urlopen(f"{base}/api/images").read())["images"]]
        assert names == ["a.png", "b.png"]
    finally:
        server.shutdown()


def _tiny_png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 200, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_method_param_validation(sprite):
    big = upscale(sprite, 3.0, interp="nearest")
    with pytest.raises(AppError):  # invalid method value
        clean_with_info(big, {"method": "quantum"})
    with pytest.raises(AppError):  # nn requested but no model configured
        clean_with_info(big, {"method": "nn"})
    # explicit classical is the default path
    png, info = clean_with_info(big, {"method": "classical"})
    assert info["method"] == "classical"


def test_nn_variants_and_palette_param(workspace, sprite):
    from ai2pixelart.webapp import ViewerApp, _parse_params

    assert ViewerApp(workspace).ckpts == {}
    app = ViewerApp(workspace, ckpts={"vae": "runs/vae/best.ckpt"})
    assert sorted(app.ckpts) == ["vae"]
    with pytest.raises(AppError):
        app.nn_bits("img2img")  # unknown variant

    parsed = _parse_params({"palette": "#102030,#f0f0f0", "model": "vae"})
    assert parsed["palette"].shape == (2, 3)
    assert parsed["model"] == "vae"
    with pytest.raises(AppError):
        _parse_params({"palette": "#zzz"})
    with pytest.raises(AppError):
        _parse_params({"palette": "#102030"})  # fewer than 2 colors

    # explicit palette drives the classical path end to end
    big = upscale(sprite, 3.0, interp="nearest")
    png, info = clean_with_info(big, {"palette": "#1e2234,#5ea740,#f0f0f0"})
    assert info["palette_size"] == 3


def test_manual_pitch_yields_valid_json(sprite):
    """A manual pitch has no detection score; the response must still be
    strict JSON (bare NaN broke browsers' JSON.parse)."""
    big = upscale(sprite, 3.0, interp="nearest")
    png, info = clean_with_info(big, {"pitch": 3.0})
    json.dumps(info, allow_nan=False)  # must not raise
    assert info["grid"]["z_y"] is None
    assert info["grid"]["pitch_y"] == 3.0


def test_ckpt_scan_discovers_new_runs_live(workspace, tmp_path):
    """A checkpoint that appears while the server runs (a training run
    finishing) must show up in /api/images without a restart."""
    runs = tmp_path / "runs"
    (runs / "va").mkdir(parents=True)
    (runs / "va" / "best.ckpt").write_bytes(b"stub")
    server = make_server(workspace, port=0, ckpt_scan=runs)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        d = json.loads(urllib.request.urlopen(f"{base}/api/images").read())
        assert d["nn_models"] == ["va"]
        (runs / "v9").mkdir()
        (runs / "v9" / "best.ckpt").write_bytes(b"stub")
        d = json.loads(urllib.request.urlopen(f"{base}/api/images").read())
        assert d["nn_models"] == ["v9", "va"] or set(d["nn_models"]) == {"va", "v9"}
        assert any(p["key"] == "nn:v9" for p in d["presets"])
    finally:
        server.shutdown()


def test_nn_model_hot_reloads_on_ckpt_change(workspace, tmp_path, monkeypatch):
    """An overwritten best.ckpt (training in progress) must trigger a model
    reload and change the run-cache version — stale early-training weights
    were served forever otherwise (user report: 'v4 colors completely off')."""
    import os

    import ai2pixelart.nninfer as nninfer

    ck = tmp_path / "runs" / "vx"
    ck.mkdir(parents=True)
    (ck / "best.ckpt").write_bytes(b"w1")
    app = ViewerApp(workspace, ckpt_scan=tmp_path / "runs")

    loads = []
    monkeypatch.setattr(nninfer, "load_checkpoint", lambda p, device: loads.append(str(p)) or object())
    app.nn_bits("vx")
    app.nn_bits("vx")
    assert len(loads) == 1  # unchanged file: cached
    v1 = app.ckpt_version("vx")
    os.utime(ck / "best.ckpt", ns=(1, 1))  # training overwrote the file
    app.nn_bits("vx")
    assert len(loads) == 2  # reloaded
    assert app.ckpt_version("vx") != v1  # cache key version moved
