import pytest

from ai2pixelart.models import (
    ModelNotAvailable,
    available_local,
    discover_runs,
    local_checkpoint,
    resolve_checkpoint,
)


def _make_run(runs, name, *files):
    d = runs / name
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_bytes(b"stub")


def test_discover_prefers_safetensors(tmp_path):
    runs = tmp_path / "runs"
    _make_run(runs, "both", "best.ckpt", "best.safetensors")
    _make_run(runs, "legacy", "best.ckpt")
    _make_run(runs, "empty")  # no checkpoint -> not discovered

    found = discover_runs(runs)
    assert set(found) == {"both", "legacy"}
    assert found["both"].name == "best.safetensors"  # preferred
    assert found["legacy"].name == "best.ckpt"
    assert available_local(runs) == ["both", "legacy"]
    assert local_checkpoint("empty", runs) is None


def test_resolve_direct_path(tmp_path):
    f = tmp_path / "my.safetensors"
    f.write_bytes(b"x")
    assert resolve_checkpoint(str(f), runs_dir=tmp_path / "runs") == f


def test_resolve_unknown_lists_available(tmp_path):
    runs = tmp_path / "runs"
    _make_run(runs, "robust", "best.safetensors")
    with pytest.raises(ModelNotAvailable) as exc:
        resolve_checkpoint("nope", runs_dir=runs)
    assert "robust" in str(exc.value)


def test_resolve_falls_through_to_hub(tmp_path, monkeypatch):
    import huggingface_hub

    import ai2pixelart.models as M

    fetched = tmp_path / "robust-v11.safetensors"
    fetched.write_bytes(b"weights")
    calls = {}

    def fake_dl(repo_id, filename, revision=None):
        calls.update(repo_id=repo_id, filename=filename, revision=revision)
        return str(fetched)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setitem(
        M.REGISTRY, "robust",
        M.ModelSpec("robust", repo_id="x/y", filename="robust-v11.safetensors", revision="abc"),
    )
    # empty runs dir -> no local copy -> resolves via the hub
    got = resolve_checkpoint("robust", runs_dir=tmp_path / "norun")
    assert got == fetched
    assert calls == {"repo_id": "x/y", "filename": "robust-v11.safetensors", "revision": "abc"}


def test_local_run_wins_over_hub(tmp_path, monkeypatch):
    import huggingface_hub

    import ai2pixelart.models as M

    runs = tmp_path / "runs"
    _make_run(runs, "robust", "best.safetensors")

    def boom(**k):
        raise AssertionError("hub should not be hit when a local copy exists")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    monkeypatch.setitem(
        M.REGISTRY, "robust",
        M.ModelSpec("robust", repo_id="x/y", filename="robust-v11.safetensors"),
    )
    got = resolve_checkpoint("robust", runs_dir=runs)
    assert got == runs / "robust" / "best.safetensors"


def test_resolve_device():
    from ai2pixelart.models import resolve_device

    # explicit requests pass through unchanged (force CPU even with a GPU)
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    # auto resolves to one of the real devices
    assert resolve_device("auto") in ("cuda", "cpu")
    assert resolve_device(None) in ("cuda", "cpu")


def test_hub_sha256_mismatch_raises(tmp_path, monkeypatch):
    import huggingface_hub

    import ai2pixelart.models as M

    f = tmp_path / "m.safetensors"
    f.write_bytes(b"weights")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **k: str(f))
    spec = M.ModelSpec("robust", repo_id="x/y", filename="m.safetensors", sha256="deadbeef")
    with pytest.raises(ModelNotAvailable):
        M._download_from_hub(spec)


def test_safetensors_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from ai2pixelart.models import load_checkpoint, save_safetensors
    from ai2pixelart.nnmodel import PixelCleanNet

    net = PixelCleanNet(base=8)
    config = {"base": 8}
    path = tmp_path / "m.safetensors"
    save_safetensors(net.state_dict(), config, path, step=1234, val={"exact": 0.9})

    loaded = load_checkpoint(path, device="cpu")
    assert loaded.ckpt_step == 1234
    # weights identical after the format round-trip
    a = dict(net.state_dict())
    b = dict(loaded.state_dict())
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k].cpu())


def test_convert_ckpt_to_safetensors(tmp_path):
    torch = pytest.importorskip("torch")
    from ai2pixelart.models import convert_checkpoint, load_checkpoint
    from ai2pixelart.nnmodel import PixelCleanNet

    net = PixelCleanNet(base=8)
    src = tmp_path / "best.ckpt"
    torch.save({"model": net.state_dict(), "config": {"base": 8}, "step": 42}, src)

    out = convert_checkpoint(src)
    assert out == src.with_suffix(".safetensors")
    assert load_checkpoint(out, device="cpu").ckpt_step == 42
