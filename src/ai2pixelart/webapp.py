"""Interactive workspace viewer: live cleanup over a folder of images.

`serve` hosts a folder of source images plus a small JSON API. Nothing is
precomputed and no outputs ever touch the disk: every cleaned image is
computed on demand (`POST /api/clean`) with user-chosen parameters and kept
in an in-memory cache keyed by (input file content, parameters), so the
cache lives exactly as long as the process. `POST /api/upload` adds a new
source image to the workspace folder (the one write the app performs).
Standard library only.

There is no ground truth for real AI images, so the quality signal is the
reconstruction residual: paint the cleaned cells back onto the input's pixel
grid and measure the ΔE against the input. It quantifies how much of the
input the cleanup could not represent.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import re
import threading
import time
import urllib.parse
from collections import OrderedDict
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image

from ai2pixelart.palette import delta_e, rgb_to_lab
from ai2pixelart.pipeline import CleanResult, clean

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Named parameter presets offered by the viewer; new entries appear in the
# panel preset dropdown automatically.
APPROACHES: dict[str, dict] = {
    "classical": {"label": "Simple", "params": {}},
}

# name -> (parser, min, max); bounds keep API-triggered runs sane
PARAM_SPEC = {
    "merge_de": (float, 0.5, 20.0),
    "max_colors": (int, 2, 256),
    "keep_frac": (float, 0.1, 1.0),
    "pitch": (float, 1.6, 64.0),
    "denoise": (bool, None, None),
    "smooth": (bool, None, None),
    "absorb_de": (float, 0.0, 40.0),
    "absorb_frac": (float, 0.0, 0.3),
    "granularity": (float, 0.25, 4.0),
    "leash": (float, 1.0, 40.0),  # nn only: margin over the nearest entry (relative)
    "consensus": (bool, None, None),  # global flat-color consensus (pipeline.consensus_indices)
    "method": (str, None, None),  # "classical" (default) | "nn"
    "model": (str, None, None),  # nn variant name (a served checkpoint)
    "palette": (str, None, None),  # explicit '#rrggbb,...' output palette
}
METHODS = {"classical", "nn"}

CACHE_MAX_ENTRIES = 256
UPLOAD_MAX_BYTES = 32 * 1024 * 1024

# Neural proposal cache: the classical palette/grid proposal + the net's raw
# logits depend on these params but NOT on leash/smooth/consensus, so they
# are reused while the user tweaks those. Logits are large (kept on the
# compute device), so the cache is a tiny LRU.
PROPOSAL_PARAM_KEYS = (
    "max_colors", "palette", "merge_de", "pitch", "granularity",
    "denoise", "keep_frac", "absorb_de", "absorb_frac",
)
PROPOSAL_CACHE_MAX = 2


class AppError(ValueError):
    """Invalid API request (reported as HTTP 400)."""


def collect_sources(src: Path) -> list[Path]:
    """A file as-is, or all images directly inside a directory."""
    if src.is_file():
        return [src]
    return sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _reconstruction(result: CleanResult) -> np.ndarray:
    """Paint the cleaned cells back onto the input pixel grid."""
    if result.grid is None:
        return result.image
    # floor(e + .5), not np.round: bankers' rounding can give two edges the
    # same pixel and a zero-width cell
    ye = np.floor(result.grid.y.edges + 0.5).astype(int)
    xe = np.floor(result.grid.x.edges + 0.5).astype(int)
    return np.repeat(np.repeat(result.image, np.diff(ye), axis=0), np.diff(xe), axis=1)


def _grid_info(result: CleanResult) -> dict | None:
    g = result.grid
    if g is None:
        return None

    def z(score: float):
        # a manually-set pitch has no detection score (NaN) — and bare NaN
        # in a JSON body is invalid for browsers' JSON.parse
        return round(float(score), 1) if np.isfinite(score) else None

    return {
        "pitch_y": round(g.y.pitch, 3),
        "pitch_x": round(g.x.pitch, 3),
        "z_y": z(g.y.score),
        "z_x": z(g.x.score),
    }


def _approach_info(result: CleanResult, img: np.ndarray, seconds: float) -> dict:
    residual = delta_e(rgb_to_lab(_reconstruction(result)), rgb_to_lab(img))
    # assignment fidelity: how far each sampled cell is from the palette
    # entry it received. Flags color hallucination (a net assigning a cell
    # to a plausible-but-wrong entry) that the residual alone can hide.
    fit = delta_e(rgb_to_lab(result.raw_cells), rgb_to_lab(result.palette[result.indices]))
    return {
        "out_w": int(result.image.shape[1]),
        "out_h": int(result.image.shape[0]),
        "cell_fit_mean": round(float(fit.mean()), 2),
        "palette_size": int(len(result.palette)),
        "input_colors": int(len(np.unique(img.reshape(-1, 3), axis=0))),
        "grid": _grid_info(result),
        "residual_mean_de": round(float(residual.mean()), 2),
        "residual_p95_de": round(float(np.percentile(residual, 95)), 2),
        "seconds": round(seconds, 2),
    }


def _parse_params(raw: dict | None) -> dict:
    parsed: dict = {}
    for key, value in (raw or {}).items():
        if key not in PARAM_SPEC:
            raise AppError(f"unknown parameter: {key}")
        if value is None or value == "":
            continue
        kind, lo, hi = PARAM_SPEC[key]
        if kind is bool:
            parsed[key] = bool(value)
            continue
        if kind is str:
            if key == "method" and value not in METHODS:
                raise AppError(f"parameter method must be one of {sorted(METHODS)}")
            if key == "palette":
                from ai2pixelart.palette import parse_hex_palette

                try:
                    pal = parse_hex_palette(str(value))
                except (ValueError, IndexError):
                    raise AppError("palette must be '#rrggbb,#rrggbb,...'") from None
                if not 2 <= len(pal) <= 256:
                    raise AppError("palette must have 2-256 colors")
                parsed[key] = pal
                continue
            parsed[key] = str(value)
            continue
        try:
            value = kind(value)
        except (TypeError, ValueError):
            raise AppError(f"parameter {key} must be {kind.__name__}") from None
        if not lo <= value <= hi:
            raise AppError(f"parameter {key} out of range [{lo}, {hi}]")
        parsed[key] = value
    if "pitch" in parsed:
        parsed["pitch"] = (parsed["pitch"], parsed["pitch"])
    return parsed


def clean_with_info(
    img: np.ndarray, params: dict | None, nn_model=None, nn_device: str = "cpu"
) -> tuple[bytes, dict]:
    """Run one cleanup method with API parameters -> (png bytes, info).

    method "classical" (default) runs the parametrized classical pipeline;
    "nn" runs the restoration net on the classical palette/grid proposal
    (proposal params — max_colors, palette, pitch, granularity — apply
    there too; the palette defaults to its natural proposal size).
    """
    parsed = _parse_params(params)
    method = parsed.pop("method", "classical")
    parsed.pop("model", None)  # variant choice is resolved by the caller
    t0 = time.perf_counter()
    if method == "nn":
        if nn_model is None:
            raise AppError("neural model not available — start `viewer` with --model")
        from ai2pixelart.nninfer import nn_clean_image

        # every classical param shapes the palette/grid PROPOSAL the net
        # classifies against, so they all pass through
        result = nn_clean_image(img, nn_model, device=nn_device, **parsed)
    else:
        parsed.pop("leash", None)  # nn-only concept
        result = clean(img, **parsed)
    info = _approach_info(result, img, time.perf_counter() - t0)
    info["method"] = method
    if method == "nn":
        # in-training checkpoints are served live; make that visible
        info["model_step"] = getattr(nn_model, "ckpt_step", None)
    buf = io.BytesIO()
    Image.fromarray(result.image).save(buf, format="PNG")
    return buf.getvalue(), info


_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class ViewerApp:
    """State + compute behind the API: workspace folder, run cache, lock."""

    def __init__(
        self,
        directory: Path,
        ckpts: dict[str, Path] | None = None,
        ckpt_scan: Path | None = None,
        device: str = "auto",
    ):
        self.directory = Path(directory).resolve()
        self._device_pref = device  # 'auto' | 'cuda' | 'cpu'
        # variant name -> checkpoint path (e.g. "robust" -> runs/robust/best.safetensors);
        # each becomes a "Neural <Name>" preset, so future corruption
        # tracks (img2img, ...) ship as additional named checkpoints
        self.ckpts = {k: Path(v) for k, v in (ckpts or {}).items()}
        # with ckpt_scan set, new runs are picked up while the server lives
        # (a training run finishing appears without a restart)
        self.ckpt_scan = Path(ckpt_scan) if ckpt_scan else None
        self._nn_models: dict[str, object] = {}  # loaded lazily per variant
        # (sha1, model, version, proposal-params) -> {palette, grid, raw_cells,
        # logits}; a tiny LRU so leash/smooth/consensus tweaks skip the net
        self._proposals: OrderedDict = OrderedDict()
        self.lock = threading.Lock()  # one heavy compute at a time
        # (input content sha1, params json) -> payload; lives only as long
        # as the process — restarts start cold by design.
        self.cache: dict = {}

    def local_ckpts(self) -> dict[str, Path]:
        """Explicitly-pinned (--model) plus locally-discovered run
        checkpoints (name -> path). With ckpt_scan set, a training run that
        finishes appears here without a restart."""
        if self.ckpt_scan and self.ckpt_scan.is_dir():
            from ai2pixelart.models import discover_runs

            for name, p in discover_runs(self.ckpt_scan).items():
                self.ckpts.setdefault(name, p)
        return self.ckpts

    def served_models(self) -> list[str]:
        """Model names offered in the viewer: local runs (for development)
        plus registered hub models, local winning on a name collision."""
        from ai2pixelart.models import REGISTRY

        names = set(self.local_ckpts())
        names |= {n for n, s in REGISTRY.items() if s.repo_id}
        return sorted(names)

    def ckpt_version(self, name: str | None) -> str:
        """Identity of the checkpoint a model name resolves to. Part of the
        run-cache key so retrained (local) or updated (hub) weights don't
        serve stale results. Never triggers a download — a hub model's
        version is its pinned revision, known from the registry."""
        names = self.served_models()
        if not names:
            return ""
        name = name or names[0]
        local = self.local_ckpts()
        if name in local and local[name].exists():
            return f"{name}:{local[name].stat().st_mtime_ns}"  # mtime -> reload on retrain
        from ai2pixelart.models import REGISTRY

        spec = REGISTRY.get(name)
        if spec and spec.repo_id:
            return f"{name}:{spec.revision or spec.filename}"  # stable
        return str(name)

    def nn_bits(self, name: str | None):
        """(model, device) for a net variant, loading it on first use and
        RELOADING when its checkpoint changes. A local run reloads when its
        best.* file's mtime moves (in-progress training); a registered hub
        model is downloaded on first use and cached by its pinned revision."""
        names = self.served_models()
        if not names:
            return None, "cpu"
        name = name or names[0]
        local = self.local_ckpts()
        if name in local:
            path = local[name]
            key = f"mtime:{path.stat().st_mtime_ns}"
        else:
            from ai2pixelart.models import REGISTRY, ModelNotAvailable, download_model

            spec = REGISTRY.get(name)
            if not (spec and spec.repo_id):
                raise AppError(f"unknown neural model: {name!r} (have {names})")
            try:
                path = download_model(name)  # cached after first fetch
            except ModelNotAvailable as e:
                raise AppError(str(e))
            key = f"rev:{spec.revision or spec.filename}"

        cached = self._nn_models.get(name)
        if cached is None or cached[0] != key:
            from ai2pixelart.models import resolve_device
            from ai2pixelart.nninfer import load_checkpoint

            self._nn_device = resolve_device(self._device_pref)
            self._nn_models[name] = (key, load_checkpoint(path, device=self._nn_device))
        return self._nn_models[name][1], self._nn_device

    def image_path(self, name: str) -> Path:
        """Resolve a workspace image by file name; reject anything else."""
        if not name or name != Path(name).name or name.startswith("."):
            raise AppError(f"invalid image name: {name!r}")
        if Path(name).suffix.lower() not in IMAGE_EXTS:
            raise AppError(f"not an image name: {name!r}")
        path = self.directory / name
        if not path.is_file():
            raise AppError(f"unknown image: {name}")
        return path

    def list_images(self) -> list[dict]:
        out = []
        for path in collect_sources(self.directory):
            try:
                with Image.open(path) as im:  # header only, cheap
                    w, h = im.size
            except OSError:
                continue
            out.append({"name": path.name, "width": w, "height": h})
        return out

    def run_clean(self, name: str, params: dict | None) -> dict:
        raw = self.image_path(name).read_bytes()
        params = params or {}
        is_nn = params.get("method") == "nn"
        sha1 = hashlib.sha1(raw).hexdigest()
        version = self.ckpt_version(params.get("model")) if is_nn else ""
        key = (sha1, json.dumps(params, sort_keys=True) + version)
        if key in self.cache:
            return self.cache[key]
        img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        with self.lock:
            if is_nn:
                png, info = self._nn_clean_with_info(img, params, sha1, version)
            else:
                png, info = clean_with_info(img, params)
        payload = {
            "image": "data:image/png;base64," + base64.b64encode(png).decode(),
            "info": info,
            "params": params or {},
        }
        if len(self.cache) >= CACHE_MAX_ENTRIES:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = payload
        return payload

    def _nn_clean_with_info(self, img, params: dict, sha1: str, version: str):
        """Neural clean reusing a cached proposal. The classical proposal and
        the net's logits depend only on PROPOSAL_PARAM_KEYS (+ model), so a
        leash/smooth/consensus change re-runs just the cheap finalize; the
        reported time reflects that."""
        from ai2pixelart.nninfer import nn_finalize, nn_propose

        model, device = self.nn_bits(params.get("model"))
        if model is None:
            raise AppError("neural model not available — start `viewer` with --model")
        parsed = _parse_params(params)
        parsed.pop("method", None)
        parsed.pop("model", None)
        leash = parsed.pop("leash", None)
        consensus = parsed.pop("consensus", False)
        smooth = parsed.get("smooth", True)

        pkey = (sha1, params.get("model") or "", version, json.dumps(
            {k: params[k] for k in PROPOSAL_PARAM_KEYS if k in params}, sort_keys=True))
        t0 = time.perf_counter()
        prop = self._proposals.get(pkey)
        if prop is None:
            prop = nn_propose(img, model, device, **parsed)
            self._proposals[pkey] = prop
            while len(self._proposals) > PROPOSAL_CACHE_MAX:
                self._proposals.popitem(last=False)
        else:
            self._proposals.move_to_end(pkey)
        result = nn_finalize(img, prop, device, leash=leash, smooth=smooth, consensus=consensus)

        info = _approach_info(result, img, time.perf_counter() - t0)
        info["method"] = "nn"
        info["model_step"] = getattr(model, "ckpt_step", None)
        buf = io.BytesIO()
        Image.fromarray(result.image).save(buf, format="PNG")
        return buf.getvalue(), info

    def save_upload(self, name: str, data: str) -> dict:
        """Store a dropped/picked image in the workspace folder."""
        safe = _UNSAFE_CHARS.sub("_", Path(name or "").name).lstrip(".")
        stem, ext = Path(safe).stem, Path(safe).suffix.lower()
        if not stem or ext not in IMAGE_EXTS:
            raise AppError(f"unsupported image name: {name!r}")
        payload = data.split(",", 1)[-1]  # tolerate a data: URL prefix
        try:
            raw = base64.b64decode(payload, validate=True)
        except (ValueError, TypeError):
            raise AppError("upload data must be base64") from None
        if len(raw) > UPLOAD_MAX_BYTES:
            raise AppError("upload too large")
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im.verify()
        except Exception:
            raise AppError("upload is not a decodable image") from None
        target = self.directory / f"{stem}{ext}"
        n = 2
        while target.exists() and target.read_bytes() != raw:
            target = self.directory / f"{stem}-{n}{ext}"
            n += 1
        target.write_bytes(raw)
        return {"name": target.name}


def make_server(directory: Path, port: int = 8412, ckpts: dict[str, Path] | None = None, ckpt_scan: Path | None = None, device: str = "auto"):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, data: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status: int, payload: dict) -> None:
            try:
                data = json.dumps(payload, allow_nan=False).encode()
            except ValueError:  # fail loudly instead of emitting invalid JSON
                status = 500
                data = json.dumps({"error": "non-finite number in response"}).encode()
            self._reply(status, data, "application/json")

        def do_GET(self):
            app = self.server.app
            path = urllib.parse.unquote(self.path.split("?", 1)[0])
            if path == "/":
                html = resources.files("ai2pixelart").joinpath("viewer.html").read_bytes()
                self._reply(200, html, "text/html; charset=utf-8")
            elif path == "/api/images":
                presets = [
                    {"key": k, "label": v["label"], "params": v["params"]}
                    for k, v in APPROACHES.items()
                ]
                names = app.served_models()
                for name in names:
                    presets.append({
                        "key": f"nn:{name}",
                        "label": f"Neural {name.title()}",
                        # leash 8: only near-ties of the pixel's nearest
                        # entry are choosable — guards large-palette
                        # extrapolation (orange can never become green)
                        "params": {"method": "nn", "model": name, "leash": 8},
                    })
                self._json(
                    200,
                    {
                        "images": app.list_images(),
                        "presets": presets,
                        "nn_models": names,
                    },
                )
            elif path.startswith("/img/"):
                try:
                    file = app.image_path(path[len("/img/") :])
                except AppError:
                    self.send_error(404)
                    return
                ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                self._reply(200, file.read_bytes(), ctype)
            else:
                self.send_error(404)

        def do_POST(self):
            app = self.server.app
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                req = json.loads(body)
                if self.path == "/api/clean":
                    payload = app.run_clean(req.get("image"), req.get("params"))
                elif self.path == "/api/upload":
                    payload = app.save_upload(req.get("name"), req.get("data") or "")
                else:
                    self.send_error(404)
                    return
                status = 200
            except AppError as e:
                payload, status = {"error": str(e)}, 400
            except Exception as e:  # keep the server alive on compute bugs
                payload, status = {"error": repr(e)}, 500
            self._json(status, payload)

        def log_message(self, *args):
            pass  # keep the terminal quiet

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.app = ViewerApp(directory, ckpts=ckpts, ckpt_scan=ckpt_scan, device=device)
    return server


def serve(directory: Path, port: int = 8412, ckpts: dict[str, Path] | None = None, ckpt_scan: Path | None = None, device: str = "auto") -> None:
    server = make_server(directory, port, ckpts=ckpts, ckpt_scan=ckpt_scan, device=device)
    nn = f", neural models: {sorted(ckpts)}" if ckpts else ""
    print(f"serving {directory} at http://127.0.0.1:{port}/  (Ctrl-C to stop{nn})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
