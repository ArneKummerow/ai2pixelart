"""Palette files: the formats pixel-art tools exchange.

Reads and writes a (K, 3) uint8 palette as a GIMP `.gpl`, a JASC `.pal`, an
Adobe `.ase`, a paint.net `.txt`, a bare `.hex` list, or a PNG swatch strip
— the same set the web viewer's palette import/export offers, so a palette
travels between the CLI, the viewer and an external editor unchanged.

Readers additionally accept Microsoft RIFF `.pal` (the other thing a `.pal`
file can be) and any image, whose distinct colors in scan order are the
palette — that reads an exported strip back at any swatch scale, and lifts
a palette off a reference picture. Self-identifying content wins over the
extension (`ASEF`/`RIFF` magic, `JASC-PAL`/`GIMP Palette` header), so a
mislabeled file still works.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

# formats write_palette produces, as extensions
FORMATS = ("gpl", "pal", "ase", "txt", "hex", "png")
IMAGE_SUFFIXES = {".png", ".gif", ".bmp", ".webp", ".jpg", ".jpeg"}
_IMAGE_MAGIC = (b"\x89PNG", b"GIF8", b"BM", b"\xff\xd8")

# Forced palettes are capped here (the viewer's limit too), and a PNG strip
# wraps into rows at this width so no image dimension explodes.
MAX_COLORS = 256


class PaletteFileError(ValueError):
    """A palette file that cannot be read or written."""


def resolve_format(path: Path | str, fmt: str | None = None) -> str:
    """Output format from an explicit name or the file extension."""
    name = (fmt or Path(path).suffix.lstrip(".")).lower()
    if name not in FORMATS:
        raise PaletteFileError(
            f"unknown palette format: {name or '(none)'!r} (have {', '.join(FORMATS)})"
        )
    return name


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def write_palette(
    path: Path | str,
    palette: np.ndarray,
    fmt: str | None = None,
    scale: int = 1,
    name: str | None = None,
) -> Path:
    """Write a (K, 3) uint8 palette. `scale` is the PNG swatch size."""
    path = Path(path)
    kind = resolve_format(path, fmt)
    pal = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
    if not len(pal):
        raise PaletteFileError("cannot write an empty palette")
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "png":
        _write_png(path, pal, scale)
    elif kind == "ase":
        path.write_bytes(_ase_bytes(pal))
    else:
        writer = {"gpl": _gpl_text, "pal": _jasc_text, "txt": _paintnet_text, "hex": _hex_text}
        # newline="" keeps the per-format line endings exactly as written
        path.write_text(writer[kind](pal, name or path.stem), encoding="utf-8", newline="")
    return path


def _hexes(pal: np.ndarray) -> list[str]:
    return ["".join(f"{v:02X}" for v in color) for color in pal]


def _hex_text(pal: np.ndarray, name: str) -> str:
    return "\n".join(_hexes(pal)) + "\n"


def _gpl_text(pal: np.ndarray, name: str) -> str:
    rows = [f"{r:3d} {g:3d} {b:3d}\t{h}" for (r, g, b), h in zip(pal, _hexes(pal))]
    return "\n".join(["GIMP Palette", f"Name: {name}", "Columns: 16", "#", *rows]) + "\n"


def _paintnet_text(pal: np.ndarray, name: str) -> str:
    head = [f"; paint.net palette - {name}", "; one AARRGGBB color per line"]
    return "\r\n".join(head + [f"FF{h}" for h in _hexes(pal)]) + "\r\n"


def _jasc_text(pal: np.ndarray, name: str) -> str:
    """JASC-PAL (Paint Shop Pro): magic, format version, count, "r g b" lines."""
    rows = [f"{r} {g} {b}" for r, g, b in pal]
    return "\r\n".join(["JASC-PAL", "0100", str(len(pal)), *rows]) + "\r\n"


def _ase_bytes(pal: np.ndarray) -> bytes:
    """Adobe Swatch Exchange: header plus one big-endian color block each."""
    blocks = b"".join(_ase_block(h, c) for h, c in zip(_hexes(pal), pal))
    return b"ASEF" + struct.pack(">HHI", 1, 0, len(pal)) + blocks


def _ase_block(name: str, color: np.ndarray) -> bytes:
    body = (
        struct.pack(">H", len(name) + 1)          # UTF-16 units incl. the NUL
        + name.encode("utf-16-be")
        + b"\x00\x00"
        + b"RGB "
        + struct.pack(">fff", *(int(v) / 255 for v in color))
        + struct.pack(">H", 2)                    # color type: normal
    )
    return struct.pack(">HI", 1, len(body)) + body  # block type: color entry


def _write_png(path: Path, pal: np.ndarray, scale: int) -> None:
    """One square swatch per color, `scale` px a side, row-major."""
    from PIL import Image

    scale = max(1, int(scale))
    cols = min(len(pal), MAX_COLORS)
    rows = -(-len(pal) // cols)
    if rows * cols == len(pal):
        img = pal.reshape(rows, cols, 3)
    else:  # a wrapped palette's last row is short — pad it transparent
        rgba = np.zeros((rows * cols, 4), dtype=np.uint8)
        rgba[: len(pal), :3] = pal
        rgba[: len(pal), 3] = 255
        img = rgba.reshape(rows, cols, 4)
    if scale > 1:
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    Image.fromarray(img).save(path)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def read_palette(path: Path | str) -> np.ndarray:
    """Read any supported palette file (or image) as a (K, 3) uint8 palette."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as e:
        raise PaletteFileError(str(e)) from None
    if data[:4] == b"ASEF":
        pal = _read_ase(data)
    elif data[:4] == b"RIFF" and data[8:12] == b"PAL ":
        pal = _read_riff(data)
    elif _is_image(path, data):
        pal = _read_image(path)
    else:
        pal = _read_text(data.decode("utf-8", errors="replace"))
    pal = dedupe_exact(pal)
    if not len(pal):
        raise PaletteFileError(f"no colors found in {path}")
    return pal


def load_palette_spec(spec: str) -> np.ndarray:
    """A CLI palette argument: a palette file, or '#rrggbb,#rrggbb,...'."""
    from ai2pixelart.palette import parse_hex_palette

    if Path(spec).exists():  # a hex list never names an existing file
        return read_palette(spec)
    try:
        return parse_hex_palette(spec)
    except (ValueError, IndexError):
        raise PaletteFileError(f"not a palette file or '#rrggbb,...' list: {spec}") from None


def dedupe_exact(pal: np.ndarray) -> np.ndarray:
    """Drop repeated colors, keeping first-seen order."""
    pal = np.asarray(pal, dtype=np.uint8).reshape(-1, 3)
    if not len(pal):
        return pal
    _, first = np.unique(pal, axis=0, return_index=True)
    return pal[np.sort(first)]


def _is_image(path: Path, data: bytes) -> bool:
    return (
        path.suffix.lower() in IMAGE_SUFFIXES
        or data.startswith(_IMAGE_MAGIC)
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )


def _read_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        arr = np.array(im.convert("RGBA")).reshape(-1, 4)
    return arr[arr[:, 3] >= 8][:, :3]  # skip transparent padding


_TRIPLET = re.compile(r"^\s*(\d{1,3})[\s,]+(\d{1,3})[\s,]+(\d{1,3})")
_COMMENT = re.compile(r"^[ \t]*;.*$", re.MULTILINE)
_NOT_HEX = re.compile(r"[^0-9a-fA-F]+")


def _read_text(text: str) -> np.ndarray:
    """JASC .pal and GIMP .gpl list "r g b" per line; everything else
    (paint.net AARRGGBB, .hex, a pasted "#rrggbb,...") is scanned for hex
    tokens, falling back to bare triplets."""
    lines = text.lstrip("\ufeff").splitlines()
    head = lines[0].strip().lower() if lines else ""
    if head.startswith(("jasc-pal", "gimp palette")):
        return _triplet_lines(lines)
    hexes = _hex_tokens(text)
    return hexes if len(hexes) else _triplet_lines(lines)


def _triplet_lines(lines: list[str]) -> np.ndarray:
    found = [m.groups() for m in (_TRIPLET.match(line) for line in lines) if m]
    return np.clip(np.array(found, dtype=int).reshape(-1, 3), 0, 255).astype(np.uint8)


def _hex_tokens(text: str) -> np.ndarray:
    found = []
    for token in _NOT_HEX.split(_COMMENT.sub("", text)):
        # 8 digits is paint.net's AARRGGBB; the alpha is dropped
        digits = token[2:] if len(token) == 8 else token if len(token) == 6 else None
        if digits:
            found.append([int(digits[i : i + 2], 16) for i in (0, 2, 4)])
    return np.array(found, dtype=np.uint8).reshape(-1, 3)


def _read_ase(data: bytes) -> np.ndarray:
    (blocks,) = struct.unpack_from(">I", data, 8)
    found: list[list[int]] = []
    offset = 12
    for _ in range(blocks):
        if offset + 6 > len(data):
            break
        kind, size = struct.unpack_from(">HI", data, offset)
        if offset + 6 + size > len(data):
            break  # truncated file: keep what parsed
        if kind == 0x0001:  # color entry; groups (0xC001/0xC002) are skipped
            try:
                found.append(_ase_color(data, offset + 6))
            except (struct.error, ValueError):
                pass  # unreadable entry, keep going
        offset += 6 + size
    return np.array([c for c in found if c], dtype=np.uint8).reshape(-1, 3)


def _ase_color(data: bytes, pos: int) -> list[int] | None:
    (chars,) = struct.unpack_from(">H", data, pos)
    pos += 2 + 2 * chars  # past the UTF-16 name
    model, pos = data[pos : pos + 4], pos + 4
    if model == b"RGB ":
        return [_u8(v * 255) for v in struct.unpack_from(">3f", data, pos)]
    if model == b"CMYK":
        c, m, y, k = struct.unpack_from(">4f", data, pos)
        return [_u8(255 * (1 - v) * (1 - k)) for v in (c, m, y)]
    if model.strip().lower() == b"gray":
        (gray,) = struct.unpack_from(">f", data, pos)
        return [_u8(gray * 255)] * 3
    if model == b"LAB ":
        light, a, b = struct.unpack_from(">3f", data, pos)
        return _lab_d50_to_rgb(light * 100, a, b)
    return None


def _read_riff(data: bytes) -> np.ndarray:
    """Microsoft RIFF .pal: chunks, entries in "data" as r, g, b, flags."""
    offset = 12
    while offset + 8 <= len(data):
        chunk = data[offset : offset + 4]
        (size,) = struct.unpack_from("<I", data, offset + 4)
        if chunk == b"data":
            if offset + 12 > len(data):
                break
            (count,) = struct.unpack_from("<H", data, offset + 10)
            entries = data[offset + 12 : offset + 12 + 4 * count]
            entries = entries[: 4 * (len(entries) // 4)]
            return np.frombuffer(entries, dtype=np.uint8).reshape(-1, 4)[:, :3]
        offset += 8 + size + (size & 1)
    return np.zeros((0, 3), dtype=np.uint8)


def _u8(value: float) -> int:
    return int(min(255, max(0, round(value))))


# sRGB matrix for a D50 white point (Bradford-adapted). ASE stores Lab
# against D50, the way Photoshop writes it — palette.lab_to_rgb is D65.
_XYZ_D50_TO_RGB = np.array(
    [
        [3.1338561, -1.6168667, -0.4906146],
        [-0.9787684, 1.9161415, 0.0334540],
        [0.0719453, -0.2289914, 1.4052427],
    ]
)


def _lab_d50_to_rgb(light: float, a: float, b: float) -> list[int]:
    fy = (light + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    xyz = np.where(f**3 > 0.008856, f**3, (116 * f - 16) / 903.3)
    xyz = xyz * (0.9642, 1.0, 0.8249)
    linear = np.clip(_XYZ_D50_TO_RGB @ xyz, 0.0, None)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return [_u8(v * 255) for v in srgb]
