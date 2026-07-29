import struct

import numpy as np
import pytest
from PIL import Image

from ai2pixelart.palettefile import (
    FORMATS,
    PaletteFileError,
    load_palette_spec,
    read_palette,
    write_palette,
)

PAL = np.array(
    [[30, 34, 52], [94, 167, 64], [240, 240, 240], [0, 0, 0], [255, 255, 255], [1, 2, 3]],
    dtype=np.uint8,
)


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_round_trips(tmp_path, fmt):
    """What each writer produces, its reader reads back exactly — the
    property that lets a palette travel between tools."""
    path = write_palette(tmp_path / f"p.{fmt}", PAL)
    assert np.array_equal(read_palette(path), PAL)


@pytest.mark.parametrize("scale", [1, 8, 32])
def test_png_strip_is_one_swatch_per_color(tmp_path, scale):
    path = write_palette(tmp_path / "p.png", PAL, scale=scale)
    with Image.open(path) as im:
        assert im.size == (len(PAL) * scale, scale)
    assert np.array_equal(read_palette(path), PAL)


def test_png_wraps_wide_palettes(tmp_path):
    """Past 256 colors the strip wraps into rows (a 100k-px-wide PNG is not
    a file), and the transparent padding of the last row is not a color."""
    n = np.arange(300)
    big = np.stack([n // 256, n % 256, np.full(300, 7)], axis=1).astype(np.uint8)
    path = write_palette(tmp_path / "big.png", big, scale=2)
    with Image.open(path) as im:
        assert im.size == (256 * 2, 2 * 2)
    assert np.array_equal(read_palette(path), big)


def test_written_text_is_what_the_tools_expect(tmp_path):
    gpl = write_palette(tmp_path / "n.gpl", PAL).read_text()
    assert gpl.startswith("GIMP Palette\nName: n\nColumns: 16\n#\n 30  34  52\t1E2234\n")

    jasc = write_palette(tmp_path / "p.pal", PAL).read_bytes()
    assert jasc.startswith(b"JASC-PAL\r\n0100\r\n6\r\n30 34 52\r\n")

    pdn = write_palette(tmp_path / "p.txt", PAL).read_bytes()
    assert pdn.startswith(b"; paint.net") and b"\r\nFF1E2234\r\nFF5EA740\r\n" in pdn

    assert write_palette(tmp_path / "p.hex", PAL).read_text() == (
        "1E2234\n5EA740\nF0F0F0\n000000\nFFFFFF\n010203\n"
    )

    ase = write_palette(tmp_path / "p.ase", PAL).read_bytes()
    assert ase[:4] == b"ASEF" and struct.unpack_from(">I", ase, 8)[0] == len(PAL)
    assert len(ase) == 12 + len(PAL) * 40  # header + one 40-byte block each


@pytest.mark.parametrize(
    "name,text",
    [
        (  # GIMP, with the comment lines it writes in the wild
            "t.gpl",
            "GIMP Palette\nName: Test\nColumns: 4\n# a comment\n"
            " 30  34  52\tDark\n 94 167  64\tGreen\n240 240 240\tLight\n",
        ),
        ("t.pal", "JASC-PAL\r\n0100\r\n3\r\n30 34 52\r\n94 167 64\r\n240 240 240\r\n"),
        (  # paint.net: its own header comments must not become colors
            "t.txt",
            ";paint.net Palette File\n;e.g. FFFF0000 is opaque red\n"
            "FF1E2234\nFF5EA740\nFFF0F0F0\n",
        ),
        ("t.hex", "1E2234\n5EA740\nF0F0F0\n"),
        ("pasted.txt", "#1e2234, #5ea740 ,#f0f0f0"),
        ("bare.txt", "30 34 52\n94 167 64\n240 240 240\n"),
    ],
)
def test_reads_files_written_by_other_tools(tmp_path, name, text):
    (tmp_path / name).write_text(text)
    assert np.array_equal(read_palette(tmp_path / name), PAL[:3])


def test_reads_microsoft_riff_pal(tmp_path):
    """A .pal is either JASC text or this — told apart by content."""
    entries = b"".join(bytes([*c, 0]) for c in PAL[:3])
    data = (
        b"RIFF" + struct.pack("<I", 4 + 8 + 4 + len(entries)) + b"PAL "
        + b"data" + struct.pack("<I", 4 + len(entries))
        + struct.pack("<HH", 0x0300, 3) + entries
    )
    (tmp_path / "riff.pal").write_bytes(data)
    assert np.array_equal(read_palette(tmp_path / "riff.pal"), PAL[:3])


def _ase_one(model: bytes, values: tuple[float, ...]) -> bytes:
    body = (
        struct.pack(">H", 2) + "A".encode("utf-16-be") + b"\x00\x00"
        + model + struct.pack(f">{len(values)}f", *values) + struct.pack(">H", 2)
    )
    return b"ASEF" + struct.pack(">HHI", 1, 0, 1) + struct.pack(">HI", 1, len(body)) + body


@pytest.mark.parametrize(
    "model,values,expected",
    [
        (b"RGB ", (1.0, 0.0, 0.0), [255, 0, 0]),
        (b"CMYK", (0.0, 1.0, 1.0, 0.0), [255, 0, 0]),
        (b"Gray", (0.5,), [128, 128, 128]),
        (b"LAB ", (1.0, 0.0, 0.0), [255, 255, 255]),
        (b"LAB ", (0.0, 0.0, 0.0), [0, 0, 0]),
        (b"LAB ", (0.5429, 80.81, 69.89), [255, 0, 0]),  # D50 Lab of sRGB red
    ],
)
def test_ase_color_models_convert_to_rgb(tmp_path, model, values, expected):
    (tmp_path / "m.ase").write_bytes(_ase_one(model, values))
    got = read_palette(tmp_path / "m.ase")[0]
    assert np.abs(got.astype(int) - expected).max() <= 2, got


def test_image_contributes_its_distinct_colors(tmp_path):
    """Scan order, transparency skipped — how a strip reads back and how a
    reference picture gives up its palette."""
    art = np.zeros((2, 3, 4), dtype=np.uint8)
    art[..., 3] = 255
    art[0] = [[30, 34, 52, 255], [94, 167, 64, 255], [30, 34, 52, 255]]
    art[1] = [[240, 240, 240, 255], [94, 167, 64, 255], [7, 7, 7, 0]]
    Image.fromarray(art).save(tmp_path / "art.png")
    assert np.array_equal(read_palette(tmp_path / "art.png"), PAL[:3])


def test_content_beats_a_misleading_extension(tmp_path):
    (tmp_path / "actually.hex").write_bytes(
        write_palette(tmp_path / "p.ase", PAL).read_bytes()
    )
    assert np.array_equal(read_palette(tmp_path / "actually.hex"), PAL)


def test_load_palette_spec_takes_a_list_or_a_file(tmp_path):
    assert np.array_equal(load_palette_spec("#1e2234,#5ea740,#f0f0f0"), PAL[:3])
    path = write_palette(tmp_path / "p.gpl", PAL)
    assert np.array_equal(load_palette_spec(str(path)), PAL)
    with pytest.raises(PaletteFileError):
        load_palette_spec("not-a-palette")


def test_unreadable_and_unwritable_are_reported(tmp_path):
    (tmp_path / "empty.hex").write_text("no colors here\n")
    with pytest.raises(PaletteFileError):
        read_palette(tmp_path / "empty.hex")
    with pytest.raises(PaletteFileError):
        write_palette(tmp_path / "p.aco", PAL)  # unsupported format
    with pytest.raises(PaletteFileError):
        write_palette(tmp_path / "p.gpl", np.zeros((0, 3), np.uint8))


def test_duplicates_drop_keeping_first_seen_order(tmp_path):
    (tmp_path / "dup.hex").write_text("1E2234\n5EA740\n1E2234\nF0F0F0\n")
    assert np.array_equal(read_palette(tmp_path / "dup.hex"), PAL[:3])
