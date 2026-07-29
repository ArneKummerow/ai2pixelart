import json

import numpy as np
from click.testing import CliRunner
from PIL import Image

from ai2pixelart.cli import main
from ai2pixelart.corrupt import upscale
from ai2pixelart.palettefile import read_palette, write_palette


def _run(*args):
    result = CliRunner().invoke(main, [str(a) for a in args])
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


def _pseudo(tmp_path, sprite, name="a.png"):
    """One pseudo-pixel-art image on disk, as a user would have it."""
    path = tmp_path / name
    Image.fromarray(upscale(sprite, 3.0, interp="nearest")).save(path)
    return path


def test_palette_extract_writes_the_cleaner_s_proposal(tmp_path, sprite):
    src = _pseudo(tmp_path, sprite)
    out = tmp_path / "pal.gpl"
    output = _run("palette", "extract", src, "-o", out, "--colors", 6)

    pal = read_palette(out)
    assert 2 <= len(pal) <= 6
    assert f"{len(pal)} colors" in output
    # the proposal is the palette 'clean' itself would use
    from ai2pixelart.pipeline import clean

    assert np.array_equal(pal, clean(np.array(Image.open(src).convert("RGB")), max_colors=6).palette)


def test_palette_extract_distinct_tallies_exact_colors(tmp_path, sprite, palette):
    path = tmp_path / "clean.png"
    Image.fromarray(sprite).save(path)
    out = tmp_path / "pal.hex"
    _run("palette", "extract", path, "-o", out, "--distinct")

    got = {tuple(c) for c in read_palette(out)}
    assert got == {tuple(c) for c in np.unique(sprite.reshape(-1, 3), axis=0)}


def test_palette_convert_between_formats(tmp_path, palette):
    gpl = write_palette(tmp_path / "in.gpl", palette)
    _run("palette", "convert", gpl, "-o", tmp_path / "out.ase")
    _run("palette", "convert", tmp_path / "out.ase", "-o", tmp_path / "out.png", "--scale", 8)
    assert np.array_equal(read_palette(tmp_path / "out.png"), palette)


def test_clean_forces_a_palette_from_a_file(tmp_path, sprite):
    src = _pseudo(tmp_path, sprite)
    forced = np.array([[30, 34, 52], [94, 167, 64], [240, 240, 240]], dtype=np.uint8)
    pal_file = write_palette(tmp_path / "forced.pal", forced)
    out = tmp_path / "out.png"
    _run("clean", src, "-o", out, "--palette", pal_file)

    got = np.unique(np.array(Image.open(out).convert("RGB")).reshape(-1, 3), axis=0)
    assert {tuple(c) for c in got} <= {tuple(c) for c in forced}


def test_clean_writes_the_palette_it_used(tmp_path, sprite):
    src = _pseudo(tmp_path, sprite)
    out, pal_out = tmp_path / "out.png", tmp_path / "used.gpl"
    info = json.loads(_run("clean", src, "-o", out, "--palette-out", pal_out, "--colors", 5))

    assert info["palette_file"] == str(pal_out)
    assert len(read_palette(pal_out)) == info["palette_size"]


def test_clean_batch_names_a_palette_after_each_image(tmp_path, sprite):
    ws = tmp_path / "ws"
    ws.mkdir()
    _pseudo(ws, sprite, "one.png")
    _pseudo(ws, sprite[::-1], "two.png")
    outdir = tmp_path / "out"
    _run("clean", ws, "-o", outdir, "--palette-out", outdir / "p.hex")

    assert sorted(p.name for p in outdir.glob("*.hex")) == ["one.hex", "two.hex"]
    assert len(read_palette(outdir / "one.hex")) >= 2


def test_bad_palette_arguments_fail_before_computing(tmp_path, sprite):
    src = _pseudo(tmp_path, sprite)
    out = tmp_path / "out.png"
    runner = CliRunner()

    bad_fmt = runner.invoke(main, [
        "clean", str(src), "-o", str(out), "--palette-out", str(tmp_path / "p.aco")])
    assert bad_fmt.exit_code != 0 and "unknown palette format" in bad_fmt.output
    assert not out.exists()  # nothing was computed or written

    bad_pal = runner.invoke(main, ["clean", str(src), "-o", str(out), "--palette", "nope"])
    assert bad_pal.exit_code != 0 and "not a palette file" in bad_pal.output
