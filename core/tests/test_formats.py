"""Round-trip checks for .splat and .spz.

Both formats quantise, so "it round-trips" is not a yes/no question -- every
test here states the error it will tolerate and where that number comes from.
A bound that is loose enough to pass whatever the code does would prove
nothing, so each one is derived from the format's own step size: half a
quantisation step, and no more.
"""
import errno
import gzip
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splatcore.io import SplatSet, replacing, save_ply           # noqa: E402
from splatcore import formats                                     # noqa: E402
from splatcore.formats import (                                   # noqa: E402
    NGSP_MAGIC, load_any, load_splat, load_spz, save_splat, save_spz, sniff,
)

SH_C0 = 0.28209479177387814


def _synthetic(n=512, deg=3, seed=11):
    """A cloud whose values all sit inside what the formats can represent.

    Deliberately not random noise across the whole float range: the point of
    these tests is the quantiser's own error, not its clipping behaviour,
    which is measured separately. The tightest ceiling is ``.splat``'s
    colour: it stores the rendered 0.5 + C0*f_dc in a byte, so |f_dc| above
    1/(2*C0) = 1.772 cannot be represented at all.
    """
    rng = np.random.default_rng(seed)
    k = (deg + 1) ** 2 - 1
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return SplatSet(
        xyz=rng.uniform(-3, 3, (n, 3)).astype(np.float32),
        f_dc=rng.uniform(-1.7, 1.7, (n, 3)).astype(np.float32),
        f_rest=rng.uniform(-0.9, 0.9, (n, k, 3)).astype(np.float32),
        opacity=rng.uniform(-4, 4, n).astype(np.float32),
        scale=rng.uniform(-6, -1, (n, 3)).astype(np.float32),
        rot=q.astype(np.float32),
    )


def _angle_deg(a, b):
    """Angle between two sets of quaternions, sign-insensitive.

    Normalises first: the pre-v3 decoder rebuilds the fourth component from
    the other three, and rounding can push their squares past 1, which leaves
    it slightly short of unit length.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    d = np.abs((a * b).sum(1))
    return np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0)))


def test_exports_preserve_existing_tmp_neighbors():
    s = _synthetic(n=16)
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        expected = set()
        for extension in ("ply", "splat", "spz"):
            path = folder / f"scene.{extension}"
            neighbor = folder / f"{path.name}.tmp"
            neighbor.write_bytes(b"unrelated file")

            formats.save_any(path, s)
            assert neighbor.exists(), f"exporting {extension} removed a neighboring file"
            assert neighbor.read_bytes() == b"unrelated file"
            assert load_any(path).n == s.n

            first_export = path.read_bytes()
            formats.save_any(path, s)
            assert path.read_bytes() == first_export
            assert neighbor.read_bytes() == b"unrelated file"
            expected.update((path, neighbor))
        assert set(folder.iterdir()) == expected


def test_failed_write_preserves_destination_and_neighbor():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        path = folder / "scene.ply"
        neighbor = folder / "scene.ply.tmp"
        path.write_bytes(b"previous export")
        neighbor.write_bytes(b"unrelated file")
        try:
            with replacing(path) as tmp:
                Path(tmp).write_bytes(b"partial export")
                raise OSError(errno.ENOSPC, "simulated write failure", tmp)
        except OSError as error:
            assert error.errno == errno.ENOSPC
            assert error.filename == str(path)
        else:
            raise AssertionError("the writer error must reach the caller")
        assert path.read_bytes() == b"previous export"
        assert neighbor.exists(), "failure cleanup removed a neighboring file"
        assert neighbor.read_bytes() == b"unrelated file"
        assert set(folder.iterdir()) == {path, neighbor}


def test_overlapping_writes_keep_independent_temporary_files():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        path = folder / "scene.ply"
        path.write_bytes(b"previous export")
        # Complete the second write while the first is still staged.
        with replacing(path) as first:
            Path(first).write_bytes(b"complete first export")
            with replacing(path) as second:
                assert Path(first).parent == Path(second).parent == folder
                assert first != second
                Path(second).write_bytes(b"complete second export")
                assert path.read_bytes() == b"previous export"
            assert path.read_bytes() == b"complete second export"
            assert Path(first).read_bytes() == b"complete first export"
        assert path.read_bytes() == b"complete first export"
        assert set(folder.iterdir()) == {path}


# ------------------------------------------------------------------- .splat --

def test_splat_roundtrip():
    s = _synthetic(deg=2)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.splat")
        save_splat(p, s, sort=False)
        assert os.path.getsize(p) == 32 * s.n, "32 bytes per splat"
        b = load_splat(p)

    assert b.n == s.n
    # position and scale are float32 on disk -- exact, bar the log/exp trip
    assert np.array_equal(b.xyz, s.xyz)
    assert np.abs(b.scale - s.scale).max() < 1e-5

    # colour: one byte over [0, 1] after the 0.5 + C0*dc mapping, so the
    # f_dc step is 1/(255*C0) and the worst error is half of it
    step = 1.0 / (255.0 * SH_C0)
    err = np.abs(b.f_dc - s.f_dc).max()
    assert err < step / 2 + 1e-6, "f_dc error %.5f exceeds %.5f" % (err, step / 2)

    # opacity: one byte over alpha, so compare there -- the logit of a byte
    # near 0 or 1 is enormous and says nothing about the format
    assert np.abs(b.alpha - s.alpha).max() < 1.0 / 510 + 1e-9

    # rotation: four bytes over [-1, 1], 1/128 per step on every component
    assert _angle_deg(b.quat, s.quat).max() < 1.5

    assert b.f_rest.shape[1] == 0, ".splat cannot carry SH past the DC term"


def test_splat_colour_ceiling_is_the_formats_doing():
    """A bright splat comes back dimmer, and the limit is where it should be.

    Not a defect to fix: .splat stores the rendered colour in a byte, so any
    f_dc past 1/(2*C0) is outside the format. Worth a test so nobody later
    "fixes" it by rescaling, which would change every colour in the file.
    """
    ceiling = 1.0 / (2.0 * SH_C0)
    s = _synthetic(n=8, deg=0)
    s.f_dc[:] = np.array([[ceiling * 0.98, ceiling * 2.0, -ceiling * 2.0]] * 8,
                         np.float32)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.splat")
        save_splat(p, s, sort=False)
        b = load_splat(p)
    step = 1.0 / (255.0 * SH_C0)
    assert np.abs(b.f_dc[:, 0] - s.f_dc[:, 0]).max() < step / 2 + 1e-6
    assert np.allclose(b.f_dc[:, 1], ceiling, atol=step)
    assert np.allclose(b.f_dc[:, 2], -ceiling, atol=step)


def test_splat_sort_is_by_importance():
    s = _synthetic(n=64, deg=0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.splat")
        save_splat(p, s, sort=True)
        b = load_splat(p)
    imp = b.sigma.prod(1) * b.alpha
    assert np.all(np.diff(imp) <= 1e-6), "file is not in descending importance"


# ---------------------------------------------------------------------- .spz --

def _spz_bounds(b, s, version):
    # positions: 24-bit fixed point, 12 fractional bits
    assert np.abs(b.xyz - s.xyz).max() < 1.0 / 8192 + 1e-9

    # scales: one byte over [-10, 5.94] in log space at 1/16 per step
    assert np.abs(b.scale - s.scale).max() < 1.0 / 32 + 1e-6

    # colour: one byte, f_dc step 1/(0.15*255)
    step = 1.0 / (formats.COLOR_SCALE * 255.0)
    assert np.abs(b.f_dc - s.f_dc).max() < step / 2 + 1e-6

    # opacity: one byte over alpha
    assert np.abs(b.alpha - s.alpha).max() < 1.0 / 510 + 1e-9

    # SH: one byte over [-1, 1] at 1/128 per step
    assert b.f_rest.shape == s.f_rest.shape
    assert np.abs(b.f_rest - s.f_rest).max() < 1.0 / 256 + 1e-6

    # Rotation. Before v3 the fourth component is recovered through a square
    # root that blows up near a half-turn, so the ceiling is high by the
    # format's own doing; from v3 nine bits cover +-sqrt(1/2) directly.
    limit = 11.0 if version < 3 else 0.25
    worst = _angle_deg(b.quat, s.quat).max()
    assert worst < limit, "v%d rotation error %.3f deg exceeds %.3f" \
        % (version, worst, limit)


def test_spz_roundtrip_v2():
    s = _synthetic()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.spz")
        info = save_spz(p, s, version=2)
        b = load_spz(p)
    assert info["shDegree"] == 3 and info["fractionalBits"] == 12
    assert info["positionsClipped"] == 0
    assert b.n == s.n
    _spz_bounds(b, s, 2)


def test_spz_roundtrip_v3_is_the_default_and_says_why():
    """Version 3 differs only in the quaternions, and by a wide margin.

    This is the test behind the choice of default. If it ever stops holding,
    the default in ``save_spz`` is the thing to revisit, not the bound here.
    """
    s = _synthetic()
    with tempfile.TemporaryDirectory() as d:
        p2, p3 = os.path.join(d, "a.spz"), os.path.join(d, "b.spz")
        save_spz(p2, s, version=2)
        assert save_spz(p3, s)["version"] == 3, "version 3 must be the default"
        b2, b3 = load_spz(p2), load_spz(p3)
    _spz_bounds(b3, s, 3)
    e2 = _angle_deg(b2.quat, s.quat)
    e3 = _angle_deg(b3.quat, s.quat)
    assert e2.max() > 1.0, "v2 is expected to be visibly worse; got %.3f deg" \
                           % e2.max()
    assert e3.max() < e2.max() / 10, \
        "smallest-three (%.3f deg worst) should beat first-three (%.3f deg) " \
        "by an order of magnitude" % (e3.max(), e2.max())


def test_spz_degree_cap():
    s = _synthetic(deg=3)
    with tempfile.TemporaryDirectory() as d:
        full, cut = os.path.join(d, "f.spz"), os.path.join(d, "c.spz")
        a = save_spz(full, s)
        c = save_spz(cut, s, sh_degree=1)
        b = load_spz(cut)
    assert b.sh_degree == 1 and b.f_rest.shape[1] == 3
    assert np.abs(b.f_rest - s.f_rest[:, :3, :]).max() < 1.0 / 256 + 1e-6
    assert c["bytes"] < a["bytes"] / 2, "dropping 12 of 15 bands must shrink it"


def test_spz_sh_axis_order():
    """spz is coefficient-major with the channel innermost -- opposite to PLY.

    Reads the bytes back out of the container rather than trusting the
    decoder, because an axis swap in both directions cancels out and passes
    every round-trip test.
    """
    s = _synthetic(n=4, deg=1)          # 3 coefficients per channel
    s.f_rest[:] = 0.0
    s.f_rest[:, 0, 0] = 0.5             # coeff 0, red   -> byte 0
    s.f_rest[:, 0, 2] = -0.5            # coeff 0, blue  -> byte 2
    s.f_rest[:, 2, 1] = 0.25            # coeff 2, green -> byte 7
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.spz")
        save_spz(p, s, version=2)
        with gzip.open(p, "rb") as g:
            blob = g.read()

    n, k = 4, 3
    off = 16 + n * 9 + n + n * 3 + n * 3 + n * 3      # header + the five before sh
    sh = np.frombuffer(blob, np.uint8, count=n * k * 3, offset=off).reshape(n, k * 3)
    row = sh[0].astype(int)
    assert row[0] == 192 and row[2] == 64 and row[7] == 160, row.tolist()
    assert [row[i] for i in (1, 3, 4, 5, 6, 8)] == [128] * 6


def test_spz_quaternion_encoders_are_inverses():
    """Each encoder against its own decoder, on quaternions that stress both.

    Includes the axis-aligned cases, where one component is 1 and the other
    three are 0 -- the boundary the smallest-three packing has to get right.
    """
    rng = np.random.default_rng(3)
    q = rng.normal(size=(4096, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = np.concatenate([np.eye(4), -np.eye(4), q])
    n = len(q)

    # Pre-v3: fine on average, poor in the tail, because the fourth component
    # comes out of a square root near zero. These three numbers are the
    # measured shape of that, and they are why version 3 is the default.
    a = formats._quat_decode_first_three(
        formats._quat_encode_first_three(q).tobytes(), n)
    e = _angle_deg(a, q)
    assert np.median(e) < 1.0, "median %.3f deg" % np.median(e)
    assert np.percentile(e, 99) > 2.0, \
        "the pre-v3 tail is supposed to be bad; p99 came out at %.3f deg -- " \
        "if that improved, the encoding is being read wrong" \
        % np.percentile(e, 99)
    assert e.max() < 11.0, "max %.3f deg" % e.max()

    b = formats._quat_decode_smallest_three(
        formats._quat_encode_smallest_three(q).tobytes(), n)
    assert _angle_deg(b, q).max() < 0.25

    # and the packed words must carry the index of the component left out
    packed = formats._quat_encode_smallest_three(q)
    assert np.array_equal((packed >> 30) & 0x3, np.argmax(np.abs(q), axis=1))


def test_spz_position_range_gives_up_precision():
    """Beyond +-2048 the fixed point would wrap; it must lose bits instead."""
    s = _synthetic(n=32, deg=0)
    s.xyz = (s.xyz * 2000.0).astype(np.float32)        # up to ~6000 units out
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.spz")
        info = save_spz(p, s)
        b = load_spz(p)
    assert info["fractionalBits"] < 12, "should have dropped fractional bits"
    assert info["positionsClipped"] == 0, "nothing may wrap around"
    assert np.abs(b.xyz - s.xyz).max() < 1.0 / (1 << (info["fractionalBits"] + 1)) + 1e-4


def test_spz_version4_reader():
    """The v4 container path: 32-byte header, table of contents, zstd streams.

    Self-consistent only -- the file is built here from the same reading of
    the layout the reader uses, so this guards the table-of-contents
    arithmetic and the stream mapping, not the spec itself.
    """
    import zstandard

    s = _synthetic(n=128, deg=2)
    with tempfile.TemporaryDirectory() as d:
        legacy = os.path.join(d, "t.spz")
        save_spz(legacy, s, version=3)
        with gzip.open(legacy, "rb") as g:
            blob = g.read()
        n = s.n
        sizes = formats._spz_stream_sizes(n, 3, s.f_rest.shape[1])

        streams, off = [], 16
        for name in formats._STREAMS:
            streams.append(blob[off:off + sizes[name]])
            off += sizes[name]

        cctx = zstandard.ZstdCompressor(level=12)
        packed = [cctx.compress(x) for x in streams]
        toc = b"".join(struct.pack("<QQ", len(c), len(p))
                       for c, p in zip(packed, streams))
        header = struct.pack("<IIIBBBBI12x", NGSP_MAGIC, 4, n,
                             s.sh_degree, 12, 0, len(packed), 32)

        p4 = os.path.join(d, "v4.spz")
        with open(p4, "wb") as f:
            f.write(header + toc + b"".join(packed))
        b = load_spz(p4)

    assert b.n == s.n and b.sh_degree == s.sh_degree
    _spz_bounds(b, s, 3)


def test_spz_rejects_mismatched_stream_count():
    header = struct.pack("<IIIBBBBI12x", NGSP_MAGIC, 4, 10, 2, 12, 0, 5, 32)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.spz")
        with open(p, "wb") as f:
            f.write(header + b"\0" * 128)
        try:
            load_spz(p)
        except ValueError as e:
            assert "streams" in str(e), e
        else:
            raise AssertionError("a degree-2 file with 5 streams must be rejected")


def test_spz_refuses_files_that_ask_for_too_much_memory():
    """A small crafted file must not make the reader allocate gigabytes.

    Three ways in: a gzipped file whose header states a huge point count, a
    gzipped file whose stream expands far past what its header promises, and a
    version 4 table of contents that states a huge decompressed size.
    """
    import zstandard

    def refused(blob, gz, word):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bomb.spz")
            with open(p, "wb") as f:
                f.write(gzip.compress(blob) if gz else blob)
            try:
                load_spz(p)
            except ValueError as e:
                assert word in str(e), e
            else:
                raise AssertionError("a crafted spz must be refused")

    huge = formats.SPZ_MAX_POINTS + 1
    refused(struct.pack("<IIIBBBB", NGSP_MAGIC, 2, huge, 0, 12, 0, 0), True, "points")
    # 10 points need a few hundred bytes; the 500 MB of zeros behind them (half
    # a megabyte gzipped) are never read
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "padded.spz")
        with gzip.open(p, "wb") as g:
            g.write(struct.pack("<IIIBBBB", NGSP_MAGIC, 2, 10, 0, 12, 0, 0))
            for _ in range(50):
                g.write(bytes(10 << 20))
        assert load_spz(p).n == 10
    refused(struct.pack("<IIIBBBBI12x", NGSP_MAGIC, 4, huge, 0, 12, 0, 5, 32)
            + bytes(128), False, "points")
    bomb = zstandard.ZstdCompressor().compress(bytes(64 << 20))
    toc = struct.pack("<QQ", len(bomb), 64 << 20) + struct.pack("<QQ", 0, 0) * 4
    refused(struct.pack("<IIIBBBBI12x", NGSP_MAGIC, 4, 10, 0, 12, 0, 5, 32) + toc + bomb,
            False, "need")


# ------------------------------------------------------------------ sniffing --

def test_sniff_and_load_any():
    s = _synthetic(n=64, deg=1)
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        save_ply(os.path.join(d, "a.ply"), s)
        save_splat(os.path.join(d, "b.splat"), s)
        save_spz(os.path.join(d, "c.spz"), s)
        paths = {"ply": "a.ply", "splat": "b.splat", "spz": "c.spz"}
        for kind, name in paths.items():
            p = os.path.join(d, name)
            assert sniff(p) == kind, name
            assert load_any(p).n == s.n, name

        # the extension must not be what decides it
        wrong = os.path.join(d, "actually_spz.ply")
        os.replace(os.path.join(d, "c.spz"), wrong)
        assert sniff(wrong) == "spz"
        assert load_any(wrong).n == s.n


def test_sniff_rejects_gzipped_ply_with_a_useful_message():
    s = _synthetic(n=8, deg=0)
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "a.ply")
        save_ply(plain, s)
        packed = os.path.join(d, "a.ply.gz")
        with open(plain, "rb") as src, gzip.open(packed, "wb") as dst:
            dst.write(src.read())
        try:
            sniff(packed)
        except ValueError as e:
            assert "gzip-compressed PLY" in str(e), e
        else:
            raise AssertionError("a gzipped PLY must not pass as a container")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("all format tests passed")
