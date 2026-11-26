"""Round-trip and layout checks for the PLY reader/writer."""
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splatcore.io import SplatSet, load_ply, save_ply, degree_for_rest  # noqa: E402


def _synthetic(n=97, deg=3):
    rng = np.random.default_rng(7)
    k = (deg + 1) ** 2 - 1
    return SplatSet(
        xyz=rng.normal(size=(n, 3)).astype(np.float32),
        f_dc=rng.normal(size=(n, 3)).astype(np.float32),
        f_rest=rng.normal(size=(n, k, 3)).astype(np.float32),
        opacity=rng.normal(size=n).astype(np.float32),
        scale=rng.normal(size=(n, 3)).astype(np.float32),
        rot=rng.normal(size=(n, 4)).astype(np.float32),
    )


def test_degree_for_rest():
    assert degree_for_rest(0) == 0
    assert degree_for_rest(9) == 1
    assert degree_for_rest(24) == 2
    assert degree_for_rest(45) == 3


def test_roundtrip_bit_exact():
    s = _synthetic()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ply")
        save_ply(p, s)
        b = load_ply(p)
    for f in ("xyz", "f_dc", "f_rest", "opacity", "scale", "rot"):
        assert np.array_equal(getattr(s, f), getattr(b, f)), f
    assert b.sh_degree == 3


def test_channel_major_on_disk():
    """f_rest must be written channel-major: f_rest_0..K-1 are all red."""
    s = _synthetic(n=4, deg=1)          # K = 3 coeffs per channel
    s.f_rest[:] = 0.0
    s.f_rest[:, 0, 0] = 1.0             # coeff 0, red  -> f_rest_0
    s.f_rest[:, 0, 1] = 2.0             # coeff 0, green-> f_rest_3
    s.f_rest[:, 2, 2] = 3.0             # coeff 2, blue -> f_rest_8
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ply")
        save_ply(p, s, write_normals=False)
        raw = np.fromfile(p, dtype="<f4", offset=_header_len(p))
    row = raw.reshape(4, -1)[0]
    # layout: x y z | f_dc 0..2 | f_rest 0..8 | opacity | scale 0..2 | rot 0..3
    rest = row[6:15]
    assert rest[0] == 1.0 and rest[3] == 2.0 and rest[8] == 3.0
    assert rest[[1, 2, 4, 5, 6, 7]].sum() == 0.0


def test_end_header_in_comment():
    s = _synthetic(n=4)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "comment.ply"
        save_ply(p, s)
        header, body = p.read_bytes().split(b"end_header\n", 1)
        header = header.replace(b"element vertex", b"comment end_header is followed by data\nelement vertex")
        for newline in (b"\n", b"\r\n"):
            p.write_bytes((header + b"end_header\n").replace(b"\n", newline) + body)
            np.testing.assert_array_equal(load_ply(p).xyz, s.xyz)


def test_vertex_after_scalar_elements():
    s = _synthetic(n=4)
    s.extra["segment_id"] = np.array([3, 255, 42, 512], dtype="<u2")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "elements.ply"
        save_ply(p, s)
        header, body = p.read_bytes().split(b"end_header\n", 1)
        header = header.replace(
            b"element vertex",
            b"element frame 2\nproperty uchar id\nproperty double time\n"
            b"element camera 1\nproperty short exposure\nelement vertex",
        )
        header += b"element note 1\nproperty uint id\nend_header\n"
        p.write_bytes(header + struct.pack("<BdBdh", 1, 42, 2, 43, -300)
                      + body + struct.pack("<I", 987))
        b = load_ply(p)
    for name in ("xyz", "f_dc", "f_rest", "opacity", "scale", "rot"):
        np.testing.assert_array_equal(getattr(b, name), getattr(s, name))
    np.testing.assert_array_equal(b.extra["segment_id"], s.extra["segment_id"])
    assert b.extra["segment_id"].dtype == s.extra["segment_id"].dtype


def test_list_properties_before_or_within_vertex_are_refused():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "lists.ply"
        save_ply(p, _synthetic(n=4))
        header, body = p.read_bytes().split(b"end_header\n", 1)
        prop = b"property list uchar int indices\n"
        leading = header.replace(b"element vertex", b"element face 1\n" + prop + b"element vertex")
        p.write_bytes(leading + b"end_header\n" + b"\x00" + body)
        with np.testing.assert_raises_regex(ValueError, "list properties on the face element"):
            load_ply(p)
        p.write_bytes(header + prop + b"end_header\n" + body)
        with np.testing.assert_raises_regex(ValueError, "list properties on the vertex element"):
            load_ply(p)


def test_truncated_ply_reports_the_missing_block():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "truncated.ply"
        save_ply(p, _synthetic(n=4))
        header, body = p.read_bytes().split(b"end_header\n", 1)
        leading = header.replace(b"element vertex", b"element frame 1\nproperty float time\nelement vertex")
        p.write_bytes(leading + b"end_header\n" + b"\x00")
        with np.testing.assert_raises_regex(ValueError, "truncated before the vertex element"):
            load_ply(p)
        p.write_bytes(leading + b"end_header\n" + struct.pack("<f", 42) + body[:-1])
        with np.testing.assert_raises_regex(ValueError, "truncated: header says 4"):
            load_ply(p)


def test_ply_header_size_is_bounded():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "long-header.ply"
        p.write_bytes(b"ply\ncomment " + b"x" * (1 << 20) + b"\nend_header\n")
        with np.testing.assert_raises_regex(ValueError, "header is implausibly large"):
            load_ply(p)


def test_copy_is_a_copy():
    """copy() must not share storage: slicing a numpy array gives a view."""
    s = _synthetic()
    c = s.copy()
    c.xyz[0, 0] += 1.0
    c.f_rest[1, 2, 0] += 1.0
    c.opacity[3] += 1.0
    assert s.xyz[0, 0] != c.xyz[0, 0], "copy() shares positions with the original"
    assert s.f_rest[1, 2, 0] != c.f_rest[1, 2, 0], "copy() shares SH"
    assert s.opacity[3] != c.opacity[3], "copy() shares opacity"


def test_subset_and_concat():
    s = _synthetic(n=50)
    a, b = s.take(np.arange(20)), s.take(np.arange(20, 50))
    j = a.concat(b)
    assert j.n == 50
    assert np.array_equal(j.xyz, s.xyz)


def _header_len(p):
    with open(p, "rb") as f:
        buf = f.read(8192)
    return buf.index(b"end_header") + len("end_header\n")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("all io tests passed")
