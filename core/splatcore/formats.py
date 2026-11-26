"""The other two splat containers: ``.splat`` and ``.spz``.

PLY is the working format -- float32 throughout, a load/save round trip is
bit-exact, and nothing is thrown away. These two are *delivery* formats. Both
quantise, and one of them discards most of the radiance field:

``.splat``
    antimatter15's web viewer format. 32 bytes per splat: position and scale as
    float32, colour and opacity as four bytes, rotation as four more. There is
    no room for spherical harmonics at all, so the view-dependent part of the
    appearance is gone -- a shiny object loads back matte. Opening a ``.splat``
    and saving it as PLY gives you a degree-0 cloud; that is the format's doing,
    not ours.

``.spz``
    Niantic's compressed format. Keeps all the SH bands but quantises
    everything: positions to 24-bit fixed point, scales, opacity, colour and
    each SH coefficient to one byte. Roughly 10x smaller than the PLY and the
    error is well under what you can see.

Both readers accept every version in the wild. The ``.spz`` writer emits
version 3, and that choice is worth explaining, because version 2 is the more
compatible one. Before version 3, spz stored only the first three quaternion
components and recovered the fourth as ``sqrt(1 - |xyz|^2)``. Near a
half-turn, where that fourth component approaches zero, the square root
amplifies one byte of rounding into a large angle: measured over random
rotations, version 2 is off by 0.63 degrees at the median but 7.0 degrees at
the 99th percentile and 10.8 degrees at worst. Version 3's "smallest three"
encoding drops whichever component is largest instead, so the three that
remain never exceed sqrt(1/2) and nine bits cover that range -- 0.20 degrees
at worst, fifty times better. Splat orientation is visible on anisotropic
splats, so ``version=2`` stays available for an old reader but is not the
default.

Version 4 (zstd streams) is read but not written: there is no reference file
here to check a v4 writer against, and an unverified writer whose files other
tools reject is worse than no writer at all.

Layout facts worth stating once, because they are the two places an
implementation silently goes wrong:

* **Quaternion order.** PLY stores (w, x, y, z); both of these store
  (x, y, z, w). ``.spz`` leans on it -- the pre-v3 encoding keeps only the
  first three components and reconstructs the fourth as positive, which only
  works if the fourth is w.
* **SH axis order.** PLY is channel-major (all of red, then all of green).
  ``.spz`` is coefficient-major with the channel innermost, which is exactly
  how ``SplatSet.f_rest`` is already laid out, so it flattens straight through.
"""
from __future__ import annotations

import gzip
import os
import struct

import numpy as np

from .io import SH_C0, SplatSet, replacing, sh_coeffs_for_degree

# ---------------------------------------------------------------- constants --

NGSP_MAGIC = 0x5053474E          # "NGSP", little-endian
SPZ_LATEST = 4
SPZ_SMALLEST_THREE_FROM = 3      # version that switched quaternion encoding
SPZ_ZSTD_FROM = 4                # version that switched container
SPZ_FLAG_ANTIALIASED = 0x1
COLOR_SCALE = 0.15               # spz's fixed colour quantisation scale
_SQRT_HALF = float(np.sqrt(0.5))

# Stream order, both in the legacy single blob and in a v4 table of contents.
_STREAMS = ("positions", "alphas", "colors", "scales", "rotations", "sh")


# ------------------------------------------------------------------ helpers --

def _logit(p, eps=1.0 / 510.0):
    """Inverse sigmoid, with the ends pulled in by half a quantisation step.

    A stored alpha of 0 or 255 means "beyond what a byte can say", not
    "exactly opaque"; mapping those to +-inf would poison every later
    computation, so they land one half-step inside instead.
    """
    p = np.clip(np.asarray(p, np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _unit(q):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, 1e-12)


def _wxyz_to_xyzw(rot):
    return np.stack([rot[:, 1], rot[:, 2], rot[:, 3], rot[:, 0]], axis=1)


def _xyzw_to_wxyz(q):
    return np.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], axis=1)


def _u8(x):
    """Round to the nearest byte, saturating rather than wrapping."""
    return np.clip(np.rint(np.asarray(x, np.float64)), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- sniffing --

def sniff(path) -> str:
    """Identify a splat file by its contents. Returns 'ply', 'splat' or 'spz'.

    The extension is a hint, not evidence: splat files get renamed, and a
    ``.spz`` that is really a gzipped PLY is a thing that happens. Only the
    first few bytes are read.
    """
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:3] == b"ply":
        return "ply"
    if head[:2] == b"\x1f\x8b":
        # gzip: either a legacy spz or someone's gzipped PLY.
        with gzip.open(path, "rb") as g:
            inner = g.read(8)
        if len(inner) >= 4 and struct.unpack("<I", inner[:4])[0] == NGSP_MAGIC:
            return "spz"
        if inner[:3] == b"ply":
            raise ValueError("this is a gzip-compressed PLY, not a splat file; "
                             "unpack it first")
        raise ValueError("compressed file is neither spz nor PLY")
    if len(head) >= 4 and struct.unpack("<I", head[:4])[0] == NGSP_MAGIC:
        return "spz"
    size = os.path.getsize(path)
    if size and size % 32 == 0 and _looks_like_splat(path):
        return "splat"
    raise ValueError("unknown splat format (%d bytes, starts with %r)"
                     % (size, head[:4]))


def _looks_like_splat(path, rows: int = 256) -> bool:
    """Whether the first rows read as splats: finite positions, positive finite scales.

    A ``.splat`` has no magic number, only a size that divides by 32 -- and
    so does one file in 32 of anything else. A photo of the right size used
    to open as twenty thousand splats at NaN. Random bytes pass one row in
    eight; nine rows in ten are asked for, so a real file with the odd
    broken splat near its start still opens.

    Plain text needs a test of its own: bytes under 0x80 read as finite,
    positive floats, every one of them. So a row must also hold a byte
    text has not -- among a splat's 18 mantissa bytes and its colour and
    rotation there always is one.
    """
    raw = np.fromfile(path, dtype=_SPLAT_DTYPE, count=rows)
    if raw.size == 0:
        return False
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(raw["pos"]).all(1) & np.isfinite(raw["scale"]).all(1) \
            & (raw["scale"] > 0).all(1)
    ok &= (raw.view(np.uint8).reshape(len(raw), 32) >= 0x80).any(1)
    return bool(ok.mean() >= 0.9)


def load_any(path) -> SplatSet:
    """Load a splat file of any supported format, detected by content."""
    kind = sniff(path)
    if kind == "ply":
        from .io import load_ply
        return load_ply(path)
    if kind == "splat":
        return load_splat(path)
    return load_spz(path)


def save_any(path, s: SplatSet, fmt: str | None = None,
             sh_degree: int | None = None, **kw) -> dict:
    """Write a splat file. Format comes from ``fmt``, else the extension.

    ``sh_degree`` caps the spherical harmonics for every format, not just the
    ones that quantise: a degree-1 PLY is a third the size of a degree-3 one,
    and the cap belongs in one place rather than in each writer.
    """
    if fmt is None:
        fmt = os.path.splitext(str(path))[1].lower().lstrip(".")
    dropped = 0
    if sh_degree is not None and int(sh_degree) < s.sh_degree:
        k = sh_coeffs_for_degree(int(sh_degree))
        dropped = s.f_rest.shape[1] - k
        s = SplatSet(xyz=s.xyz, f_dc=s.f_dc, f_rest=s.f_rest[:, :k, :],
                     opacity=s.opacity, scale=s.scale, rot=s.rot, extra=s.extra)
    if fmt == "ply":
        from .io import save_ply
        save_ply(path, s, **kw)
        info = {"bytes": os.path.getsize(path)}
    elif fmt == "splat":
        info = save_splat(path, s, **kw)
    elif fmt == "spz":
        info = save_spz(path, s, **kw)
    else:
        raise ValueError("unknown output format %r (ply, splat or spz)" % fmt)
    # The degree actually written. Document.save used to report the
    # document's: a PLY capped to degree 0 came back as "shDegree": 3.
    info.setdefault("shDegree", 0 if fmt == "splat" else s.sh_degree)
    return {**info, "format": fmt, "shBandsCapped": int(dropped)}


# -------------------------------------------------------------------- .splat --

_SPLAT_DTYPE = np.dtype([
    ("pos", "<f4", 3),
    ("scale", "<f4", 3),          # linear, not log
    ("rgba", "u1", 4),
    ("rot", "u1", 4),             # (w, x, y, z) -- see below
])
assert _SPLAT_DTYPE.itemsize == 32


def load_splat(path) -> SplatSet:
    """Load an antimatter15 ``.splat``. Comes back as a degree-0 cloud."""
    raw = np.fromfile(path, dtype=_SPLAT_DTYPE)
    if raw.size == 0:
        raise ValueError("%s contains no splats" % path)
    n = len(raw)

    rgb = raw["rgba"][:, :3].astype(np.float64) / 255.0
    f_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)
    opacity = _logit(raw["rgba"][:, 3].astype(np.float64) / 255.0).astype(np.float32)

    # The byte 128 means "zero", so the usable range is [-1, 0.9922].
    rot = (raw["rot"].astype(np.float64) - 128.0) / 128.0
    rot = _unit(rot).astype(np.float32)        # already (w, x, y, z) on disk

    scale = np.log(np.maximum(raw["scale"].astype(np.float64), 1e-20))

    return SplatSet(
        xyz=raw["pos"].astype(np.float32),
        f_dc=f_dc,
        f_rest=np.zeros((n, 0, 3), np.float32),
        opacity=opacity,
        scale=scale.astype(np.float32),
        rot=rot,
    )


def save_splat(path, s: SplatSet, sort: bool = True) -> dict:
    """Write an antimatter15 ``.splat``.

    Spherical harmonics beyond the DC term do not fit in the format and are
    dropped. With ``sort``, splats go out biggest-and-most-opaque first, which
    is what the web viewers expect -- they composite in file order and only
    re-sort once the whole file has arrived, so an unsorted file looks wrong
    for the first second.
    """
    n = s.n
    order = np.arange(n)
    if sort and n:
        # the viewer's own importance measure: volume times opacity
        importance = s.sigma.prod(1) * s.alpha
        order = np.argsort(-importance)

    out = np.empty(n, dtype=_SPLAT_DTYPE)
    out["pos"] = s.xyz[order]
    out["scale"] = s.sigma[order].astype(np.float32)
    out["rgba"][:, :3] = _u8(s.rgb[order] * 255.0)
    out["rgba"][:, 3] = _u8(s.alpha[order] * 255.0)
    out["rot"] = _u8(s.quat[order] * 128.0 + 128.0)
    with replacing(path) as tmp:
        with open(tmp, "wb") as f:
            out.tofile(f)

    # The colour ceiling is the format's, not a setting: anything past
    # 1/(2*C0) cannot be stored, so say how many splats hit it.
    ceiling = 1.0 / (2.0 * SH_C0)
    clipped = int(np.count_nonzero(np.abs(s.f_dc) > ceiling)) if n else 0
    return {"shBandsDropped": int(s.f_rest.shape[1]), "sorted": bool(sort),
            "coloursClipped": clipped, "bytes": os.path.getsize(path)}


# ---------------------------------------------------------------------- .spz --

def _int24_decode(buf, n):
    b = np.frombuffer(buf, np.uint8, count=n * 9).reshape(n, 3, 3).astype(np.int32)
    v = b[:, :, 0] | (b[:, :, 1] << 8) | (b[:, :, 2] << 16)
    return np.where(v & 0x800000, v - 0x1000000, v)


def _int24_encode(q):
    q = (np.clip(q, -0x800000, 0x7FFFFF).astype(np.int64) & 0xFFFFFF).astype(np.uint32)
    out = np.empty(q.shape + (3,), np.uint8)
    out[..., 0] = q & 0xFF
    out[..., 1] = (q >> 8) & 0xFF
    out[..., 2] = (q >> 16) & 0xFF
    return out


def _quat_decode_first_three(buf, n):
    """Pre-v3: x, y, z as bytes; w is whatever is left over, and positive."""
    b = np.frombuffer(buf, np.uint8, count=n * 3).reshape(n, 3).astype(np.float64)
    xyz = b / 127.5 - 1.0
    w = np.sqrt(np.maximum(0.0, 1.0 - (xyz ** 2).sum(1)))
    return np.concatenate([xyz, w[:, None]], axis=1)


def _quat_encode_first_three(q):
    q = np.where(q[:, 3:4] < 0, -q, q)        # q and -q are the same rotation
    return _u8(q[:, :3] * 127.5 + 127.5)


def _quat_decode_smallest_three(buf, n):
    """v3+: drop the largest component instead of always dropping w.

    The three that are kept can never exceed sqrt(1/2), so nine bits spread
    over that range beat eight bits spread over [-1, 1] by a factor of four.
    """
    p = np.frombuffer(buf, "<u4", count=n).astype(np.uint32)
    largest = ((p >> 30) & 0x3).astype(np.int64)
    groups = np.stack([(p >> 20) & 0x3FF, (p >> 10) & 0x3FF, p & 0x3FF], axis=1)
    mag = (groups & 0x1FF).astype(np.float64) / 511.0 * _SQRT_HALF
    vals = np.where((groups >> 9) & 1, -mag, mag)

    q = np.zeros((n, 4), np.float64)
    # The kept components are the ones whose index is not `largest`, in order.
    rows = np.arange(n)[:, None]
    idx = np.argsort(np.arange(4)[None, :] == largest[:, None],
                     axis=1, kind="stable")[:, :3]
    q[rows, idx] = vals
    q[rows[:, 0], largest] = np.sqrt(np.maximum(0.0, 1.0 - (q ** 2).sum(1)))
    return q


def _quat_encode_smallest_three(q):
    n = len(q)
    largest = np.argmax(np.abs(q), axis=1)
    rows = np.arange(n)
    q = np.where(q[rows, largest][:, None] < 0, -q, q)   # keep the dropped one positive
    idx = np.argsort(np.arange(4)[None, :] == largest[:, None],
                     axis=1, kind="stable")[:, :3]
    vals = q[rows[:, None], idx]
    mag = np.clip(np.rint(np.abs(vals) / _SQRT_HALF * 511.0), 0, 511).astype(np.uint32)
    groups = (np.where(vals < 0, 1, 0).astype(np.uint32) << 9) | mag
    packed = (largest.astype(np.uint32) << 30) \
        | (groups[:, 0] << 20) | (groups[:, 1] << 10) | groups[:, 2]
    return packed.astype("<u4")


def _spz_stream_sizes(n, version, sh_dim):
    pos = 6 if version == 1 else 9
    rot = 3 if version < SPZ_SMALLEST_THREE_FROM else 4
    return {"positions": n * pos, "alphas": n, "colors": n * 3,
            "scales": n * 3, "rotations": n * rot, "sh": n * sh_dim * 3}


def _spz_decode(parts, n, version, sh_degree, fractional_bits) -> SplatSet:
    """Turn the six byte streams into a SplatSet."""
    if version == 1:
        pos = np.frombuffer(parts["positions"], "<f2", count=n * 3) \
            .reshape(n, 3).astype(np.float64)
    else:
        pos = _int24_decode(parts["positions"], n) / float(1 << fractional_bits)

    alpha = np.frombuffer(parts["alphas"], np.uint8, count=n).astype(np.float64) / 255.0
    col = np.frombuffer(parts["colors"], np.uint8, count=n * 3) \
        .reshape(n, 3).astype(np.float64) / 255.0
    f_dc = (col - 0.5) / COLOR_SCALE
    sc = np.frombuffer(parts["scales"], np.uint8, count=n * 3) \
        .reshape(n, 3).astype(np.float64) / 16.0 - 10.0

    if version < SPZ_SMALLEST_THREE_FROM:
        q = _quat_decode_first_three(parts["rotations"], n)
    else:
        q = _quat_decode_smallest_three(parts["rotations"], n)

    k = sh_coeffs_for_degree(sh_degree)
    if k:
        sh = np.frombuffer(parts["sh"], np.uint8, count=n * k * 3) \
            .reshape(n, k, 3).astype(np.float64)
        f_rest = ((sh - 128.0) / 128.0).astype(np.float32)
    else:
        f_rest = np.zeros((n, 0, 3), np.float32)

    return SplatSet(
        xyz=pos.astype(np.float32),
        f_dc=f_dc.astype(np.float32),
        f_rest=f_rest,
        opacity=_logit(alpha).astype(np.float32),
        scale=sc.astype(np.float32),
        rot=_xyzw_to_wxyz(_unit(q)).astype(np.float32),
    )


# A file states its own point count, and the streams are compressed, so a
# small crafted file could otherwise ask for tens of gigabytes. 2^27 points is
# far beyond any real scan (the large ones have a few million).
SPZ_MAX_POINTS = 1 << 27


def _spz_check_count(n):
    if n > SPZ_MAX_POINTS:
        raise ValueError("spz states %d points; more than %d is refused"
                         % (n, SPZ_MAX_POINTS))


def load_spz(path) -> SplatSet:
    """Load a Niantic ``.spz``, any version from 1 to 4."""
    splats, _ = load_spz_with_antialiasing(path)
    return splats


def load_spz_with_antialiasing(path) -> tuple[SplatSet, bool]:
    """Load the splats and the source file's antialiasing flag."""
    with open(path, "rb") as f:
        head = f.read(2)
    if head == b"\x1f\x8b":
        with gzip.open(path, "rb") as g:
            header = g.read(16)
            if len(header) < 16:
                raise ValueError("spz truncated: no header")
            magic, version, n = struct.unpack_from("<III", header, 0)
            sh_degree = header[12]
            _spz_check_count(n)
            if magic != NGSP_MAGIC or not 1 <= version <= 3 or sh_degree > 3:
                return _load_spz_legacy(header)       # raises the precise error
            sizes = _spz_stream_sizes(n, version, sh_coeffs_for_degree(sh_degree))
            # no more than the header promises, plus one byte to notice extra
            blob = header + g.read(sum(sizes.values()) + 1)
        return _load_spz_legacy(blob)
    with open(path, "rb") as f:
        return _load_spz_ngsp(f.read())


def _load_spz_legacy(blob: bytes) -> tuple[SplatSet, bool]:
    """Versions 1-3: a 16-byte header and six arrays, the lot gzipped."""
    if len(blob) < 16:
        raise ValueError("spz truncated: no header")
    magic, version, n = struct.unpack_from("<III", blob, 0)
    sh_degree, fractional_bits, flags, _reserved = struct.unpack_from("<BBBB", blob, 12)
    if magic != NGSP_MAGIC:
        raise ValueError("spz with wrong magic number 0x%08x" % magic)
    if not 1 <= version <= 3:
        raise ValueError("gzip-compressed spz states version %d; expected 1-3" % version)
    _spz_check_count(n)
    if sh_degree > 3:
        raise ValueError("spz states SH degree %d" % sh_degree)

    sizes = _spz_stream_sizes(n, version, sh_coeffs_for_degree(sh_degree))
    parts, off = {}, 16
    for name in _STREAMS:
        size = sizes[name]
        if off + size > len(blob):
            raise ValueError("spz truncated in stream %s (%d bytes expected, %d present)"
                             % (name, size, len(blob) - off))
        parts[name] = blob[off:off + size]
        off += size
    return _spz_decode(parts, n, version, sh_degree, fractional_bits), bool(flags & SPZ_FLAG_ANTIALIASED)


def _load_spz_ngsp(blob: bytes) -> tuple[SplatSet, bool]:
    """Version 4: a 32-byte header, a table of contents, zstd streams."""
    if len(blob) < 32:
        raise ValueError("spz truncated: no header")
    magic, version, n = struct.unpack_from("<III", blob, 0)
    sh_degree, fractional_bits, flags, n_streams = struct.unpack_from("<BBBB", blob, 12)
    toc_off, = struct.unpack_from("<I", blob, 16)
    if magic != NGSP_MAGIC:
        raise ValueError("spz with wrong magic number 0x%08x" % magic)
    if version < SPZ_ZSTD_FROM:
        raise ValueError("spz without gzip states version %d; below %d "
                         "they are gzip-compressed" % (version, SPZ_ZSTD_FROM))
    if version > SPZ_LATEST:
        raise ValueError("spz version %d is newer than this reader (%d)"
                         % (version, SPZ_LATEST))
    if sh_degree > 3:
        raise ValueError("spz states SH degree %d" % sh_degree)
    _spz_check_count(n)

    # Only the SH stream can be absent, so the count pins down which is which.
    expected = [s for s in _STREAMS if s != "sh" or sh_degree > 0]
    if n_streams != len(expected):
        raise ValueError("spz states %d streams; a file with SH degree %d has %d"
                         % (n_streams, sh_degree, len(expected)))

    try:
        import zstandard
    except ImportError as e:      # pragma: no cover - depends on the install
        raise RuntimeError("spz version 4 needs the 'zstandard' package "
                           "(pip install zstandard)") from e
    dctx = zstandard.ZstdDecompressor()

    sizes = _spz_stream_sizes(n, version, sh_coeffs_for_degree(sh_degree))
    toc = []
    for i in range(n_streams):
        comp, plain = struct.unpack_from("<QQ", blob, toc_off + i * 16)
        toc.append((comp, plain))
    off = toc_off + n_streams * 16

    parts = {name: b"" for name in _STREAMS}
    for name, (comp, plain) in zip(expected, toc):
        if off + comp > len(blob):
            raise ValueError("spz truncated in stream %s" % name)
        # checked before decompressing: the stated size bounds the output
        if plain != sizes[name]:
            raise ValueError("spz stream %s has %d bytes; %d points in version %d "
                             "need %d" % (name, plain, n, version, sizes[name]))
        data = dctx.decompress(blob[off:off + comp], max_output_size=plain)
        if len(data) != plain:
            raise ValueError("spz stream %s decompresses to %d bytes instead of %d"
                             % (name, len(data), plain))
        parts[name] = data
        off += comp
    return _spz_decode(parts, n, version, sh_degree, fractional_bits), bool(flags & SPZ_FLAG_ANTIALIASED)


def save_spz(path, s: SplatSet, version: int = 3, sh_degree: int | None = None,
             antialiased: bool = False) -> dict:
    """Write a Niantic ``.spz``. Returns what the quantisation had to do.

    ``version`` 3 is the default: version 2 is readable by more tools but
    mangles rotations near a half-turn (see the module docstring for the
    numbers). ``sh_degree`` caps the bands that are kept -- dropping from 3 to
    1 removes two thirds of the file, and indoors most people cannot see it.
    """
    if version not in (2, 3):
        raise ValueError("spz is written as version 2 or 3, not %r" % version)
    n = s.n
    if n == 0:
        raise ValueError("an empty spz is not written")

    deg = s.sh_degree if sh_degree is None else min(int(sh_degree), s.sh_degree)
    k = sh_coeffs_for_degree(deg)

    # 24 bits of fixed point reach +-2048 at the usual 12 fractional bits.
    # A scene wider than that would wrap around silently, so give up precision
    # instead; readers honour whatever the header says.
    #
    # How wide is measured on the splats, not on the farthest one: one
    # floater at 100,000 units took six bits from every other splat (steps
    # of 1/64 unit in place of 1/4096), one at infinity all twelve. The
    # extent is where 99.9 % of the splats lie; what is farther out is
    # clipped to the range and counted. A scene that is merely wide loses
    # nothing to that: if covering every splat costs one bit or none, every
    # splat is covered.
    def bits_for(extent):
        b = 12
        while b > 0 and extent * (1 << b) > 0x7FFFFF:
            b -= 1
        return b

    finite = np.isfinite(s.xyz).all(1)
    reach = np.abs(s.xyz[finite]).max(1).astype(np.float64) if finite.any() else np.zeros(1)
    fractional_bits = bits_for(float(np.percentile(reach, 99.9)))
    if bits_for(float(reach.max())) >= fractional_bits - 1:
        fractional_bits = bits_for(float(reach.max()))

    pos_q = np.rint(s.xyz.astype(np.float64) * (1 << fractional_bits))
    lost = np.isnan(pos_q)                       # no position at all: to the origin
    pos_q[lost] = 0.0
    # (counted in splats, not in coordinates: one splat far out on two axes is one)
    clipped = int(np.count_nonzero(((pos_q < -0x800000) | (pos_q > 0x7FFFFF) | lost).reshape(n, -1).any(1))) if n else 0
    positions = _int24_encode(pos_q).reshape(n, 9)

    alphas = _u8(s.alpha * 255.0)
    colors = _u8(s.f_dc.astype(np.float64) * (COLOR_SCALE * 255.0) + 127.5)
    scales = _u8((s.scale.astype(np.float64) + 10.0) * 16.0)

    q = _wxyz_to_xyzw(s.quat)
    rotations = (_quat_encode_first_three(q) if version < SPZ_SMALLEST_THREE_FROM
                 else _quat_encode_smallest_three(q))

    if k:
        # (N, K, 3) flattened is already spz's order: coefficient outer,
        # colour channel inner.
        sh = _u8(s.f_rest[:, :k, :].astype(np.float64) * 128.0 + 128.0)
        sh = sh.reshape(n, k * 3)
    else:
        sh = np.zeros((n, 0), np.uint8)

    flags = SPZ_FLAG_ANTIALIASED if antialiased else 0
    header = struct.pack("<IIIBBBB", NGSP_MAGIC, version, n,
                         deg, fractional_bits, flags, 0)

    # Level 6, not 9: on the hut (1.2 M splats, SH 1) level 9 took 9.2 s for
    # 20.36 MB, level 6 2.2 s for 20.48 MB. The SH bytes are noise to
    # deflate, and level 9 spends its time searching them for matches.
    # Written under the final name as far as gzip's own header goes (it
    # records the file name), so the bytes are what they were before the
    # file went through a temporary one.
    with replacing(path) as tmp:
        with open(tmp, "wb") as raw, \
                gzip.GzipFile(os.fspath(path), "wb", compresslevel=6, mtime=0, fileobj=raw) as g:
            g.write(header)
            for block in (positions, alphas, colors, scales,
                          np.asarray(rotations), sh):
                g.write(np.ascontiguousarray(block).tobytes())

    info = {"version": version, "shDegree": deg, "fractionalBits": fractional_bits,
            "positionsClipped": clipped, "bytes": os.path.getsize(path)}
    if fractional_bits < 12:
        info["warning"] = ("the scene is very large for .spz: positions are accurate to 1/%d "
                           "unit only (%d instead of 12 fractional bits)"
                           % (1 << fractional_bits, fractional_bits))
    return info
