"""Sidecar process: the editor's engine, driven over stdin/stdout.

Deliberately not an HTTP server. A localhost port would mean Chrome's Local
Network Access prompt, a firewall dialog on first run, a port to collide, and
an endpoint any other page on the machine could drive. Framed messages on the
pipe the parent already owns have none of that, and the process dies with its
parent.

Wire format, both directions:

    [u32 header_len][u32 payload_len][header JSON utf-8][payload bytes]

The header carries the command and its small arguments; the payload carries
whatever is large and numeric. Splat indices and attribute blocks go through as
raw little-endian arrays rather than JSON numbers -- 35k new Gaussians is 8 MB
of float32 and about 90 MB of JSON.
"""
from __future__ import annotations

import atexit
import errno
import io
import json
import linecache
import math
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import traceback
from contextlib import nullcontext

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from splatcore.document import Document           # noqa: E402
from splatcore.io import save_ply                 # noqa: E402
from splatcore.repair import RepairParams, RepairError  # noqa: E402

PROTOCOL = 1
MIRROR_PREFIX = "splatcore-view-"     # the viewport's PLY copies, in the temp directory
EDIT_COMMANDS = frozenset({
    "delete", "drop_nonfinite", "repair", "deep_repair", "unshadow", "solidify", "transform", "clone",
    "undo", "redo", "refine",
})


def _read_exactly(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def read_frame(stream):
    head = _read_exactly(stream, 8)
    if head is None:
        return None, None
    hl, pl = struct.unpack("<II", head)
    hdr = json.loads(_read_exactly(stream, hl).decode("utf-8")) if hl else {}
    payload = _read_exactly(stream, pl) if pl else b""
    return hdr, payload


def _plain(v):
    """A header value as JSON can carry it.

    numpy scalars and arrays (anything with ``tolist``) become Python's own
    types, and a NaN or an infinity becomes null. ``json.dumps`` writes those
    two bare, which is not JSON: the app's parser throws on the header and
    the call it belongs to never resolves.
    """
    if isinstance(v, dict):
        return {(k.item() if isinstance(k, np.generic) else k): _plain(x)
                for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, float):
        return float(v) if math.isfinite(v) else None
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        return _plain(v.tolist())
    return v


def write_frame(stream, header, payload=b""):
    try:
        # as it is, if it is JSON: nearly always, and a list of a million
        # indices is not walked a second time for nothing
        raw = json.dumps(header, allow_nan=False)
    except (TypeError, ValueError):
        raw = json.dumps(_plain(header), allow_nan=False)
    raw = raw.encode("utf-8")
    stream.write(struct.pack("<II", len(raw), len(payload)))
    stream.write(raw)
    if payload:
        stream.write(payload)
    stream.flush()


class Session:
    def __init__(self, out=None):
        self.doc: Document | None = None
        self.out = out                 # where progress events go
        self._mirror_dir = None        # holds the viewport's PLY of a .splat/.spz

    # ------------------------------------------------------------ commands
    def cmd_ping(self, h, p):
        return {"pong": True, "protocol": PROTOCOL,
                "python": sys.version.split()[0], "numpy": np.__version__}, b""

    def cmd_open(self, h, p):
        path = h["path"]
        t = time.time()
        # On a local until nothing can fail any more. The bounds, the SH
        # degree and the viewport's copy all can, and the session used to be
        # on the new document by then: the app showed the old file, the
        # engine edited the new one.
        d = Document.open(path)
        deg = d.base.sh_degree
        # Bounds over the positions that are numbers. One NaN made all of
        # them NaN; ``nonFinite`` says how many splats were left out.
        xyz = d.base.xyz
        ok = np.isfinite(xyz).all(1)
        if not ok.any():
            raise ValueError("the file contains no splats" if d.n_base == 0 else
                             "no position in the file is a number (all NaN or infinite)")
        bad = int(d.n_base - ok.sum())
        good = xyz[ok] if bad else xyz
        lo, hi = good.min(0).astype(np.float64), good.max(0).astype(np.float64)
        view_path = self._view_mirror(path, d)
        reply = {
            "n": d.n_base,
            "shDegree": deg,
            "bounds": {"min": lo.tolist(), "max": hi.tolist(),
                       "centre": ((lo + hi) * 0.5).tolist(),
                       "radius": float(np.linalg.norm(hi - lo) * 0.5)},
            "nonFinite": bad,
            "loadSeconds": round(time.time() - t, 3),
            "path": os.path.abspath(path),
            "format": d.source_format,
            "viewPath": view_path,
        }
        self.doc = d
        return reply, b""

    def _view_mirror(self, path, d) -> str:
        """A path the viewport can load. PLY passes straight through.

        PlayCanvas parses PLY, SOG and glTF -- not .splat or .spz. Rather
        than write a second reader in TypeScript and have two parsers to keep
        honest, the engine writes a PLY of what it just read and hands that
        over. Both sides then show the same splats in the same order, because
        neither the reader nor the PLY writer reorders anything.
        """
        if d.source_format == "ply":
            # The copy of the .splat or .spz open before this one is dead
            # too. It used to stay until the engine ended, a whole PLY
            # beside a document that never needed it.
            self._clear_mirror()
            return os.path.abspath(path)
        if self._mirror_dir is None or not os.path.isdir(self._mirror_dir):
            self._mirror_dir = tempfile.mkdtemp(prefix=MIRROR_PREFIX)
        name = os.path.splitext(os.path.basename(path))[0] + ".ply"
        out = os.path.join(self._mirror_dir, name)
        save_ply(out, d.base)
        # One mirror at a time: the previous file is dead the moment a new
        # document is open, and these run to a hundred megabytes. Removed
        # once the new one is written, so an open that fails here leaves
        # the old one where the app may still want it.
        self._clear_mirror(keep=name)
        return out

    def _clear_mirror(self, keep=None):
        """Remove the files in the mirror folder, all but ``keep``. The folder stays."""
        if self._mirror_dir is None:
            return
        try:
            names = os.listdir(self._mirror_dir)
        except OSError:
            return
        for old in names:
            if keep is not None and os.path.normcase(old) == os.path.normcase(keep):
                continue
            try:
                os.remove(os.path.join(self._mirror_dir, old))
            except OSError:
                pass

    def drop_mirror(self):
        """Remove the viewport's copy and its folder: nobody else knows of them.

        Called when the session ends. Every engine that opened a .splat or
        .spz used to leave its folder in the temp directory for good, a
        whole PLY each.
        """
        folder, self._mirror_dir = self._mirror_dir, None
        if folder:
            shutil.rmtree(folder, ignore_errors=True)

    def cmd_delete(self, h, p):
        d = self._doc()
        idx = np.frombuffer(p, dtype="<u4").astype(np.int64)
        e = d.delete(idx)
        return {"removed": _u32(e.removed), "alive": d.n_alive,
                "canUndo": d.can_undo, "canRedo": d.can_redo}, b""

    def cmd_drop_nonfinite(self, h, p):
        """Delete the live splats whose position is NaN or infinite: one undo step.

        ``open`` counts them (``nonFinite``) and leaves them in, and the
        search for debris refuses a document that holds any. Nobody can see
        or paint them, so they cannot be selected and deleted by hand. The
        reply is a delete's; with none to remove ``removed`` is empty and no
        step is made (an empty one would cost the redo stack for nothing).
        """
        d = self._doc()
        bad = _non_finite(d)
        if len(bad) == 0:
            return {"removed": [], "alive": d.n_alive,
                    "canUndo": d.can_undo, "canRedo": d.can_redo}, b""
        e = d.delete(bad)
        return {"removed": _u32(e.removed), "alive": d.n_alive,
                "canUndo": d.can_undo, "canRedo": d.can_redo}, b""

    def cmd_repair(self, h, p):
        d = self._doc()
        idx = np.frombuffer(p, dtype="<u4").astype(np.int64)
        params = _params(RepairParams, h.get("params"))
        _dump_request("repair", h, idx)
        t = time.time()
        # The painted volume, when the app sends one: {"capsules": [[x0, y0,
        # z0, x1, y1, z1, r, depth], ...], "subtract": [...], "view": [dx, dy,
        # dz]}. With it the hole is what was painted, splats or not.
        # "objects": grow the stroke to the object it covers first
        # (objsel.py; the app's "Objekt erkennen"); absent: the engine's default.
        # "engine": "image" | "copy" | "auto" picks the fill for this call
        # (Document.fill_engine; absent: the document's own, "auto").
        prev = d.fill_engine
        if h.get("engine") in ("image", "copy", "auto"):
            d.fill_engine = h["engine"]
        try:
            e = d.repair(idx, params, volume=_volume(h), objects=_objects(h))
        finally:
            d.fill_engine = prev
        payload, layout = _pack_added(d.generated, e.added)
        return {
            "removed": _u32(e.removed),
            "addedCount": e.added,
            "addedFirstIndex": d.n_total - e.added,
            "layout": layout,
            "info": _jsonable(e.info),
            "seconds": round(time.time() - t, 3),
            "alive": d.n_alive, "total": d.n_total,
            "canUndo": d.can_undo, "canRedo": d.can_redo,
        }, payload

    def _edit_reply(self, d, e, t0):
        payload, layout = _pack_added(d.generated, e.added)
        return {
            "removed": _u32(e.removed), "addedCount": e.added,
            "addedFirstIndex": d.n_total - e.added, "layout": layout,
            "info": _jsonable(e.info), "seconds": round(time.time() - t0, 3),
            "alive": d.n_alive, "total": d.n_total,
            "canUndo": d.can_undo, "canRedo": d.can_redo,
        }, payload

    def cmd_unshadow(self, h, p):
        from splatcore.unshadow import ShadowParams
        d = self._doc()
        idx = np.frombuffer(p, dtype="<u4").astype(np.int64)
        t = time.time()
        # params: "mode" "auto" (find the shadow) or "lift" (brighten by
        # "lift"), "strength", "edge". Keys this engine does not know are
        # dropped, not fatal (`_params`).
        e = d.unshadow(idx, _params(ShadowParams, h.get("params")), h.get("view"),
                       volume=_volume(h))
        return self._edit_reply(d, e, t)

    def cmd_solidify(self, h, p):
        from splatcore.solidify import SolidParams
        d = self._doc()
        idx = np.frombuffer(p, dtype="<u4").astype(np.int64)
        t = time.time()
        e = d.solidify(idx, h.get("view"), _params(SolidParams, h.get("params")))
        return self._edit_reply(d, e, t)

    def cmd_capabilities(self, h, p):
        """What this machine can actually do, so the UI can say so up front.

        Without loading anything. This used to load LaMa and CUDA here, at
        app start: three to four seconds during which the single-threaded
        engine could not open a file, a gigabyte of VRAM held for a session
        that might never use it, and on a fresh machine a silent 200 MB
        download. Whether the packages are importable is enough to say what
        is possible; the model loads on first use.
        """
        import importlib.util
        from splatcore import gpu
        # The GPU probe runs on its own thread from engine start; asked
        # before it is done, say so instead of holding up every other
        # command (an "open" right after launch waited 79-107 s under load).
        # The app asks again until the answer is final.
        if h.get("wait"):
            gpu.available()
        st = gpu.peek()
        if st is None:
            gpu.probe_in_background()
            return {"pending": True, "gpu": False, "gpuDetail": "checking the GPU ...",
                    "deep": False, "photoshop": False, "viewFill": False}, b""
        ok, why = st
        lama = importlib.util.find_spec("simple_lama_inpainting") is not None
        cv = importlib.util.find_spec("cv2") is not None
        inpainter = "LaMa" if lama else ("OpenCV" if cv else None)
        from splatcore import photoshop
        ps_ok, ps_why = photoshop.available()
        # The wide-view fill renders on the GPU; LaMa fills it, or Photoshop.
        view_lama = bool(ok and lama)
        view_ps = bool(ok and ps_ok)
        return {"gpu": ok, "gpuDetail": why,
                "photoshop": view_ps, "photoshopDetail": None if view_ps else
                (ps_why or "needs a GPU"),
                "viewFill": view_lama,
                "viewFillDetail": None if view_lama else
                ("needs a GPU" if not ok else "LaMa is not installed"),
                "inpainter": inpainter,
                "inpainterDetail": None if inpainter else
                "neither LaMa nor OpenCV is installed",
                "deep": bool(inpainter) or view_ps,
                # a CPU fallback exists, but it is minutes, not seconds
                "deepSeconds": 5 if ok else 120}, b""

    def cmd_deep_repair(self, h, p):
        from splatcore.deep import DeepParams
        d = self._doc()
        idx = np.frombuffer(p, dtype="<u4").astype(np.int64)
        params = _params(RepairParams, h.get("params"))
        dp = _params(DeepParams, h.get("deep"))
        _dump_request("deep", h, idx)
        rid = h.get("id")

        def progress(stage, frac):
            write_frame(self.out, {"event": "progress", "id": rid,
                                   "stage": stage, "frac": round(float(frac), 3)})

        # A picture filled elsewhere (Photoshop) and the camera it was
        # rendered from, in place of the inpainter: {"image": path,
        # "camera": path to its camera JSON}.
        reference = None
        ref = h.get("reference")
        if ref:
            import json as _json
            from PIL import Image as _Image
            cam = _json.load(open(ref["camera"], encoding="utf-8"))
            reference = {"image": np.asarray(_Image.open(ref["image"]).convert("RGB")),
                         "eye": cam["eye"], "target": cam["target"], "up": cam["up"],
                         "fov": cam["fov_deg_vertical"]}
        # What fills the picture: "lama" or "photoshop" on a wide view, or
        # "patch" -- the old tight view around the patch. Not named: what the
        # app offers first, LaMa where it runs here (Document.deep_repair).
        fill = h.get("fill") or None
        if reference is not None:
            fill = "patch"
        t = time.time()
        # "volume": what was painted (see cmd_repair); the masks follow it.
        # "views" / "sheet": how many views the wide fill renders, and
        # whether one call fills them all (viewfill.view_fill).
        e = d.deep_repair(idx, params, dp, on_progress=progress, reference=reference,
                          fill=fill, volume=_volume(h), views=int(h.get("views") or 1),
                          sheet=bool(h.get("sheet", True)), objects=_objects(h))
        payload, layout = _pack_added(d.generated, e.added)
        return {
            "removed": _u32(e.removed),
            "addedCount": e.added,
            "addedFirstIndex": d.n_total - e.added,
            "layout": layout,
            "info": _jsonable(e.info),
            "seconds": round(time.time() - t, 3),
            "alive": d.n_alive, "total": d.n_total,
            "canUndo": d.can_undo, "canRedo": d.can_redo,
        }, payload

    def cmd_refine(self, h, p):
        """The last quick fill, refined by the deep fill in place (`Document.refine_last`).

        Same splats, new colours: the reply carries the whole new tail
        (``addedFirstIndex``, ``addedCount``) for the viewport to swap in,
        and ``refined`` false with ``info.refine.rejected`` when the deep fill
        kept nothing. Still one undo step.

        ``refineKept``: the judge's verdict (`splatcore.judge`) -- false when
        the refine looked worse than the quick fill and was dropped, None
        when it was not judged; ``info.judge`` its scores and ``why``.
        ``judge`` false in the request swaps in without looking.
        """
        from splatcore.deep import DeepParams
        d = self._doc()
        # no progress frames: it runs behind the user's back, and the app
        # turns every progress frame into its busy overlay
        t = time.time()
        e = d.refine_last(_params(DeepParams, h.get("deep")), fill=h.get("fill") or "lama",
                          judge=bool(h.get("judge", True)))
        head = {"refined": bool(e.info.get("refined")), "info": _jsonable(e.info),
                "refineKept": e.info.get("refineKept"),
                "seconds": round(time.time() - t, 3), "alive": d.n_alive, "total": d.n_total,
                "canUndo": d.can_undo, "canRedo": d.can_redo}
        if not head["refined"]:
            return head, b""
        payload, layout = _pack_added(d.generated, e.added)
        head.update({"addedCount": e.added, "addedFirstIndex": d.n_total - e.added,
                     "layout": layout})
        return head, payload

    def cmd_transform(self, h, p):
        d = self._doc()
        idx = np.frombuffer(p, dtype="<u4").astype(np.int64)
        t = time.time()
        e = d.transform(idx, h.get("rotation", [1, 0, 0, 0]),
                        h.get("translation", [0, 0, 0]),
                        float(h.get("scale", 1.0)), h.get("pivot"))
        payload, layout = _pack_added(d.generated, e.added)
        return {"removed": _u32(e.removed), "addedCount": e.added,
                "addedFirstIndex": d.n_total - e.added, "layout": layout,
                "info": _jsonable(e.info), "seconds": round(time.time() - t, 3),
                "alive": d.n_alive, "total": d.n_total,
                "canUndo": d.can_undo, "canRedo": d.can_redo}, payload

    def cmd_clone(self, h, p):
        from splatcore.clone import clone_params
        d = self._doc()
        t = time.time()
        # Unknown keys (a painted ``volume``, options of a newer app) are
        # ignored rather than ending the stroke with a TypeError.
        params = clone_params(h.get("params"))
        # The camera's forward direction: which side of a surface is the
        # front, which geometry alone cannot tell on a floor or a wall.
        view = h.get("view")
        # A painted stroke arrives as many dabs in one call: one engine round
        # trip, one undo step, and later dabs see what earlier ones laid down.
        # ``picked``: which dabs' sources the app found on the surface under
        # the crosshair. The payload, when ``exclude`` says so, holds what
        # the crop preview hides (u32 indices).
        exclude = (np.frombuffer(p, dtype="<u4").astype(np.int64)
                   if p and h.get("exclude") == "payload" else None)
        if h.get("dabs"):
            pairs = [(np.asarray(a, float), np.asarray(b, float))
                     for a, b in h["dabs"]]
            # ``eye``: the camera, so the dabs' points go onto the surface
            # along the rays they were picked on
            e = d.clone_stroke(pairs, float(h["radius"]), params, view=view,
                               radii=h.get("radii"), picked=h.get("picked"),
                               exclude=exclude, eye=h.get("eye"))
        else:
            e = d.clone(np.asarray(h["source"], float),
                        np.asarray(h["target"], float),
                        float(h["radius"]), params, view=view)
        payload, layout = _pack_added(d.generated, e.added)
        return {
            "removed": _u32(e.removed), "addedCount": e.added,
            "addedFirstIndex": d.n_total - e.added, "layout": layout,
            "info": _jsonable(e.info), "seconds": round(time.time() - t, 3),
            "alive": d.n_alive, "total": d.n_total,
            "canUndo": d.can_undo, "canRedo": d.can_redo,
        }, payload

    def cmd_object_select(self, h, p):
        """What a repair would take for this selection: the object, grown.

        Changes nothing. ``view``: the painting camera's forward direction.
        Returns ``mode`` ("none" | "lift" | "fill"), ``indices`` (everything
        the repair would take or delete; the selection itself for "none"),
        ``object`` (the object alone) and the measurements, so the app can
        show the grown selection before the user presses anything.
        """
        d = self._doc()
        idx = np.frombuffer(p, dtype="<u4").astype(np.int64)
        t = time.time()
        r = d.object_selection(idx, view=h.get("view"), volume=_volume(h))
        return {"mode": r["mode"], "indices": _u32(r["indices"]), "object": _u32(r["object"]),
                "info": _jsonable(r["info"]), "seconds": round(time.time() - t, 3)}, b""

    def cmd_floaters(self, h, p):
        """Debris above a threshold, as document indices.

        ``up``: the scene's up direction in file coordinates, as the app's
        "Oben" toggle has it; -y when not sent.
        """
        d = self._doc()
        t = time.time()
        live, st = d.floater_stats(h.get("up"))
        thr = float(h.get("threshold", 0.3))
        local = st.select(thr)
        idx = live[local]
        det = st.detached[local]
        # Why they were picked: cut off from everything, hidden behind what
        # the cameras saw, or the fringe under a crown (a big lone blob seen
        # from the side and below). Without a GPU only the first. ``kept``:
        # hidden haze left in place because without it a surface no camera
        # saw would show (the ground under a hut).
        fringe = np.zeros(len(local), bool)
        if st.fringe is not None:
            fringe = ~det & (st.fringe[local] > st.base[local] * st.hidden[local])
        return {"indices": _u32(idx), "count": int(len(idx)),
                "total": int(d.n_alive),
                "histogram": st.histogram(),
                "medianSpacing": round(st.median_spacing, 5),
                "detached": int(det.sum()),
                "hidden": int((~det & ~fringe).sum()),
                "fringe": int(fringe.sum()),
                "kept": st.kept(thr),
                "gpu": bool(st.gpu),
                "seconds": round(time.time() - t, 3)}, b""

    def cmd_undo(self, h, p):
        d = self._doc()
        e = d.undo()
        if e is None:
            return {"applied": False, "canUndo": False, "canRedo": d.can_redo}, b""
        return {"applied": True, "kind": e.kind, "restored": _u32(e.removed),
                "droppedTail": e.added, "total": d.n_total, "alive": d.n_alive,
                "canUndo": d.can_undo, "canRedo": d.can_redo}, b""

    def cmd_redo(self, h, p):
        d = self._doc()
        e = d.redo()
        if e is None:
            return {"applied": False, "canUndo": d.can_undo,
                    "canRedo": d.can_redo}, b""
        payload, layout = (_pack_added(d.generated, e.added) if e.added
                           else (b"", {}))
        return {"applied": True, "kind": e.kind, "removed": _u32(e.removed),
                "addedCount": e.added, "addedFirstIndex": d.n_total - e.added,
                "layout": layout, "total": d.n_total,
                "alive": d.n_alive, "canUndo": d.can_undo,
                "canRedo": d.can_redo}, payload

    def cmd_save(self, h, p):
        d = self._doc()
        t = time.time()
        opts = {k: v for k, v in (h.get("options") or {}).items()
                if k in ("version", "sh_degree", "sort", "antialiased",
                         "write_normals")}
        info = d.save(h["path"], h.get("format"), **opts)
        return {**info, "path": os.path.abspath(h["path"]),
                "bytes": os.path.getsize(h["path"]),
                "seconds": round(time.time() - t, 3)}, b""

    def cmd_stats(self, h, p):
        d = self._doc()
        return {"base": d.n_base, "generated": d.generated.n,
                "alive": d.n_alive, "total": d.n_total,
                "canUndo": d.can_undo, "canRedo": d.can_redo,
                "undoDepth": len(d.undo_stack)}, b""

    def _doc(self) -> Document:
        if self.doc is None:
            raise RuntimeError("no document is open")
        return self.doc


def _params(cls, given):
    """The dataclass ``cls`` from the app's dict, without the keys it has no field for.

    The app grows options faster than the engine. An unknown key passed
    straight on was a TypeError before the brush did anything, and the user
    read "Internal error" for a setting the engine could have ignored.
    """
    import dataclasses
    known = {f.name for f in dataclasses.fields(cls) if f.init}
    return cls(**{k: v for k, v in (given or {}).items() if k in known})


def _dump_request(kind, h, idx):
    """With SPLAT_DUMP_REQUESTS=<folder>: keep each fill request to replay it offline."""
    folder = os.environ.get("SPLAT_DUMP_REQUESTS")
    if not folder:
        return
    try:
        os.makedirs(folder, exist_ok=True)
        k = len([f for f in os.listdir(folder) if f.endswith(".json")])
        stem = os.path.join(folder, "%02d_%s" % (k, kind))
        np.save(stem + ".npy", np.asarray(idx, np.int64))
        with open(stem + ".json", "w", encoding="utf-8") as fh:
            json.dump(_jsonable(h), fh)
    except Exception:                      # noqa: BLE001 - a debug aid never breaks a fill
        pass


def _volume(h):
    """The painted volume from a command header, read once; None if absent.

    Read here so a malformed volume is refused before any work starts, with
    the reason, instead of failing halfway through the fill.
    """
    v = h.get("volume")
    if not v:
        return None
    from splatcore.repair import as_volume
    return as_volume(v)


def _objects(h):
    """The header's "objects" switch: True, False, or None for the default."""
    v = h.get("objects")
    return None if v is None else bool(v)


def _u32(a):
    """Indices as a plain list -- small enough for JSON at brush scale."""
    return np.asarray(a, np.int64).astype(np.uint32).tolist()


def _non_finite(d):
    """Document indices of the live splats whose position is NaN or infinite."""
    bad = ~np.isfinite(d.base.xyz).all(1)
    if d.generated.n:
        bad = np.concatenate([bad, ~np.isfinite(d.generated.xyz).all(1)])
    return np.where(bad & d.alive)[0]


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _pack_added(generated, count):
    """Tail of ``generated`` as one little-endian float32 block.

    Interleaved per splat so the viewport can upload it without a transpose:
    position, DC colour, opacity, scale, rotation, then SH bands.
    """
    if count == 0:
        return b"", {}
    s = generated.take(np.arange(generated.n - count, generated.n))
    return _pack_set(s)


def _pack_set(s):
    count = s.n
    if count == 0:
        return b"", {}
    K = s.f_rest.shape[1]
    # f_rest goes across CHANNEL-MAJOR -- [R_0..R_K-1][G_...][B_...] -- the same
    # order the PLY uses, so the receiver can map column i straight onto
    # f_rest_i. Sending it coefficient-major instead swaps red with green two
    # coefficients in, which renders as vivid green and red confetti over the
    # patch and reads like a broken fill rather than a broken array index.
    parts = [s.xyz, s.f_dc, s.opacity[:, None], s.scale, s.rot,
             s.f_rest.transpose(0, 2, 1).reshape(count, 3 * K)]
    block = np.concatenate([p.astype("<f4") for p in parts], axis=1)
    layout = {"count": count, "stride": int(block.shape[1]),
              "fields": [["xyz", 0, 3], ["f_dc", 3, 3], ["opacity", 6, 1],
                         ["scale", 7, 3], ["rot", 10, 4], ["f_rest", 14, 3 * K]],
              "shCoeffs": K, "shOrder": "channel-major"}
    return block.tobytes(order="C"), layout


def _sweep_mirrors(keep=None, root=None, max_age=24 * 3600.0):
    """Remove the mirror folders that engines before this one left behind.

    An engine that is killed -- the app closing, a crash -- never gets to
    `Session.drop_mirror`, and its folder holds a whole PLY. Only folders by
    our prefix, in the temp directory, untouched for a day; ``keep()`` names
    the running session's own, which stays whatever its age. Errors are
    ignored: a folder that will not go today is tried again at the next
    start. Returns how many were removed.
    """
    gone = 0
    try:
        root = root or tempfile.gettempdir()
        with os.scandir(root) as it:
            found = [e.path for e in it if e.name.startswith(MIRROR_PREFIX)
                     and e.is_dir(follow_symlinks=False)]
    except OSError:
        return gone
    for folder in found:
        try:
            mine = keep() if keep else None
            if mine and os.path.normcase(os.path.abspath(mine)) == \
                    os.path.normcase(os.path.abspath(folder)):
                continue
            if time.time() - os.path.getmtime(folder) < max_age:
                continue
            shutil.rmtree(folder, ignore_errors=True)
            gone += not os.path.exists(folder)
        except OSError:
            pass
    return gone


def _os_error_text(e):
    """An OSError as a sentence for the status line: what failed, on which file.

    A recent file that has moved, a save into a folder that is gone, a file
    another program holds open, a full disk: all of them used to read
    "Internal error".
    """
    why = (e.strerror or str(e)).strip() or type(e).__name__
    name = e.filename
    if isinstance(name, bytes):
        name = os.fsdecode(name)
    if isinstance(e, FileNotFoundError):
        lead = "File or folder not found"
    elif isinstance(e, PermissionError):
        lead = "Access denied, the file is locked or read-only"
    elif e.errno == errno.ENOSPC:
        lead = "The disk is full"
    else:
        # not every OSError is about a file (a download that fails is one too)
        lead = "File error" if name else "System error"
    return "%s%s (%s)" % (lead, ': "%s"' % name if name else "", why)


_HERE = os.path.dirname(os.path.abspath(__file__))
_OWN_RAISE = ("raise ValueError(", "raise RuntimeError(")


def _own_refusal(e):
    """Where the engine itself refused, as "file.py:line"; None for anything else.

    A ValueError or RuntimeError is one of two things: a sentence the engine
    wrote for the user ("the selection is empty"), or something out of numpy,
    scipy or torch. The second needs its traceback in the log. The first
    happens all day and is no fault; with a traceback each, the log was
    mostly refusals. Told apart by where it was raised: the innermost frame
    lies in this file or in splatcore/ (not in the packages under .venv,
    which sit in this folder too) and the line there is a plain ``raise``
    of one of the two. Unsure, or no source to read: None, and the
    traceback is printed as before.
    """
    try:
        tb = e.__traceback__
        if tb is None:
            return None
        while tb.tb_next is not None:
            tb = tb.tb_next
        code = tb.tb_frame.f_code
        path = os.path.normcase(os.path.abspath(code.co_filename))
        own = (path == os.path.normcase(os.path.abspath(__file__))
               or path.startswith(os.path.normcase(os.path.join(_HERE, "splatcore")) + os.sep))
        if not own:
            return None
        line = linecache.getline(code.co_filename, tb.tb_lineno, tb.tb_frame.f_globals)
        if not line.strip().startswith(_OWN_RAISE):
            return None
        return "%s:%d" % (os.path.basename(path), tb.tb_lineno)
    except Exception:                                # noqa: BLE001 - then with traceback
        return None


def _release_gpu():
    """Hand the GPU memory torch keeps cached back between commands.

    torch holds on to what a command allocated, for the next one: after a
    fill that is memory other programs cannot have while the editor sits
    idle. Only when torch is loaded and CUDA is up in this process, asked
    with ``is_initialized`` -- a flag. ``is_available`` asks the driver, and
    while the start-up probe is still bringing CUDA up on its thread that
    could hold the command loop as the probe once did. Never an error.
    """
    try:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:                                # noqa: BLE001
        pass


def main():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    # The log is UTF-8, which is how the app reads it (sidecar.ts). Left to
    # itself Python takes the system code page for a piped stderr on
    # Windows, and the app showed every umlaut in a refusal as a
    # replacement character.
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:                                # noqa: BLE001 - not a text stream of that kind
        pass
    # Anything a library prints to stdout would corrupt the frame stream.
    sys.stdout = sys.stderr
    session = Session(stdout)
    atexit.register(session.drop_mirror)
    from splatcore import gpu as _gpu
    _gpu.probe_in_background()
    write_frame(stdout, {"event": "ready", "protocol": PROTOCOL, "pid": os.getpid()})
    # after "ready", on its own thread: it deletes gigabytes on a bad day
    threading.Thread(target=_sweep_mirrors, args=(lambda: session._mirror_dir,),
                     name="mirror-sweep", daemon=True).start()

    while True:
        try:
            hdr, payload = read_frame(stdin)
        except Exception:
            break
        if hdr is None:
            break
        rid = hdr.get("id")
        cmd = hdr.get("cmd", "")
        fn = getattr(session, "cmd_" + cmd, None)
        if fn is None:
            write_frame(stdout, {"id": rid, "ok": False,
                                 "error": "unknown command %r" % cmd})
            continue
        doc = session.doc
        editing = doc is not None and cmd in EDIT_COMMANDS

        def fail(e, error, kind, where=None, **more):
            """Log it, take back what the command left in the document, tell the app.

            ``where``: the engine's own refusal (`_own_refusal`), one line.
            """
            if where:
                print("[%s] refused: %s (%s)" % (cmd, error, where), file=sys.stderr)
            elif not isinstance(e, RepairError):
                print("[%s] %s%s: %s" % (cmd, "internal error: " if kind == "internal" else "",
                                         type(e).__name__, e), file=sys.stderr)
                traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
            session.doc = doc          # an open whose reply failed keeps the old document
            write_frame(stdout, {"id": rid, "ok": False, "error": error, "kind": kind, **more})

        try:
            with doc.edit_transaction(cmd) if editing else nullcontext():
                result, out = fn(hdr, payload)
                result = dict(result)
                result.update({"id": rid, "ok": True})
                write_frame(stdout, result, out)
        except RepairError as e:
            # The engine's own refusals, written for the user: verbatim.
            fail(e, str(e), type(e).__name__)
        except OSError as e:
            fail(e, _os_error_text(e), type(e).__name__)
        except (ValueError, RuntimeError) as e:
            # Verbatim as well: most are the engine's own sentences ("the
            # selection is empty"), and those get one line in the log. One out
            # of numpy, scipy or torch reads the same on the wire; it keeps
            # its traceback (`_own_refusal` tells them apart).
            fail(e, str(e) or type(e).__name__, type(e).__name__, where=_own_refusal(e))
        except Exception as e:                       # noqa: BLE001
            # Unexpected: the user gets a sentence, the log gets the details.
            # "IndexError: index 812345 is out of bounds for axis 0" in the
            # status line told nobody anything they could act on.
            fail(e, 'Internal error in "%s". Details are in the engine '
                    'log.' % cmd, "internal", detail=type(e).__name__)
        _release_gpu()

    session.drop_mirror()


if __name__ == "__main__":
    main()
