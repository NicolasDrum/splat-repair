"""The edit model: what the editor considers "the document", and undo.

Splat indices are stable for the lifetime of a document. The loaded file
occupies ``[0, n_base)`` and never changes; everything the repair brush
generates is appended after it and also never changes. Editing only moves the
``alive`` flags. That means:

* an index means the same splat forever, so the UI and this process can talk
  about splats by number without a shared ordering protocol;
* undo is restoring flags and truncating the generated tail, not recomputing;
* the 130 MB of loaded attributes is never rewritten, only masked.

This mirrors how the viewport holds its data, so the two stay in step with a
few thousand integers crossing the pipe instead of megabytes of geometry.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .io import SplatSet, load_ply, save_ply
from .formats import load_any, save_any, sniff
from .repair import NothingToFill, RepairParams, RepairResult, repair


DEFAULT_UP = (0.0, -1.0, 0.0)       # the 3DGS / COLMAP convention: y points down


def _unit_up(up) -> tuple:
    """``up`` as a hashable unit 3-vector; the default for anything unusable."""
    if up is None:
        return DEFAULT_UP
    try:
        v = np.asarray(up, float).ravel()
    except (TypeError, ValueError):
        return DEFAULT_UP
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-9:
        return DEFAULT_UP
    v = v / np.linalg.norm(v)
    return tuple(round(float(x), 6) for x in v)


@dataclass
class Edit:
    """One undoable step."""

    kind: str                       # "delete" | "repair"
    removed: np.ndarray             # indices switched off by this edit
    added: int = 0                  # how many splats were appended
    info: dict = field(default_factory=dict)
    # The splats this edit appended, kept aside while it is undone. Redo used
    # to refuse anything that appended, on the grounds that a repair is
    # stochastic and re-running it would hand the user different geometry.
    # True, but beside the point: nothing has to be re-run. Undo already has
    # the exact splats in its hand at the moment it throws them away, so it
    # keeps them instead, and redo is exact for every kind of edit.
    tail: Optional[SplatSet] = None


def _stroke_tone(dabs, p):
    """A stroke's tone shift as one smooth field over the points ``p``.

    ``dabs``: ``(target, shift at its centre, cells it was measured on,
    radius)`` per dab. Each dab measures its own shift on a ring a brush
    wide, and over a sign that ring is half sign, shadow and the wall behind
    it: along the hut's crossbar neighbouring dabs came out 0.15 apart (38
    grey levels), and overlapping copies of them were the blotches in the
    band. Dabs far off the rest are dropped; the others are averaged with
    a Gaussian two radii wide, weighted by how much each one saw.
    """
    t = np.array([d[0] for d in dabs], float)
    d = np.array([d[1] for d in dabs], float)
    w = np.array([max(d_[2], 1.0) for d_ in dabs], float)
    r = float(np.median([d_[3] for d_ in dabs]))
    dev = np.linalg.norm(d - np.median(d, 0), axis=1)
    mad = 1.4826 * float(np.median(dev))
    ok = dev <= 3.0 * max(mad, 0.02)
    if ok.sum() >= 2:
        t, d, w = t[ok], d[ok], w[ok]
    dd = ((p[:, None, :] - t[None, :, :]) ** 2).sum(2)
    k = w[None, :] * np.exp(-0.5 * dd / (2.0 * r) ** 2) + 1e-12 * w[None, :]
    return (k @ d) / k.sum(1, keepdims=True)


def _take_parts(parts, starts, keys, idx) -> SplatSet:
    """``idx`` (ascending) out of the splat sets ``parts`` laid end to end, as one set.

    What joining them all with `SplatSet.concat` and then ``take(idx)`` gives,
    without building the whole: ``starts[k]`` is where part ``k`` begins,
    ``keys`` the extra properties they all have. Each array is joined in one
    go, so all of them at once (``idx`` everything) costs one copy, not one
    per part.
    """
    idx = np.asarray(idx, np.int64)
    which = np.searchsorted(starts, idx, side="right") - 1
    cut = np.flatnonzero(np.diff(which)) + 1             # where idx moves on to another part
    pieces = [parts[which[a]].take(idx[a:b] - starts[which[a]])
              for a, b in zip(np.r_[0, cut], np.r_[cut, len(idx)]) if b > a]
    if not pieces:
        pieces = [parts[0].take(np.zeros(0, np.int64))]

    def join(get):
        return np.concatenate([get(p) for p in pieces])
    return SplatSet(xyz=join(lambda p: p.xyz), f_dc=join(lambda p: p.f_dc),
                    f_rest=join(lambda p: p.f_rest), opacity=join(lambda p: p.opacity),
                    scale=join(lambda p: p.scale), rot=join(lambda p: p.rot),
                    extra={k: join(lambda p: p.extra[k]) for k in keys})


def _dump_fill(live, local, view, volume):
    """With SPLAT_FILL_DUMP set to a folder, write what a fill was asked to do.

    A fill made by hand in the app (a stroke, a view) cannot be made again in
    a script; with this it can be replayed and timed outside the app
    (`tests/replay_fill.py`). Off unless the variable is set.
    """
    import os
    folder = os.environ.get("SPLAT_FILL_DUMP")
    if not folder:
        return
    try:
        import json
        import time
        spec = volume if isinstance(volume, dict) else (volume.to_dict() if hasattr(volume, "to_dict") else None)
        os.makedirs(folder, exist_ok=True)
        np.savez_compressed(os.path.join(folder, "fill_%d.npz" % int(time.time() * 1000)),
                            selection=np.asarray(local, dtype=np.int64), view=np.asarray(view, dtype=np.float64),
                            volume=json.dumps(spec), n=np.int64(live.n))
    except Exception:                                      # noqa: BLE001  (a debugging aid must not break a fill)
        pass


class Document:
    def __init__(self, splats: SplatSet, path: Optional[str] = None):
        self.base = splats
        self.n_base = splats.n
        self.generated = splats.take(np.zeros(0, np.int64))
        self.alive = np.ones(splats.n, bool)
        self.path = path
        self.source_format = "ply"   # what it was opened from, for the export default
        self.undo_stack: list[Edit] = []
        self.redo_stack: list[Edit] = []
        self._floaters = None        # measured lazily, re-used across thresholds
        self._refine = None          # the last quick fill, for `refine_last`

    # ---------------------------------------------------------------- data
    @classmethod
    def open(cls, path) -> "Document":
        """Open a splat file of any supported format, identified by content.

        The extension is not consulted: a ``.spz`` that has been renamed to
        ``.ply`` still opens, and a gzipped PLY gets told what it is instead
        of failing somewhere inside the parser.
        """
        kind = sniff(path)
        doc = cls(load_any(path), str(path))
        doc.source_format = kind
        return doc

    @property
    def n_total(self) -> int:
        return self.n_base + self.generated.n

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    def combined(self) -> SplatSet:
        """Base plus generated, in index order (including dead splats)."""
        return self.base.concat(self.generated) if self.generated.n else self.base

    def live_splats(self) -> SplatSet:
        return self.combined().take(self.alive)

    def _live_indices(self, indices) -> np.ndarray:
        """``indices`` that name a live splat of this document; the rest dropped.

        The viewport's selection mask can outlive the splats it was made
        over -- an undo truncates the generated tail -- and an index past
        the end used to raise IndexError from inside an edit.
        """
        idx = np.asarray(indices, np.int64).ravel()
        idx = idx[(idx >= 0) & (idx < self.n_total)]
        return idx[self.alive[idx]]

    # --------------------------------------------------------------- edits
    def floater_stats(self, up=None):
        """Debris measurements for the live set, computed once.

        The k-d tree query is the expensive part, so it is kept until the
        geometry actually changes. Moving the threshold slider then costs a
        comparison rather than a rebuild. ``up`` is the scene's up direction
        in file coordinates (the app's "Oben" toggle), -y if not given; it is
        part of the key, since the hidden-haze test splits its cameras by it.
        """
        from .cleanup import analyse
        up = _unit_up(up)
        key = (self.n_alive, self.n_total, len(self.undo_stack), up)
        if self._floaters is None or self._floaters[0] != key:
            live = np.where(self.alive)[0]
            s = self.live_splats()
            self._need_positions(s, "Floaters cannot be found with them")
            self._floaters = (key, live, analyse(s, up=up))
        return self._floaters[1], self._floaters[2]

    @staticmethod
    def _need_positions(s, consequence):
        """Refuse in words when a position is NaN or infinite.

        The k-d tree refuses such a cloud with "data must be finite, check
        for nan or inf values", which reached the status line as it was.
        """
        bad = int(np.count_nonzero(~np.isfinite(s.xyz).all(1)))
        if bad:
            raise ValueError("Splats without a valid position (NaN or infinite): %d. %s."
                             % (bad, consequence))

    def delete(self, indices) -> Edit:
        idx = self._live_indices(indices)
        if len(idx) == 0:
            # Nothing live among them (a mask older than the last edit, a
            # second press): an empty step would be an undo that does
            # nothing, paid for with the whole redo stack.
            raise ValueError("the selection is empty")
        e = Edit("delete", idx, 0, {"removed": int(len(idx))})
        self._commit_edit(e)
        return e

    @staticmethod
    def _with_volume(params, volume):
        """``params`` carrying the painted volume (the app's dict), if one came."""
        if volume is None:
            return params
        import dataclasses
        return dataclasses.replace(params or RepairParams(), volume=volume)

    # Grow a stroke over an object to the whole object before a repair
    # (objsel.py): its sides, faint edges and drop shadow go with it, and an
    # object standing clear of a whole surface is only deleted. Off: the
    # selection is repaired exactly as painted.
    object_select: bool = True

    def _object(self, live, local, params, objects):
        """`objsel.find_object` for this stroke, or None (off, or nothing found)."""
        if not (self.object_select if objects is None else objects) or len(local) == 0:
            return None
        from .objsel import find_object
        view = getattr(params, "view", None) if params is not None else None
        try:
            got = find_object(live, local, view)
        except Exception as ex:           # never let the helper cost the user the repair
            import sys
            print("objsel:", repr(ex), file=sys.stderr)
            return None
        return None if got.mode == "none" else got

    def _lift(self, live_ids, got) -> Edit:
        """An object standing clear of a whole surface: delete it, nothing else."""
        e = self.delete(live_ids[got.indices])
        e.info.update({"mode": "delete", "standoff": got.info.get("standoff"),
                       "object": got.info})
        return e

    def object_selection(self, indices, view=None, volume=None) -> dict:
        """What the repair would take for this selection: the object, grown.

        For the app to show before anything changes. ``indices``: document
        indices; returns document indices (``indices``: everything the
        repair takes or deletes, ``object``: the object itself), ``mode``
        and the measurements.
        """
        idx = self._live_indices(indices)
        live_ids = np.where(self.alive)[0]
        pos_of = np.full(self.n_total, -1, np.int64)
        pos_of[live_ids] = np.arange(len(live_ids))
        local = pos_of[idx]
        local = local[local >= 0]
        params = RepairParams(view=list(view)) if view is not None else None
        got = self._object(self.live_splats(), local, params, True)
        if got is None:
            return {"mode": "none", "indices": idx, "object": np.zeros(0, np.int64), "info": {}}
        return {"mode": got.mode, "indices": live_ids[got.indices],
                "object": live_ids[got.object], "info": got.info}

    def repair(self, indices, params: Optional[RepairParams] = None,
               volume=None, objects: Optional[bool] = None) -> Edit:
        """Run the repair brush over ``indices`` and fold the result in.

        ``volume``: the painted volume (see `repair.as_volume`). With it a
        selection may be empty -- painted wholly over a hole in the scan.
        ``objects``: grow the stroke to the object it covers first
        (`objsel`); None takes `object_select`.
        """
        params = self._with_volume(params, volume)
        idx = self._live_indices(indices)
        if len(idx) == 0 and getattr(params, "volume", None) is None:
            raise ValueError("the selection is empty")

        # The engine works on live geometry only; map indices through.
        live_ids = np.where(self.alive)[0]
        pos_of = np.full(self.n_total, -1, np.int64)
        pos_of[live_ids] = np.arange(len(live_ids))
        local = pos_of[idx]
        local = local[local >= 0]

        live = self.live_splats()
        got = self._object(live, local, params, objects)
        if got is not None and got.mode == "lift":
            return self._lift(live_ids, got)
        if got is not None:
            local = got.indices
        res = self._image_fill(live, local, params)
        if res is None:
            # why the picture fill gave up, if it was tried: it goes into
            # the result of what runs in its place
            note = self._image_note
            try:
                res = repair(live, local, params)
            except NothingToFill as why:
                e = self._delete_instead(idx, why)
                if note:
                    e.info["imageFallback"] = note
                return e
            if note:
                res.info["imageFallback"] = note
        if got is not None:
            res.info["object"] = got.info
        removed, added = self._finish(live, res, params)
        removed = live_ids[removed]
        e = Edit("repair", removed, added.n, res.info)
        # kept for `refine_last`: the quick fill the deep fill would start from
        # (the picture fill is already what a refine would aim for)
        refine = None if res.info.get("engine") == "image" else \
            {"edit": e, "res": res, "local": local, "params": params}
        self._commit_edit(e, added)
        self._refine = refine
        return e

    # Two steps finish every new skin, whichever engine made it
    # (`skin.finish`). ``clear_under``: what the engine left of the old scene
    # under its new skin goes with the edit (`under.py`); it showed through
    # gaps, at a slant and from afar. ``backing_layer``: the skin gets a sparse
    # layer of large splats behind it (`lod.py`), for viewers that stop
    # drawing small splats: without it a fill disappears when zoomed out.
    # Off: the engine's result as it is. What each did is in the edit's
    # ``info["under"]`` and ``info["lod"]``.
    clear_under: bool = True
    backing_layer: bool = True

    def _finish_skin(self, live, removed, added, view, spacing, info, face=None, signed=False):
        """`skin.finish` on what an engine made, by this document's switches.

        Returns (removed, added) over ``live`` as the engine's are, with what
        was cleared and the coarse layer in them; ``info`` gets the counts.
        ``signed``: the new splats' third axis points out of the surface (the
        picture engines'; copied Gaussians are turned to face the user).
        A step that fails leaves the engine's result as it is.
        """
        if not (self.clear_under or self.backing_layer) or added.n == 0:
            return removed, added
        from .skin import finish, spacing as scan_spacing
        try:
            # The scan's spacing, from the file as it was loaded where it has
            # splats there: over an earlier fill's dense skin the engine's own
            # measure comes out several times too small, and "eight spacings
            # under the skin" would end above that fill's coarse layers.
            skin = added if face is None else added.take(np.asarray(face, bool))
            if skin.n:
                c = np.median(skin.xyz.astype(np.float64), 0)
                reach = 2.5 * float(np.percentile(np.linalg.norm(skin.xyz.astype(np.float64) - c, axis=1), 90))
                if spacing:
                    reach = min(reach, 60.0 * float(spacing))       # (around a large fill's middle is enough)
                was = scan_spacing(self.base.xyz, self.base.alpha, c, reach)
                if was is not None:
                    spacing = max(float(spacing or 0.0), was)
            removed2, added2, got = finish(live, removed, added, view, spacing, face=face, signed=signed,
                                           clear=self.clear_under, back=self.backing_layer)
        except Exception as ex:           # never let the finish cost the user the edit
            import sys
            print("skin:", repr(ex), file=sys.stderr)
            return removed, added
        for key in ("under", "lod"):
            if got.get(key) is not None:
                info[key] = dict(got[key], spacing=float(spacing))
        info["finishSeconds"] = got.get("seconds")
        if "removed" in info:
            info["removed"] = int(len(removed2))
        if "added" in info:
            info["added"] = int(added2.n)
        return removed2, added2

    @staticmethod
    def _skin_of(res, params):
        """What `skin.finish` needs to know of a fill (`RepairResult`): (view, spacing, face)."""
        view = getattr(params, "view", None) if params is not None else None
        ctx = res.context or {}
        if view is None and ctx.get("basis") is not None:
            # no line of sight: the copy engine's own frame says where out is
            view = -np.asarray(ctx["basis"][2], float) * float(ctx.get("outward", 1.0) or 1.0)
        face = None
        if res.backing is not None and len(res.backing) == res.added.n and np.any(res.backing):
            face = ~np.asarray(res.backing, bool)      # (the slab the copy engine lays behind its face is no skin)
        return view, res.info.get("spacing") or ctx.get("spacing"), face

    def _finish(self, live, res, params):
        """`_finish_skin` for a fill (`RepairResult`): (removed, added) over ``live``."""
        view, spacing, face = self._skin_of(res, params)
        return self._finish_skin(live, res.removed, res.added, view, spacing, res.info, face,
                                 signed=res.info.get("engine") == "image")

    # Which engine closes a hole. "image": the picture fill (`imfill.py`:
    # fill the surface's own picture, lift it to splats), where a GPU is and
    # the stroke came with a line of sight. "copy": the transplant of whole
    # Gaussians (`repair.py`). "auto": the picture fill where it can run.
    fill_engine: str = "auto"
    _image_note: Optional[str] = None      # why the last picture fill gave up, if it did

    def _image_fill(self, live, local, params):
        """The picture fill's result as a `RepairResult`, or None to use the copy engine.

        When it was tried and failed, ``_image_note`` holds the reason and
        the traceback is in the log. It used to fail without a trace: the
        copy engine's result came back as if nothing had happened, whether
        the picture fill had declined the stroke or run out of GPU memory.
        """
        self._image_note = None
        if self.fill_engine == "copy":
            return None
        view = getattr(params, "view", None)
        if view is None:
            return None
        from . import gpu
        if not gpu.available()[0]:
            return None
        from . import imfill
        from .repair import as_volume
        _dump_fill(live, local, view, getattr(params, "volume", None))
        try:
            r = imfill.fill(live, local, view,
                            imfill.ImFillParams(seed=int(getattr(params, "seed", 0) or 0)),
                            volume=as_volume(getattr(params, "volume", None)))
        except (ValueError, RuntimeError) as ex:
            import sys
            import traceback
            print("imfill: the image fill failed, the copying fill takes over",
                  file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self._image_note = str(ex) or type(ex).__name__
            return None
        return RepairResult(removed=r.removed, added=r.added, info=dict(r.info))

    def refine_last(self, deep_params=None, fill: str = "lama", on_progress=None,
                    judge: bool = True) -> Edit:
        """Refine the last quick fill with the deep fill, in place: one undo step.

        "Repair" answers at once with the quick fill; where a GPU and LaMa
        are there, the app then asks for this. The deep fill starts from that
        very quick fill (its geometry, its removed splats) and only recolours
        its splats, so the count and every index stay what the viewport
        already has: the new tail replaces the old one and the edit on the
        undo stack is the same edit, now refined. ``info["refined"]`` says
        whether it was; a fit the guard rejects leaves the quick fill.

        A refine the guard lets through still has to look better than the
        quick fill (`judge.judge`): both are rendered against the untouched
        ring around them, and the quick fill stays unless the refine wins by
        a margin. ``info["refineKept"]`` is that verdict (None: not judged,
        e.g. no GPU to look with -- the refine then swaps in as before),
        ``info["judge"]`` the scores and why. ``judge=False``: swap in
        whenever the guard lets it.

        Refuses when the last edit is no longer that quick fill.
        """
        from .deep import DeepParams
        from .viewfill import view_fill
        st = getattr(self, "_refine", None)
        if not st or not self.undo_stack or self.undo_stack[-1] is not st["edit"]:
            raise ValueError("nothing to refine: the last change is not a Fast Fill")
        e, res = st["edit"], st["res"]
        first = self.n_total - e.added
        n_back = e.added - res.added.n      # the coarse layer behind the fill (`_finish`), last in the tail
        # the live set the quick fill was made on
        alive_b = self.alive[:first].copy()
        alive_b[e.removed] = True
        live_b = self.combined().take(np.where(alive_b)[0])
        r2 = view_fill(live_b, st["local"], st["params"], source=fill,
                       deep=deep_params or DeepParams(), base=res, on_progress=on_progress)
        same = (r2.added.n == res.added.n and len(r2.removed) == len(res.removed))
        info = dict(e.info)
        info.update({"refined": bool(r2.info.get("deep")) and same, "refineKept": None,
                     "refine": {k: r2.info.get(k) for k in (
                           "deep", "fill_source", "fit", "offAxisGeo", "offAxisDeep",
                           "rejected", "grain_from_copy", "view_seconds",
                           "fromPicture", "pictureBound", "keptCopy")}})
        if not info["refined"]:
            e.info = info
            return e
        if judge:
            from .judge import judge as look
            try:
                v = look(live_b, res.removed, res.added, r2.added, res.context,
                         view=getattr(st["params"], "view", None), backing=res.backing)
            except Exception as ex:      # never cost the user the refine over the judge
                import sys
                print("judge:", repr(ex), file=sys.stderr)
                v = {"keep": None, "why": "comparison failed: %s" % ex}
            info["judge"] = v
            info["refineKept"] = v.get("keep")
            if v.get("keep") is False:
                info["refined"] = False
                e.info = info
                self._refine = None
                return e
        g0 = first - self.n_base
        tail = r2.added
        if n_back > 0:
            # the layer takes the refined colours; its splats stay the ones they
            # were (same cells: the refine moves nothing), so every index holds
            back = self.generated.take(np.arange(self.generated.n - n_back, self.generated.n))
            try:
                from .skin import finish
                view, spacing, face = self._skin_of(res, st["params"])
                # (all the edit removed, what was cleared under the fill included, as places in ``live_b``)
                gone = np.searchsorted(np.where(alive_b)[0], np.asarray(e.removed, np.int64))
                again = finish(live_b, gone, r2.added, view, e.info.get("lod", {}).get("spacing") or spacing,
                               face=face, signed=False, clear=False)[1]
                if again.n - r2.added.n == n_back:
                    back = again.take(np.arange(r2.added.n, again.n))
            except Exception as ex:       # the layer as it was is still a layer
                import sys
                print("skin:", repr(ex), file=sys.stderr)
            tail = tail.concat(back)
        self.generated = (self.generated.take(np.arange(g0)).concat(tail) if g0
                          else tail)
        e.info = info
        self._refine = None
        self._floaters = None
        return e

    def _delete_instead(self, idx, why) -> Edit:
        """Painted over something standing clear of a whole surface: delete it.

        The antenna pole in front of the hut's wall: once its splats are gone
        the wall behind is complete, so there is no hole and the repair has
        nothing to do -- but the user who pressed it wants the pole gone.
        Something painted on the surface itself (standoff under two
        spacings) is refused instead: deleting that would only thin it.
        """
        # (a stroke that holds no live splat has nothing to delete either)
        if why.standoff < 2.0 or len(idx) == 0:
            raise why
        e = self.delete(idx)
        e.info.update({"mode": "delete", "standoff": round(why.standoff, 1)})
        return e

    def _apply_local(self, kind, indices, run) -> Edit:
        """Run an engine function over ``indices`` of the live set; fold it in.

        Every brush that edits in place works the same way: map the selection
        to live positions, let the function return what it took out and what
        it put back, and record that as one undoable step.
        """
        idx = self._live_indices(indices)
        if len(idx) == 0:
            raise ValueError("the selection is empty")
        live_ids = np.where(self.alive)[0]
        pos_of = np.full(self.n_total, -1, np.int64)
        pos_of[live_ids] = np.arange(len(live_ids))
        local = pos_of[idx]
        res = run(self.live_splats(), local[local >= 0])
        removed = live_ids[res.removed]
        e = Edit(kind, removed, res.added.n, res.info)
        self._commit_edit(e, res.added)
        return e

    def solidify(self, indices, view=None, params=None) -> Edit:
        """Make the painted surface as solid as the surface around it."""
        from .solidify import solidify as run, SolidParams
        return self._apply_local("solidify", indices,
                                 lambda s, l: run(s, l, view, params or SolidParams()))

    def unshadow(self, indices, params=None, view=None, volume=None) -> Edit:
        """Lift a shadow, and take out a coloured reflection, in the painted surface.

        ``volume``: the painted volume (see `repair.as_volume`). When no
        index came the live splats inside it are the selection; "Brighten"
        also takes the painted area from it (`unshadow._picture_share`).
        """
        from .unshadow import unshadow as run, ShadowParams
        if volume is not None and len(self._live_indices(indices)) == 0:
            inside = np.asarray(volume.inside(self.combined().xyz), bool)
            indices = np.where(inside & self.alive)[0]
        return self._apply_local("unshadow", indices,
                                 lambda s, l: run(s, l, params or ShadowParams(), view,
                                                  volume=volume))

    def deep_repair(self, indices, params: Optional[RepairParams] = None,
                    deep_params=None, on_progress=None, reference=None,
                    fill: Optional[str] = None, volume=None, views: int = 1,
                    sheet: bool = True, objects: Optional[bool] = None) -> Edit:
        """Repair, then let a 2-D inpainter decide the patch's appearance.

        ``fill``: "patch" (a tight view, the inpainter guided by the copy),
        or "lama" / "photoshop" (a wide view filled by that, see viewfill).
        None takes what the app offers first: "lama" where this machine can
        run the wide view, else "patch" (`viewfill.default_fill`). With a
        ``reference`` picture it is always the picture.

        A quick fill made face by face is deep-filled face by face on every
        path, and a deep step that cannot run keeps the quick fill.

        ``volume``: the painted volume; the masks are then what was painted.
        ``views`` / ``sheet``: on the wide view, how many views to fill and
        whether in one call over a sheet (`viewfill.view_fill`).
        """
        from .deep import DeepParams
        from .viewfill import SOURCES, view_fill, deep_patch, default_fill
        if fill is None:
            fill = "patch" if reference is not None else default_fill()
        if fill != "patch" and fill not in SOURCES:
            raise ValueError("fill must be patch, lama or photoshop, not %r" % fill)

        params = self._with_volume(params, volume)
        idx = self._live_indices(indices)
        if len(idx) == 0 and getattr(params, "volume", None) is None:
            raise ValueError("the selection is empty")
        live_ids = np.where(self.alive)[0]
        pos_of = np.full(self.n_total, -1, np.int64)
        pos_of[live_ids] = np.arange(len(live_ids))
        local = pos_of[idx]
        local = local[local >= 0]

        live = self.live_splats()
        # An object standing clear of a whole surface is deleted, as in
        # `repair`. Otherwise the grown selection goes, as in `repair`: with
        # the painted mask alone the EHBO letters' sides, their drop shadow
        # and their dark red backs under the wall's skin stayed, showed
        # through the fill and drew the letters' outlines. On the hut's cross
        # the grown footprint once left a faint pale cross; since the fit
        # starts from the picture's colour (`deep._colour_from_picture`) it
        # comes out cleaner than with the painted mask.
        got = self._object(live, local, params, objects)
        if got is not None and got.mode == "lift":
            return self._lift(live_ids, got)
        if got is not None:
            local = got.indices
        try:
            if fill in SOURCES and reference is None:
                res = view_fill(live, local, params, source=fill,
                                deep=deep_params or DeepParams(), on_progress=on_progress,
                                views=max(int(views or 1), 1), sheet=bool(sheet))
            else:
                if on_progress:
                    on_progress("geometry", 0.05)
                base = repair(live, local, params)
                res = deep_patch(live, local, params, deep_params or DeepParams(),
                                 base=base, on_progress=on_progress, reference=reference)
        except NothingToFill as e:
            return self._delete_instead(idx, e)
        if got is not None:
            res.info["object"] = got.info
        removed, added = self._finish(live, res, params)
        removed = live_ids[removed]
        e = Edit("repair", removed, added.n, res.info)
        self._commit_edit(e, added)
        return e

    def clone(self, source, target, radius, params=None, view=None) -> Edit:
        """Stamp the surface around ``source`` onto the surface around ``target``."""
        return self.clone_stroke([(source, target)], radius, params, view=view)

    # Which engine stamps. "image": the picture stamp (`imstamp.py`: copy the
    # stroke in the surface's own picture, lift it to splats), where a GPU is
    # and the stroke came with a line of sight. "copy": the stamp that copies
    # whole Gaussians dab by dab (`clone.py`). "auto": the picture stamp where
    # it can run. A stroke may ask for one itself (`CloneParams.engine`).
    clone_engine: str = "auto"
    _stamp_note: Optional[str] = None      # why the last picture stamp gave up, if it did

    def _image_stamp(self, pairs, radius, params, view, radii, picked, exclude, eye):
        """The stroke through the picture stamp, as an `Edit`; None to use the copying stamp.

        When it was tried and failed, ``_stamp_note`` holds the reason and
        the traceback is in the log.
        """
        self._stamp_note = None
        want = getattr(params, "engine", "auto") or "auto"
        if want == "auto":
            want = self.clone_engine
        if want == "copy" or view is None or not len(pairs):
            return None
        from . import gpu
        if not gpu.available()[0]:
            return None
        from . import imfill, imstamp
        ids = np.where(self.alive)[0]
        live = self.live_splats()
        self._need_positions(live, "The Clone Stamp cannot work with them")
        if exclude is not None and len(exclude):
            # what the crop preview hides is neither copied nor surface
            ex = np.asarray(exclude, np.int64).ravel()
            shown = ~np.isin(ids, ex[(ex >= 0) & (ex < self.n_total)])
            ids, live = ids[shown], live.take(shown)
        targets = np.array([np.asarray(t, float) for _, t in pairs]).reshape(-1, 3)
        own = [] if radii is None else [float(r) for r in radii]
        rads = np.array([own[k] if k < len(own) else (own[-1] if own else float(radius))
                         for k in range(len(pairs))])
        # The scan's own spacing there, from the file as it was loaded: under
        # an earlier stamp's dense splats it can no longer be measured, and
        # each stroke's picture would be several times finer than the last.
        c = targets.mean(0)
        reach = float(np.linalg.norm(targets - c, axis=1).max()) + 2.5 * float(rads.max())
        spacing = imfill._spacing(self.base.xyz.astype(np.float64), self.base.alpha, c, reach)
        P = imstamp.ImStampParams(feather=float(params.feather), color_match=float(params.color_match))
        sources = np.array([np.asarray(a, float) for a, _ in pairs]).reshape(-1, 3)
        align = None
        if getattr(params, "align", False):
            # the source, nudged so its pattern meets the surface around the
            # stroke, as for the copying stamp (`clone.stroke_alignment`)
            from .clone import stroke_alignment
            try:
                shift, align = stroke_alignment(live, live.xyz.astype(np.float64), None, sources,
                                                targets, rads, np.asarray(view, float))
            except Exception as exc:                      # noqa: BLE001
                shift, align = np.zeros(3), {"why": str(exc)}
            sources = sources + shift
        try:
            r = imstamp.stamp(live, sources, targets, rads, view, P, eye=eye, picked=picked,
                              spacing=spacing)
        except Exception as ex:                           # noqa: BLE001
            # Whatever it was (no surface there, no memory on the card, a
            # case nobody thought of): the stroke is not lost, the copying
            # stamp does it, and the reason goes into its result.
            import sys
            import traceback
            print("imstamp: the image stamp failed, the copying stamp takes over",
                  file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self._stamp_note = str(ex) or type(ex).__name__
            return None
        info = dict(r.info)
        removed, added = self._finish_skin(live, r.removed, r.added, view, spacing or info.get("spacing"), info,
                                           signed=True)
        removed = ids[removed]
        if align is not None:
            info["align"] = align
        e = Edit("clone", removed, int(added.n), info)
        self._commit_edit(e, added)
        return e

    def clone_stroke(self, pairs, radius, params=None, view=None,
                     radii=None, picked=None, exclude=None, eye=None) -> Edit:
        """Stamp a whole painted stroke as one edit.

        Two engines can do it (`clone_engine`). Where a GPU is and the stroke
        came with a line of sight, the picture stamp takes the whole stroke
        as one picture (`_image_stamp`, `imstamp.py`); what is said below is
        the copying stamp, which also takes over when the picture stamp
        gives up. ``info["engine"]`` says which one it was, and
        ``info["imageFallback"]`` why the picture stamp did not.

        A stroke is a list of ``(source, target)`` pairs, one per dab. They
        are applied in order, so a later dab sees what the earlier ones laid
        down and the overlaps join instead of stacking -- and the whole
        stroke is a single step on the undo stack, which is what a user who
        dragged once expects to get back from one Ctrl+Z.

        **The stroke works on a copy of itself.** Dabs a quarter brush apart
        overlap, so each one deletes much of what the previous one laid. That
        used to happen in the document: the edit then listed as "removed"
        splats that it had itself appended, the viewport -- which applies
        removals before appending -- kept showing them, and redo indexed past
        the end of the array and lost the edit. Now the dabs run against a
        local working set, only what survives the stroke is appended, and
        only splats that existed before it count as removed.

        **The source is where the crosshair is.** ``picked[k]`` says the app
        found dab ``k``'s source on the surface under its crosshair, which
        keeps its screen distance from the brush the way Photoshop's does;
        that point is used as it is. It used to be a world offset carried
        from dab to dab and turned by the change of the target's normal. On
        the hut's red cross those normals swung 20-40 degrees between dabs,
        the source swung with them, and 18 of 26 dabs were skipped as "the
        source leaves the surface". The carry is kept for dabs without a
        pick (crosshair off the object), now with normals twice the brush
        wide and restarted from the last real source.

        **One neighbourhood per dab.** One k-d tree per stroke; each dab sees
        only the splats around its own source and target.

        **A sign is painted out by the stroke, not by one dab.** A dab that
        finds something standing proud under the brush says so; the dabs
        within two radii of it are painting the same sign out, however little
        of it is left under their own brush. And the tone match is evened out
        over the stroke (``_stroke_tone``) instead of jumping from dab to dab.

        ``exclude``: document indices the stamp must neither copy nor count
        as surface -- what the crop preview hides. ``eye``: the camera's
        position, so each dab's points go onto the surface along their rays.
        """
        from scipy.spatial import cKDTree
        from .clone import clone as run_clone, CloneParams, stroke_alignment
        from .repair import fit_tangent_frame
        from .shtools import rotation_between

        params = params or CloneParams()
        view = None if view is None else np.asarray(view, float)
        # The picture stamp does the stroke as one picture; what follows is
        # the copying stamp, for when that cannot run or gives up.
        done = self._image_stamp(pairs, radius, params, view, radii, picked, exclude, eye)
        if done is not None:
            return done
        base_ids = np.where(self.alive)[0]
        base = self.live_splats()
        self._need_positions(base, "The Clone Stamp cannot work with them")
        bx = base.xyz.astype(np.float64)
        tree = cKDTree(bx)
        b_alive = np.ones(base.n, bool)
        hidden = np.zeros(base.n, bool)
        if exclude is not None and len(exclude):
            pos_of = np.full(self.n_total, -1, np.int64)
            pos_of[base_ids] = np.arange(len(base_ids))
            ex = np.asarray(exclude, np.int64).ravel()
            ex = pos_of[ex[(ex >= 0) & (ex < self.n_total)]]
            hidden[ex[ex >= 0]] = True
            b_alive &= ~hidden
        solid = base.alpha > 0.2
        # What the dabs laid down so far, part by part; joined once after the
        # stroke. Joining after every dab copied the whole stroke again each
        # time. ``x_xyz``: the parts' positions, for the dabs' neighbourhoods.
        parts, starts, x_keys = [], [], []
        x_xyz, x_n = np.zeros((0, 3), np.float32), 0
        x_alive = np.zeros(0, bool)
        info: dict = {}
        rot_seen, shift_seen, src_rgb, copy_rgb = [], [], [], []
        targets = np.array([np.asarray(t, float) for _, t in pairs]).reshape(-1, 3)
        # one radius per dab; a list that ends early (an app a step out of
        # sync) carries its last radius on, and none at all is ``radius``
        own = [] if radii is None else [float(r) for r in radii]
        rads = np.array([own[k] if k < len(own) else (own[-1] if own else float(radius))
                         for k in range(len(pairs))])

        def normal_at(p, r):
            """Plane through the opaque splats near ``p``, facing the camera."""
            nb = np.asarray(tree.query_ball_point(p, r), np.int64)
            nb = nb[b_alive[nb]] if len(nb) else nb
            # A translucent scan -- the hut, 8 % of splats above alpha 0.5 --
            # has too few opaque splats in a dab to fit a plane, and every
            # such dab was skipped. Take the faint ones too, by opacity.
            if solid[nb].sum() >= 12:
                nb = nb[solid[nb]]
            if len(nb) < 12:
                return None, None
            o, B, _ = fit_tangent_frame(bx[nb], w=base.alpha[nb])
            n = B[2]
            if view is not None and float(np.dot(n, view)) > 0:
                n = -n
            return o, n

        # The source, nudged by a few millimetres so its pattern meets the
        # wall around the stroke (``stroke_alignment``): one shift for the
        # whole stroke, which keeps its dabs in step with each other.
        if params.align and len(pairs):
            try:
                shift, ainfo = stroke_alignment(
                    base, bx, b_alive, [np.asarray(a, float) for a, _ in pairs],
                    targets, rads, view)
            except Exception as exc:                      # noqa: BLE001
                shift, ainfo = np.zeros(3), {"why": str(exc)}
            info["align"] = ainfo
            if np.any(shift):
                pairs = [(np.asarray(a, float) + shift, b) for a, b in pairs]
        offset, n_prev = None, None
        sign_at, sign_chroma = [], None
        x_tone, tone_dabs = [], []
        for k, (source, target) in enumerate(pairs):
            target = np.asarray(target, float)
            rk = rads[k]
            o_t, n_t = normal_at(target, 2.0 * rk)
            if n_t is None:
                info["skipped"] = info.get("skipped", 0) + 1
                info.setdefault("skippedWhy", "there is no surface at the target")
                continue
            if view is None and n_prev is not None and float(np.dot(n_prev, n_t)) < 0:
                n_t = -n_t
            if picked is not None and k < len(picked) and picked[k]:
                src = np.asarray(source, float)
                o_s, n_s = normal_at(src, 2.0 * rk)
            else:
                if offset is None:
                    offset = np.asarray(source, float) - target
                    offset -= n_t * float(np.dot(offset, n_t))
                else:
                    offset = rotation_between(n_prev, n_t) @ offset
                    offset -= n_t * float(np.dot(offset, n_t))
                # onto the surface: the plane through what lies around the guess
                guess = target + offset
                o_s, n_s = normal_at(guess, 2.0 * rk)
                if n_s is None:
                    # The target stands proud of the surface the source lies
                    # on, so the guess hangs in the air. Look for that surface
                    # along the normal, keeping the offset across it.
                    dd, ii = tree.query(guess, k=24, distance_upper_bound=4.0 * rk)
                    ii = ii[np.isfinite(dd)]
                    ii = ii[b_alive[ii]] if len(ii) else ii
                    if len(ii) >= 6:
                        guess = guess + n_t * float(np.dot(bx[ii].mean(0) - guess, n_t))
                        o_s, n_s = normal_at(guess, 2.0 * rk)
                src = guess if n_s is None else \
                    guess - n_s * float(np.dot(guess - o_s, n_s))
            n_prev = n_t
            # A carried source more than 60 degrees off has left the surface;
            # one the user points at may face up to 80 degrees away -- the
            # stamp turns it -- but not away from the camera.
            was_picked = picked is not None and k < len(picked) and bool(picked[k])
            if n_s is None or float(np.dot(n_s, n_t)) < (0.17 if was_picked else 0.5):
                info["skipped"] = info.get("skipped", 0) + 1
                info.setdefault("skippedWhy", "the source leaves the surface"
                                if not was_picked else
                                "the source points too far away from the surface")
                continue
            # where the source really is carries on to the next dab
            offset = src - target
            offset -= n_t * float(np.dot(offset, n_t))

            reach = 3.0 * rk
            near = np.unique(np.concatenate([
                np.asarray(tree.query_ball_point(src, reach), np.int64),
                np.asarray(tree.query_ball_point(target, reach), np.int64)]))
            near = near[b_alive[near]]
            sub = base.take(near)
            ex = np.zeros(0, np.int64)
            if parts and x_alive.any():
                ex = np.where(x_alive)[0]
                exy = x_xyz[ex].astype(np.float64)
                close = ((np.linalg.norm(exy - src, axis=1) < reach)
                         | (np.linalg.norm(exy - target, axis=1) < reach))
                ex = ex[close]
                if len(ex):
                    sub = sub.concat(_take_parts(parts, starts, x_keys, ex))
            # Painting over a sign is a property of the stroke, not of one
            # dab: after the first dab the brush is mostly its own copy. Two
            # radii from a dab that found one, this one is still over it.
            over = bool(sign_at) and float(np.min(np.linalg.norm(
                np.asarray(sign_at) - target, axis=1))) < 2.0 * rk
            try:
                res = run_clone(sub, src, target, rk, params, view=view,
                                footprint=(targets, rads), eye=eye, over_sign=over,
                                sign_chroma=sign_chroma)
            except Exception as exc:                      # noqa: BLE001
                # One dab landing somewhere without a surface must not throw
                # away the rest of the stroke.
                info["skipped"] = info.get("skipped", 0) + 1
                info.setdefault("skippedWhy", str(exc))
                continue
            if res.info.get("proudShare", 0.0) > 0.05:
                sign_at.append(target)
            if res.info.get("signChroma") is not None:
                sign_chroma = res.info["signChroma"]
            rem = np.asarray(res.removed, np.int64)
            in_base = rem < len(near)
            b_alive[near[rem[in_base]]] = False
            x_alive[ex[rem[~in_base] - len(near)]] = False
            if parts and res.added.f_rest.shape[1] != parts[0].f_rest.shape[1]:
                raise ValueError("the splat sets have different SH degrees")
            # the extra properties every part has, as `concat` would keep them
            x_keys = [k for k in (x_keys if parts else res.added.extra) if k in res.added.extra]
            starts.append(x_n)
            parts.append(res.added)
            if x_n + res.added.n > len(x_xyz):
                grown = np.empty((max(2 * len(x_xyz), x_n + res.added.n, 4096), 3), np.float32)
                grown[:x_n] = x_xyz[:x_n]
                x_xyz = grown
            x_xyz[x_n:x_n + res.added.n] = res.added.xyz
            x_n += res.added.n
            x_alive = np.concatenate([x_alive, np.ones(res.added.n, bool)])
            tn = res.context.get("tone") if res.context else None
            x_tone.append(tn if tn is not None else np.zeros((res.added.n, 3)))
            if "toneCentre" in res.info:
                tone_dabs.append((target, np.asarray(res.info["toneCentre"], float),
                                  float(res.info.get("toneCells", 0)), rk))
            rot_seen.append(res.info.get("rotation", 0.0))
            shift_seen.append(res.info.get("seamShift", 0.0))
            info["proud"] = info.get("proud", 0) + int(res.info.get("proud", 0))
            info.setdefault("seam", res.info.get("seam"))
            info["shRotated"] = res.info.get("shRotated", False)
            info["source"] = [float(x) for x in src]
            nb = np.asarray(tree.query_ball_point(src, rk), np.int64)
            nb = nb[solid[nb]] if len(nb) else nb
            if len(nb):
                src_rgb.append(base.rgb[nb].mean(0))
            if "copyRgb" in res.info:
                copy_rgb.append(res.info.pop("copyRgb"))

        if not parts:
            raise ValueError(info.get("skippedWhy", "nothing could be stamped"))
        extra = _take_parts(parts, starts, x_keys, np.arange(x_n))
        if len(tone_dabs) >= 2 and params.color_match > 0:
            # One tone for the stroke, not one per dab (see _stroke_tone):
            # each copied splat trades the shift its own dab gave it for the
            # stroke's at its place.
            xt = np.concatenate(x_tone, 0)
            live = np.where(x_alive)[0]
            moved = np.abs(xt[live]).sum(1) > 0
            if moved.any():
                ii = live[moved]
                tfield = _stroke_tone(tone_dabs, extra.xyz[ii].astype(np.float64))
                from .io import SH_C0
                extra.f_dc[ii] = (extra.f_dc[ii].astype(np.float64)
                                  + (params.color_match * tfield - xt[ii]) / SH_C0
                                  ).astype(np.float32)
        kept = extra.take(np.where(x_alive)[0])
        gone = ~b_alive & ~hidden
        removed = base_ids[gone]
        # For checks: where the new splats landed and what colour they carry,
        # next to what was there and what the source showed.
        if kept.n:
            # "where it was painted" is what the camera saw: across the
            # view, not along it, where relief and haze legitimately spread
            kx = kept.xyz.astype(np.float64)
            dd, jj = cKDTree(targets).query(kx)
            if view is not None and np.linalg.norm(view) > 0:
                vv = view / np.linalg.norm(view)
                rel = kx - targets[jj]
                dd = np.linalg.norm(rel - np.outer(rel @ vv, vv), axis=1)
            info["landed"] = round(float(np.mean(dd <= 1.15 * rads[jj])), 3)
        if copy_rgb:
            info["newRgb"] = [round(float(v), 3) for v in np.mean(copy_rgb, 0)]
        if gone.any():
            info["oldRgb"] = [round(float(v), 3) for v in base.rgb[gone].mean(0)]
        if src_rgb:
            info["srcRgb"] = [round(float(v), 3) for v in np.mean(src_rgb, 0)]
        n_own = kept.n
        if kept.n and view is not None:
            # (what the crop preview hides counts as gone here: it is neither cleared nor surface)
            from .skin import spacing as skin_spacing
            c = targets.mean(0)
            reach = float(np.linalg.norm(targets - c, axis=1).max()) + 2.5 * float(rads.max())
            sp = skin_spacing(self.base.xyz, self.base.alpha, c, reach)
            more, kept = self._finish_skin(base, np.where(~b_alive)[0], kept, view, sp, info)
            gone = gone.copy()
            gone[more] = True
            gone &= ~hidden
            removed = base_ids[gone]
        info["mode"] = "clone"
        info["engine"] = "copy"
        if self._stamp_note:
            info["imageFallback"] = self._stamp_note
        info["dabs"] = len(pairs) - int(info.get("skipped", 0))
        info["removed"] = int(len(removed))
        info["added"] = int(kept.n)
        info["discarded"] = int(extra.n - n_own)
        # over the whole stroke, not whichever dab happened to come first
        info["rotation"] = round(float(np.max(rot_seen)), 2) if rot_seen else 0.0
        info["seamShift"] = round(float(np.median(shift_seen)), 3) if shift_seen else 0.0
        e = Edit("clone", removed, int(kept.n), info)
        self._commit_edit(e, kept)
        return e

    def transform(self, indices, rotation, translation, scale, pivot=None) -> Edit:
        """Move, turn and scale the selected splats.

        Implemented as "hide the originals, append transformed copies" rather
        than editing in place. The loaded file then stays exactly as it arrived
        for the whole session -- the viewport never has to re-upload its 130 MB
        -- and undo is the same flag flip every other edit uses instead of a
        second, inverse transform that would slowly drift.
        """
        from . import transform as xf

        idx = self._live_indices(indices)
        if len(idx) == 0:
            raise ValueError("the selection is empty")

        moving = self.combined().take(idx)
        if pivot is None:
            pivot = moving.xyz.astype(np.float64).mean(0)
        info = xf.apply(moving, np.arange(moving.n), rotation, translation,
                        scale, pivot)

        e = Edit("transform", idx, moving.n, dict(info))
        self._commit_edit(e, moving)
        return e

    def _commit_edit(self, e: Edit, added: Optional[SplatSet] = None):
        """Prepare the mask and generated splats before publishing a new edit."""
        generated = self.generated
        if added is not None:
            generated = generated.concat(added) if generated.n else added
        alive = (np.concatenate([self.alive, np.ones(e.added, bool)])
                 if e.added else self.alive.copy())
        alive[e.removed] = False
        redo = []
        self.undo_stack.append(e)
        self.generated = generated
        self.alive = alive
        self.redo_stack = redo
        self._floaters = None

    @contextmanager
    def edit_transaction(self, command):
        """Hold the previous state until an edit's reply has been written.

        Edit operations replace the arrays. It is enough to retain their old
        references and the history entries affected by this command.
        """
        state = self.generated, self.alive, self._floaters, self._refine
        depth = len(self.undo_stack)
        undo = self.undo_stack.copy() if command == "undo" else self.undo_stack
        redo = self.redo_stack.copy() if command in ("undo", "redo") else self.redo_stack
        active = self.redo_stack if command == "redo" else self.undo_stack
        edit = active[-1] if active else None
        if edit is not None:
            info, tail = edit.info, edit.tail
        try:
            yield
        except BaseException:
            self.generated, self.alive, self._floaters, self._refine = state
            self.undo_stack, self.redo_stack = undo, redo
            while len(self.undo_stack) > depth:
                self.undo_stack.pop()
            if edit is not None:
                edit.info, edit.tail = info, tail
            raise

    def undo(self) -> Optional[Edit]:
        if not self.undo_stack:
            return None
        e = self.undo_stack[-1]
        generated, tail = self.generated, e.tail
        if e.added:
            keep = self.generated.n - e.added
            # Keep the tail rather than drop it: this is what makes redo exact.
            tail = self.generated.take(np.arange(keep, self.generated.n))
            generated = self.generated.take(np.arange(keep))
        alive = self.alive[:self.n_base + generated.n].copy()
        alive[e.removed] = True
        self.redo_stack.append(e)
        self.undo_stack.pop()
        e.tail = tail
        self.generated, self.alive = generated, alive
        self._floaters = None
        return e

    def redo(self) -> Optional[Edit]:
        """Replay an undone edit, exactly as it was.

        Nothing is recomputed -- the appended splats were put aside by
        ``undo``, so even a stochastic repair comes back the same.
        """
        if not self.redo_stack:
            return None
        e = self.redo_stack[-1]
        generated = self.generated
        if e.added:
            if e.tail is None or e.tail.n != e.added:
                # Should not happen; refuse rather than corrupt the document.
                return None
            generated = generated.concat(e.tail) if generated.n else e.tail
        alive = (np.concatenate([self.alive, np.ones(e.added, bool)])
                 if e.added else self.alive.copy())
        alive[e.removed] = False
        self.undo_stack.append(e)
        self.redo_stack.pop()
        e.tail = None
        self.generated, self.alive = generated, alive
        self._floaters = None
        return e

    @property
    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    # -------------------------------------------------------------- output
    def save(self, path, fmt: Optional[str] = None, **opts) -> dict:
        """Write the surviving splats. Format from ``fmt``, else the extension.

        Returns what was written, including whatever the chosen format had to
        give up -- the caller is expected to pass that on, because "saved"
        reads very differently when half the radiance field went with it.
        """
        s = self.live_splats()
        info = save_any(path, s, fmt, **opts) or {}
        return {"written": s.n, "shDegree": s.sh_degree, **info}
