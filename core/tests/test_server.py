"""Drive the sidecar exactly as the app will: spawn it, talk framed stdio."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
from io import BytesIO, StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(HERE)
sys.path.insert(0, CORE)
from splatcore.io import SplatSet, SH_C0, save_ply, load_ply   # noqa: E402

PY = os.path.join(CORE, ".venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable


class Client:
    def __init__(self):
        self.p = subprocess.Popen(
            [PY, os.path.join(CORE, "server.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # The engine's log is read as it comes, as the app reads it. Left in
        # the pipe it stops the engine at its 4097th byte, and the engine
        # logs a traceback for every command it refuses: this test alone is
        # at 1.8 kB.
        self._log = []
        self._logger = threading.Thread(
            target=lambda: self._log.append(self.p.stderr.read()), daemon=True)
        self._logger.start()
        self._id = 0
        hdr, _ = self.read()
        assert hdr.get("event") == "ready", hdr

    def read(self):
        head = self.p.stdout.read(8)
        if not head or len(head) < 8:
            self._logger.join(timeout=10)
            err = b"".join(self._log).decode("utf-8", "replace")
            raise RuntimeError("sidecar died\n" + err)
        hl, pl = struct.unpack("<II", head)
        hdr = json.loads(self.p.stdout.read(hl).decode("utf-8")) if hl else {}
        payload = self.p.stdout.read(pl) if pl else b""
        return hdr, payload

    def call(self, cmd, payload=b"", **kw):
        self._id += 1
        h = dict(kw); h.update({"cmd": cmd, "id": self._id})
        raw = json.dumps(h).encode("utf-8")
        self.p.stdin.write(struct.pack("<II", len(raw), len(payload)))
        self.p.stdin.write(raw)
        if payload:
            self.p.stdin.write(payload)
        self.p.stdin.flush()
        self.progress = []
        while True:
            hdr, body = self.read()
            # progress arrives on the same pipe; it is not the reply
            if hdr.get("event") == "progress":
                self.progress.append(hdr)
                continue
            return hdr, body

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def make_ply(path, n=16000, extent=5.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-extent, extent, size=(n, 2))
    xyz = np.column_stack([xy, np.zeros(n)]).astype(np.float32)
    tone = 0.5 + 0.2 * np.sin(xy[:, 0] * 5.0) * np.cos(xy[:, 1] * 5.0)
    rgb = np.column_stack([tone, tone * 0.9, tone * 0.8])
    s = SplatSet(
        xyz=xyz, f_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
        f_rest=rng.normal(0, 0.01, (n, 15, 3)).astype(np.float32),
        opacity=np.full(n, 2.5, np.float32),
        scale=np.full((n, 3), np.log(0.05), np.float32),
        rot=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)))
    save_ply(path, s)
    return s


def idx_payload(a):
    return np.asarray(a, np.uint32).astype("<u4").tobytes()


def inprocess_checks(src):
    """What the server hands on, checked without a pipe.

    The camera direction each brush sends must reach the engine as sent --
    the app now sends the view the selection was made from, not the camera
    at the moment the button is pressed -- and an unexpected error must
    reach the user as a sentence, with the details in the log only.
    """
    sys.path.insert(0, CORE)
    import io as _io
    import server
    from splatcore.document import Document
    from splatcore import viewfill
    seen = {}
    sess = server.Session()
    sess.doc = Document(src)
    d = sess.doc
    v = [0.25, -0.5, 0.83]

    class Stop(Exception):
        pass

    def spy(name):
        def f(*a, **k):
            seen[name] = (a, k)
            raise Stop()
        return f
    d.repair = spy("repair")
    d.unshadow = spy("unshadow")
    d.solidify = spy("solidify")
    d.deep_repair = spy("deep_repair")
    pay = idx_payload([1, 2, 3])
    for cmd, hdr in (("repair", {"params": {"view": v}}),
                     ("deep_repair", {"params": {"view": v}, "fill": "patch"}),
                     ("unshadow", {"view": v}),
                     ("solidify", {"view": v})):
        try:
            getattr(sess, "cmd_" + cmd)(hdr, pay)
        except Stop:
            pass
    assert seen["repair"][0][1].view == v, seen["repair"]
    assert seen["deep_repair"][0][1].view == v and seen["deep_repair"][1]["fill"] == "patch"
    assert seen["unshadow"][0][2] == v, seen["unshadow"]
    assert seen["solidify"][0][1] == v, seen["solidify"]
    # no fill named: what the app offers first
    try:
        sess.cmd_deep_repair({"params": {}}, pay)
    except Stop:
        pass
    assert seen["deep_repair"][1]["fill"] is None, seen["deep_repair"][1]
    print("  view        repair, deep, shadow, solidify: the view direction arrives; "
          "default fill %s" % viewfill.default_fill())

    # an unexpected error: a German sentence on the wire, the details in the log
    out, err = _io.BytesIO(), _io.StringIO()
    stdin = _io.BytesIO()
    raw = json.dumps({"cmd": "boom", "id": 7}).encode("utf-8")
    stdin.write(struct.pack("<II", len(raw), 0) + raw)
    stdin.seek(0)
    server.Session.cmd_boom = lambda self, h, p: [][1]
    class Pipe:
        def __init__(self, b):
            self.buffer = b
    old = sys.stdin, sys.stdout, sys.stderr
    try:
        sys.stdin = Pipe(stdin)
        sys.stdout = Pipe(out)
        sys.stderr = err
        server.main()
    finally:
        sys.stdin, sys.stdout, sys.stderr = old
        del server.Session.cmd_boom
    buf = out.getvalue()
    frames, at = [], 0
    while at < len(buf):
        hl, pl = struct.unpack("<II", buf[at:at + 8])
        frames.append(json.loads(buf[at + 8:at + 8 + hl].decode("utf-8")))
        at += 8 + hl + pl
    reply = [f for f in frames if f.get("id") == 7][0]
    assert not reply["ok"] and reply["kind"] == "internal", reply
    assert "Internal error" in reply["error"] and "IndexError" not in reply["error"], reply
    assert "IndexError" in err.getvalue(), "the details did not reach the log"
    print("  internal    %r, details in the log" % reply["error"])


def failed_edit_reply(session, cmd, payload=b"", **params):
    """Exercise the command and the next request through the framed dispatcher."""
    import server

    stdin, stdout = BytesIO(), BytesIO()
    server.write_frame(stdin, {"cmd": cmd, "id": 1, **params}, payload)
    server.write_frame(stdin, {"cmd": "stats", "id": 2})
    stdin.seek(0)
    write_frame = server.write_frame

    def fail_success_frame(stream, header, data=b""):
        if header.get("id") == 1 and header.get("ok"):
            raise MemoryError("could not write reply")
        return write_frame(stream, header, data)

    with (
        patch.object(server, "Session", return_value=session),
        patch.object(server, "write_frame", side_effect=fail_success_frame),
        patch.multiple(sys, stdin=SimpleNamespace(buffer=stdin),
                       stdout=SimpleNamespace(buffer=stdout), stderr=StringIO()),
    ):
        server.main()
    stdout.seek(0)
    assert server.read_frame(stdout)[0]["event"] == "ready"
    failed, _ = server.read_frame(stdout)
    assert not failed["ok"] and failed["detail"] == "MemoryError", failed
    stats, _ = server.read_frame(stdout)
    assert stats["ok"], stats
    return stats


def check_transform_reply_failure(src):
    """Rolling back a failed reply must work even when undo cannot allocate."""
    import server
    from splatcore.document import Document

    session = server.Session()
    doc = session.doc = Document(src)
    doc.transform([4], (1, 0, 0, 0), (1, 0, 0), 1)
    doc.transform([6], (1, 0, 0, 0), (0, 1, 0), 1)
    doc.undo()
    before = doc.live_splats()
    before_mask = doc.alive.copy()
    kept = doc.redo_stack[-1]
    with (
        patch.object(server, "_pack_added", side_effect=MemoryError("could not pack reply")),
        patch.object(doc, "undo", side_effect=MemoryError("could not allocate undo")),
    ):
        stats = failed_edit_reply(session, "transform", idx_payload([2, 3]))
    assert stats["canUndo"] and stats["canRedo"], stats
    assert len(doc.undo_stack) == 1 and doc.redo_stack[-1] is kept
    np.testing.assert_array_equal(doc.alive, before_mask)
    np.testing.assert_array_equal(doc.live_splats().xyz, before.xyz)
    assert doc.redo() is kept and doc.undo() is kept
    print("  rollback    failed transform reply preserved geometry and redo")


def check_history_reply_failure(src):
    import server
    from splatcore.document import Document

    session = server.Session()
    doc = session.doc = Document(src)
    edit = doc.transform([4], (1, 0, 0, 0), (1, 0, 0), 1)
    for command in ("undo", "redo"):
        before = doc.alive, doc.generated, edit.tail
        undo_depth, redo_depth = len(doc.undo_stack), len(doc.redo_stack)
        stats = failed_edit_reply(session, command)
        assert stats["canUndo"] == bool(undo_depth) and stats["canRedo"] == bool(redo_depth)
        assert doc.alive is before[0] and doc.generated is before[1] and edit.tail is before[2]
        assert len(doc.undo_stack) == undo_depth and len(doc.redo_stack) == redo_depth
        assert getattr(doc, command)() is edit
    print("  rollback    failed undo and redo replies preserved both history stacks")


def check_refine_reply_failure(src):
    import server
    from dataclasses import replace
    from splatcore import viewfill
    from splatcore.document import Document
    from splatcore.repair import RepairParams

    session = server.Session()
    doc = session.doc = Document(src)
    doc.fill_engine = "copy"
    doc.backing_layer = False
    indices = np.flatnonzero((src.xyz[:, :2] ** 2).sum(1) < 1.4 ** 2)
    edit = doc.repair(indices, RepairParams(mode="rim"), objects=False)
    state, generated, info = doc._refine, doc.generated, edit.info
    quick = state["res"]
    recolored = quick.added.copy()
    recolored.f_dc += 0.125
    result = replace(quick, added=recolored, info={"deep": True})
    with patch.object(viewfill, "view_fill", return_value=result):
        failed_edit_reply(session, "refine", judge=False)
        assert doc._refine is state and doc.generated is generated and edit.info is info
        assert doc.refine_last(judge=False) is edit
    assert edit.info["refined"]
    np.testing.assert_array_equal(doc.generated.f_dc, recolored.f_dc)
    print("  rollback    failed refine reply preserved geometry and allowed retry")


def main():
    tmp = tempfile.mkdtemp()
    ply = os.path.join(tmp, "plane.ply")
    src = make_ply(ply)
    inprocess_checks(src)
    check_transform_reply_failure(src)
    check_history_reply_failure(src)
    check_refine_reply_failure(src)
    c = Client()
    try:
        h, _ = c.call("ping")
        assert h["ok"] and h["protocol"] == 1, h
        print("  ping        protocol %d, python %s" % (h["protocol"], h["python"]))

        h, _ = c.call("open", path=ply)
        assert h["ok"] and h["n"] == src.n, h
        print("  open        %d splats, SH deg %d, %.2fs" % (h["n"], h["shDegree"], h["loadSeconds"]))

        # select a disc and repair it
        d = np.hypot(src.xyz[:, 0], src.xyz[:, 1])
        sel = np.where(d < 1.2)[0]
        h, payload = c.call("repair", idx_payload(sel), params={"mode": "rim", "seed": 3})
        assert h["ok"], h
        lay = h["layout"]
        assert len(payload) == lay["count"] * lay["stride"] * 4, "payload size mismatch"
        block = np.frombuffer(payload, "<f4").reshape(lay["count"], lay["stride"])
        fields = {f[0]: (f[1], f[2]) for f in lay["fields"]}
        o, n_ = fields["xyz"]
        z = block[:, o + 2]
        assert np.abs(z).max() < 0.2, "new splats off the plane: %.3f" % np.abs(z).max()
        assert h["addedFirstIndex"] == src.n, h["addedFirstIndex"]
        print("  repair      -%d +%d in %.2fs, payload %.1f MB, mode %s"
              % (len(h["removed"]), h["addedCount"], h["seconds"],
                 len(payload) / 1e6, h["info"]["mode"]))

        # The block's f_rest columns must line up with PLY property order --
        # i.e. channel-major. Check it against what the engine itself writes,
        # because getting this wrong is invisible in the numbers and loud on
        # screen: whole-patch colour corruption that looks like a bad fill.
        assert lay["shOrder"] == "channel-major", lay
        probe = os.path.join(tmp, "probe.ply")
        c.call("save", path=probe)
        saved = load_ply(probe)
        o, n_ = fields["f_rest"]
        blk_rest = block[:, o:o + n_]
        tail = saved.take(np.arange(saved.n - lay["count"], saved.n))
        flat = tail.f_rest.transpose(0, 2, 1).reshape(lay["count"], -1)
        assert np.allclose(flat, blk_rest, atol=1e-6), (
            "f_rest in the wire block does not match PLY property order")
        oc, _ = fields["f_dc"]
        assert np.allclose(tail.f_dc, block[:, oc:oc + 3], atol=1e-6)
        orr, _ = fields["rot"]
        assert np.allclose(tail.rot, block[:, orr:orr + 4], atol=1e-6)
        print("  layout      f_rest/f_dc/rot columns match PLY property order")

        h2, _ = c.call("stats")
        assert h2["generated"] == h["addedCount"] and h2["canUndo"]
        print("  stats       base %d, generated %d, alive %d"
              % (h2["base"], h2["generated"], h2["alive"]))

        h3, _ = c.call("undo")
        assert h3["ok"] and h3["applied"] and h3["droppedTail"] == h["addedCount"]
        h4, _ = c.call("stats")
        assert h4["generated"] == 0 and h4["alive"] == src.n, h4
        print("  undo        back to %d alive, generated %d" % (h4["alive"], h4["generated"]))

        # A repair is stochastic, so redo must not re-run it -- it must hand
        # back the exact splats undo took away, down to the last byte.
        h5, pay5 = c.call("redo")
        assert h5["ok"] and h5["applied"], h5
        assert h5["addedCount"] == h["addedCount"], (h5, h["addedCount"])
        assert pay5 == payload, "redo produced different geometry than undo took"
        h5b, _ = c.call("stats")
        assert h5b["generated"] == h["addedCount"] and h5b["alive"] == h2["alive"]
        print("  redo        replays the repair exactly (%d splats, byte-identical)"
              % h5["addedCount"])
        c.call("undo")

        # delete + undo + redo round trip
        h6, _ = c.call("delete", idx_payload(sel))
        assert h6["ok"] and h6["alive"] == src.n - len(sel), h6
        h7, _ = c.call("undo"); assert h7["applied"]
        h8, _ = c.call("redo"); assert h8["applied"], h8
        h9, _ = c.call("stats"); assert h9["alive"] == src.n - len(sel), h9
        print("  delete      -%d, undo/redo round trip exact" % len(sel))

        # save and reload
        out = os.path.join(tmp, "out.ply")
        h10, _ = c.call("save", path=out)
        assert h10["ok"] and h10["written"] == src.n - len(sel), h10
        back = load_ply(out)
        assert back.n == h10["written"] and back.sh_degree == 3
        print("  save        %d splats, %d MB, reloads clean" % (back.n, h10["bytes"] >> 20))

        # Export to the delivery formats, then open each one back through the
        # sidecar. This is the whole round trip the UI drives, including the
        # PLY mirror the viewport needs for a format PlayCanvas cannot parse.
        # Every export comes from the same document, so all the writes happen
        # before any of the opens -- opening one replaces the document, and a
        # loop that saved and opened in turn would quietly export the
        # previous format's losses into the next file.
        written = {}
        for ext in ("spz", "splat", "ply"):
            p = os.path.join(tmp, "out." + ext)
            he, _ = c.call("save", path=p)
            assert he["ok"] and he["format"] == ext, he
            written[ext] = (p, he)

        for ext, want_mirror in (("spz", True), ("splat", True), ("ply", False)):
            p, he = written[ext]
            ho, _ = c.call("open", path=p)
            assert ho["ok"] and ho["format"] == ext, ho
            assert ho["n"] == h10["written"], (ext, ho["n"], h10["written"])
            mirrored = ho["viewPath"] != os.path.abspath(p)
            assert mirrored == want_mirror, (ext, ho["viewPath"])
            assert ho["viewPath"].lower().endswith(".ply"), ho["viewPath"]
            assert os.path.exists(ho["viewPath"]), ho["viewPath"]
            assert load_ply(ho["viewPath"]).n == ho["n"], "mirror has a different count"
            print("  %-10s  %d splats, %.1f MB, SH %d, displayed via %s"
                  % (ext, ho["n"], he["bytes"] / 1048576, ho["shDegree"],
                     "a PLY mirror" if mirrored else "the file itself"))
        assert load_ply(written["ply"][0]).sh_degree == 3, "PLY must keep its SH"

        # The SH cap has to reach every format, not just the ones that quantise.
        c.call("open", path=ply)
        c.call("delete", idx_payload(sel))
        small = os.path.join(tmp, "capped.ply")
        hc1, _ = c.call("save", path=small, options={"sh_degree": 1})
        assert hc1["ok"] and hc1["shBandsCapped"] == 12, hc1
        assert load_ply(small).sh_degree == 1
        assert hc1["bytes"] < h10["bytes"] / 2, (hc1["bytes"], h10["bytes"])
        print("  sh cap      degree 3 -> 1: %d MB -> %d MB, %d bands dropped"
              % (h10["bytes"] >> 20, hc1["bytes"] >> 20, hc1["shBandsCapped"]))

        # and put the real document back, so later checks see what they expect
        c.call("open", path=ply)
        c.call("delete", idx_payload(sel))

        # errors are reported, not fatal
        h11, _ = c.call("repair", idx_payload([]), params={"mode": "rim"})
        assert not h11["ok"] and "empty" in h11["error"].lower(), h11
        h12, _ = c.call("bogus")
        assert not h12["ok"] and "unknown" in h12["error"], h12
        h13, _ = c.call("ping")
        assert h13["ok"], "sidecar did not survive errors"
        print("  errors      reported and survivable: %r" % h11["error"])

        # deep repair, when this machine can do it at all
        hc, _ = c.call("capabilities", wait=True)
        assert hc["ok"], hc
        print("  caps        gpu=%s inpainter=%s" % (hc["gpu"], hc["inpainter"]))
        if hc["deep"]:
            c.call("undo")                        # back to a clean document
            c.call("undo")
            hd, pd = c.call("deep_repair", idx_payload(sel), fill="patch",
                            params={"mode": "rim", "seed": 3},
                            deep={"steps": 20, "views": 2, "resolution": 192})
            assert hd["ok"], hd
            assert c.progress, "no progress events during a long operation"
            assert hd["addedCount"] > 0, hd
            info = hd["info"]
            assert "errBefore" in info and "errAfter" in info, info
            lay = hd["layout"]
            assert len(pd) == lay["count"] * lay["stride"] * 4
            print("  deep        %s, %s, err %.3f -> %.3f, deep=%s, %d messages"
                  % (info["inpainter"], info.get("backend"), info["errBefore"],
                     info["errAfter"], info["deep"], len(c.progress)))

            # Every .splat file, and every PLY saved below SH degree 3, used to
            # crash the GPU pass: the renderer was always told degree 3.
            c.call("open", path=ply)
            flat = os.path.join(tmp, "sh0.ply")
            c.call("save", path=flat, options={"sh_degree": 0})
            c.call("open", path=flat)
            h0, _ = c.call("deep_repair", idx_payload(sel), fill="patch",
                           params={"mode": "rim", "seed": 3},
                           deep={"steps": 20, "resolution": 192})
            assert h0["ok"], h0
            print("  deep sh0    %s, deep=%s" % (h0["info"].get("backend"),
                                                h0["info"]["deep"]))

            # The wide view filled by LaMa, projected back as a reference.
            # Bad names are refused, and Photoshop is only ever reported,
            # never started, by a test.
            assert "photoshop" in hc, hc
            c.call("open", path=ply)
            # The wide view needs the GPU renderer as well as LaMa; "deep"
            # alone is also true with OpenCV on a machine without CUDA.
            if hc["viewFill"]:
                hv, pv = c.call("deep_repair", idx_payload(sel), fill="lama",
                                params={"mode": "rim", "seed": 3},
                                deep={"steps": 20})
                assert hv["ok"], hv
                assert hv["info"].get("fill_source") == "LaMa", hv["info"]
                assert hv["addedCount"] > 0, hv
                print("  deep view   %s, deep=%s, Photoshop=%s"
                      % (hv["info"]["fill_source"], hv["info"]["deep"],
                         hc["photoshop"]))
            else:
                print("  deep view   skipped: %s" % hc["viewFillDetail"])
            hb, _ = c.call("deep_repair", idx_payload(sel), fill="gimp",
                           params={"mode": "rim", "seed": 3})
            assert not hb["ok"] and "fill" in hb["error"], hb
        else:
            print("  deep        skipped: %s" % (hc["inpainterDetail"]
                                                      or hc["gpuDetail"]))
        print("all server tests passed")
    finally:
        c.close()


if __name__ == "__main__":
    main()
