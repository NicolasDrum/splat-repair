"""Transforms must move all four things a Gaussian is made of."""
import os
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splatcore.io import SplatSet                         # noqa: E402
from splatcore import transform, shtools                  # noqa: E402
from splatcore.document import Document                   # noqa: E402


def make(n=400, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return SplatSet(
        xyz=rng.normal(size=(n, 3)).astype(np.float32),
        f_dc=rng.normal(size=(n, 3)).astype(np.float32),
        f_rest=rng.normal(size=(n, 15, 3)).astype(np.float32),
        opacity=rng.normal(size=n).astype(np.float32),
        scale=rng.uniform(-3, -1, (n, 3)).astype(np.float32),
        rot=q.astype(np.float32))


def rand_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_translation_moves_only_positions():
    s = make()
    before = s.copy()
    idx = np.arange(0, 200)
    transform.apply(s, idx, translation=(1.0, -2.0, 0.5))
    assert np.allclose(s.xyz[idx] - before.xyz[idx], [1.0, -2.0, 0.5], atol=1e-5)
    assert np.array_equal(s.rot[idx], before.rot[idx])
    assert np.array_equal(s.f_rest[idx], before.f_rest[idx])
    # untouched splats stay untouched
    assert np.array_equal(s.xyz[200:], before.xyz[200:])
    print("    translation moves positions and nothing else")


def test_rotation_turns_geometry_and_sh():
    rng = np.random.default_rng(3)
    s = make()
    before = s.copy()
    idx = np.arange(s.n)
    q = rand_quat(rng)
    pivot = np.array([0.5, -0.25, 1.0])
    info = transform.apply(s, idx, rotation=q, pivot=pivot)
    R = transform.quat_to_matrix(q)

    want = pivot + (before.xyz[idx] - pivot) @ R.T
    assert np.allclose(s.xyz[idx], want, atol=1e-4), "positions"

    # the radiance field must have turned with the geometry
    want_sh = shtools.rotate_sh(before.f_rest[idx].astype(np.float64), R)
    assert np.allclose(s.f_rest[idx], want_sh, atol=1e-4), "SH bands"
    assert info["shRotated"]

    # orientations too: the covariance has to follow
    want_q = shtools.rotate_quats(before.quat[idx], R)
    got = s.quat[idx] * np.sign(s.quat[idx][:, :1])
    want_q = want_q * np.sign(want_q[:, :1])
    assert np.allclose(got, want_q, atol=1e-4), "quaternions"
    print("    rotation turns positions, covariances and SH together")


def test_scale_grows_the_splats_too():
    s = make()
    before = s.copy()
    idx = np.arange(s.n)
    transform.apply(s, idx, scale=2.0, pivot=(0, 0, 0))
    assert np.allclose(s.xyz, before.xyz * 2.0, atol=1e-4)
    # log-space scale, so doubling is an additive log 2
    assert np.allclose(s.sigma, before.sigma * 2.0, rtol=1e-4), \
        "splats must grow with the geometry, or a scaled patch turns to dots"
    print("    uniform scale grows positions and splat sizes alike")


def test_round_trip_returns_to_start():
    rng = np.random.default_rng(11)
    s = make()
    before = s.copy()
    idx = np.arange(s.n)
    q = rand_quat(rng)
    t = np.array([0.7, -1.3, 2.0])
    pivot = s.xyz[idx].mean(0).astype(np.float64)
    transform.apply(s, idx, rotation=q, translation=t, scale=1.5, pivot=pivot)
    inv = transform.invert(q, t, 1.5, pivot)
    transform.apply(s, idx, rotation=inv["rotation"], translation=inv["translation"],
                    scale=inv["scale"], pivot=inv["pivot"])
    assert np.allclose(s.xyz, before.xyz, atol=2e-3), \
        "max drift %.4f" % np.abs(s.xyz - before.xyz).max()
    assert np.allclose(s.sigma, before.sigma, rtol=1e-3)
    print("    forward then inverse returns to the start")


def test_failed_transform_keeps_geometry_and_redo():
    doc = Document(make(32))
    rotation = (1, 0, 0, 0)
    doc.transform([4, 5], rotation, (1, 0, 0), 1)
    doc.transform([6], rotation, (0, 1, 0), 1)
    doc.undo()
    before = doc.live_splats()
    alive, generated = doc.alive, doc.generated
    expected_undo, expected_redo = doc.undo_stack[-1], doc.redo_stack[-1]
    concat = np.concatenate

    def fail_mask_allocation(parts, *args, **kwargs):
        if parts[0] is alive:
            raise MemoryError("simulated mask allocation failure")
        return concat(parts, *args, **kwargs)

    with patch.object(np, "concatenate", side_effect=fail_mask_allocation):
        with np.testing.assert_raises(MemoryError):
            doc.transform([0, 1], rotation, (0, 0, 1), 1)

    assert doc.alive is alive and doc.generated is generated
    assert len(doc.undo_stack) == len(doc.redo_stack) == 1
    assert doc.undo_stack[-1] is expected_undo and doc.redo_stack[-1] is expected_redo
    np.testing.assert_array_equal(doc.live_splats().xyz, before.xyz)
    assert doc.redo() is expected_redo
    assert doc.undo() is expected_redo
    np.testing.assert_array_equal(doc.live_splats().xyz, before.xyz)


def test_rejects_nonsense():
    s = make()
    for kw in ({"scale": 0.0}, {"scale": -1.0}, {"rotation": (0, 0, 0, 0)}):
        try:
            transform.apply(s, np.arange(10), **kw)
            raise AssertionError("expected ValueError for %s" % kw)
        except ValueError:
            pass
    try:
        transform.apply(s, [])
        raise AssertionError("expected ValueError for an empty selection")
    except ValueError:
        pass
    print("    bad input refused")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print("  " + name)
            try:
                fn()
            except AssertionError as e:
                fails += 1
                print("    FAIL:", e)
    print("all transform tests passed" if not fails else "%d FAILURES" % fails)
    sys.exit(1 if fails else 0)
