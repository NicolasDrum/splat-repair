/**
 * Splat Repair -- application glue.
 *
 * Division of labour: the viewport owns rendering and selection (both must feel
 * instant), the Python engine owns the document and the repair algorithm (it
 * needs NumPy, and later Torch). They agree on splat indices and nothing else.
 */
import { Viewer, UP_AXES, DEFAULT_UP, UP_VECTOR, type UpAxis } from './engine/viewer';
import { SplatDocument } from './engine/document';
import { project, depthGrid, surfaceGrid, strokeToCapsules, selectInCapsules,
         fillPolygon, selectInRegion, applyMode, maskToIndices,
         SIZE_K, SIZE_CAP,
         type CapsulePoint, type DepthGrid, type ProjectResult,
         type SelectMode, type DepthMode, type SizeReach,
         type StrokeSample } from './engine/selection';
import { segmentRows, fillRegionHoles, strokeHoles } from './engine/selection';
import { EMPTY_VOLUME, SUBTRACT_REACH, withOp, volumeToWire, evaluateVolume, paintedReach,
         type OpSize, type Volume } from './engine/volume';
import { VolumeOutline } from './engine/volumeview';
import { GpuSelector } from './engine/gpuselect';
import { SelectionGizmo, toEngineQuat, type GizmoMode, type GizmoDelta }
  from './engine/gizmo';
import { History, type HistoryEntry } from './ui/history';
import { formatSideError } from './ui/fill-status';
import { Shell } from './ui/shell';
import { ClonePreview } from './ui/clonepreview';
import { robustBounds } from './engine/navpick';

declare global {
  interface Window {
    host: {
      engine: {
        call(cmd: string, args?: Record<string, any>, payload?: ArrayBuffer):
          Promise<{ header: Record<string, any>; payload: ArrayBuffer | null }>;
        onLog(fn: (s: string) => void): void;
        onExit(fn: (code: number) => void): void;
        onEvent(fn: (e: Record<string, any>) => void): void;
      };
      openSplatDialog(): Promise<string | null>;
      saveSplatDialog(s: string): Promise<string | null>;
      confirmDiscard(): Promise<boolean>;
      onUnexportedQuery(fn: () => boolean): void;
      closeWindow(): void;
      pathForFile(f: File): string;
      splatUrl(p: string): string;
    };
  }
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const canvas = $<HTMLCanvasElement>('canvas');
const guides = $<HTMLCanvasElement>('guides');
const ring = $<HTMLDivElement>('brush-ring');
const hint = $<HTMLDivElement>('hint');
const busy = $<HTMLDivElement>('busy');
const busyText = $<HTMLSpanElement>('busy-text');
const busyFill = $<HTMLElement>('busy-fill');

const viewer = new Viewer();
const doc = new SplatDocument();

let filePath: string | null = null;
let sourceFormat = 'ply';          // what the open file was, for the export default
type Tool = 'orbit' | 'brush' | 'lasso' | 'box' | 'clone' | 'xform';
let tool: Tool = 'orbit';
/**
 * The selection is a mask, not a list.
 *
 * Adding to and subtracting from a selection is a set operation, and doing that
 * on a sorted index list means merging on every stroke. One byte per splat is
 * 550 kB and makes every modifier a single write.
 */
let selMask = new Uint8Array(0);
let selCount = 0;
let selection: Uint32Array<ArrayBufferLike> = new Uint32Array(0);
let selDirty = false;
let selectMode: SelectMode = 'replace';
let selDepth: DepthMode = 'surface';
let cropActive = false;
let cloneSource: [number, number, number] | null = null;
// A painted stamp stroke: the offset is locked when the drag starts and the
// source travels with the brush, the way a clone stamp has worked since
// Photoshop 3. Dabs are collected here and sent as one edit on release.
let cloneOffset: [number, number, number] | null = null;
/** Within a stroke: the offset of the last dab that found its source. */
let cloneCarry: [number, number, number] | null = null;
/** Where the user Ctrl+clicked, and the offset an aligned stamp keeps. */
let cloneOrigin: [number, number, number] | null = null;
let cloneAlignedOffset: [number, number, number] | null = null;
/**
 * The camera direction the selection was made from.
 *
 * Every brush that edits a surface sends a view: which side is the front,
 * and for "Solidify" the view in which the patch showed through. It used
 * to be read when the button was pressed -- after the user had orbited
 * round to look at the selection, or from below the sole after painting
 * from the side -- so the engine was told the wrong side. Captured when a
 * stroke, lasso or box is committed; null for selections without a view
 * (debris, invert), which fall back to the current camera.
 */
let selView: [number, number, number] | null = null;
/** The view a stamp stroke was started from. */
let cloneView: [number, number, number] | null = null;
/** And where the camera stood: each dab's points go onto the surface along their rays. */
let cloneEye: [number, number, number] | null = null;
/**
 * The last brush stroke in world space, the selection before it, and the
 * selection as it was at pointerdown -- so the depth slider can redo exactly
 * that stroke at another depth, from the view it was painted in, on top of
 * what was selected before it.
 */
let strokeCaps: CapsulePoint[] = [];
let strokeBase: Uint8Array | null = null;
let strokeStart: Uint8Array | null = null;
/**
 * The last fill, so the same stroke can be filled again differently. `step`
 * is its entry in the history: "Retry" undoes the step on top, so it is
 * only offered while that step is this fill.
 */
let lastFill: { idx: ArrayBuffer; deep: boolean; seed: number; serial: number;
                view: [number, number, number]; step: HistoryEntry | null;
                /** Whether the fill asked for the object grow (objectGrow). */
                objects: boolean } | null = null;
let editSerial = 0;
/** Geometry revision captured by the last successful export for this file. */
let exportedRevision = 0;
/** The engine's last report, for the test harness to read. */
let lastInfo: any = null;
/** What the engine said about the last fill (views taken, seconds each): for the autotest's log. */
let lastFillInfo: any = null;
let cloneDabs: Array<[[number, number, number], [number, number, number]]> = [];
let cloneLast: [number, number, number] | null = null;
let cloneRadii: number[] = [];
let clonePainting = false;
/** The stroke's source offset in canvas pixels, and which dabs found a surface there. */
let cloneScreenOff: { x: number; y: number } | null = null;
let clonePicked: boolean[] = [];
/** The stroke that fixes the aligned offset, once it has stamped. */
let cloneStartedAligned = false;
/** An Alt + press in the stamp: a click sets the source, a drag orbits. */
let cloneAltDown: { x: number; y: number } | null = null;
/** The stamp's preview (ui/clonepreview.ts) and where the brush last hovered. */
let clonePreview: ClonePreview | null = null;
let cloneCursor: { x: number; y: number } | null = null;
let gizmo: SelectionGizmo | null = null;
let gizmoMode: GizmoMode = 'translate';
let lassoPts: { x: number; y: number }[] = [];
let boxStart: { x: number; y: number } | null = null;
let boxNow: { x: number; y: number } | null = null;
let proj: ProjectResult | null = null;
let grid: DepthGrid | null = null;
let projDirty = true;
let stroke: StrokeSample[] = [];
let painting = false;
let depthMode: 'quick' | 'deep' = 'quick';
let gpuSel: GpuSelector | null = null;
let lastCaps: CapsulePoint[] = [];
let caps: Record<string, any> = { deep: false };
/**
 * How far a splat's own size widens the brush's depth window: k sigmas along
 * the view ray, at most `cap` brush radii. One value for the CPU and the GPU
 * selector, so they cannot disagree. k = 0 is the old point test.
 */
let sizeReach = { k: SIZE_K, cap: SIZE_CAP };
/** Lasso and box: the same k, capped at this many depth windows. */
const REGION_SIZE_CAP = 2;

/* ------------------------------------------------------- painted volume */

/**
 * What the selection is in space (engine/volume.ts): every stroke, lasso and
 * box as world geometry, in order. `selMask` stays the per-splat answer; the
 * volume goes to the engine with every command that uses the selection, so
 * it can mask and fill where there are no splats. null: the selection was
 * not painted (debris, invert, a transform's result).
 */
let selVolume: Volume | null = EMPTY_VOLUME;
/** The volume before the live stroke, so the depth slider can redo it. */
let strokeVolumeBase: Volume | null = EMPTY_VOLUME;
/** The volume at pointerdown, next to `strokeStart`. */
let strokeStartVolume: Volume | null = null;
/** The volume the last fill was made with, for "Retry". */
let lastFillVolume: Record<string, any> | null = null;
let volumeOutline: VolumeOutline | null = null;
/** Close holes inside a lasso or box (off only to measure what it changes). */
let regionHoleFill = true;
/** The view-projection and viewport the cached projection was made with. */
let projMeta: { vp: Float32Array; w: number; h: number } | null = null;

function setVolume(v: Volume | null) {
  selVolume = v;
  // mid-stroke (no GPU preview: the exact path runs every frame) the
  // outline takes it when its turn comes; the stroke's end sets it for good
  if (painting) volumeOutline?.preview(v);
  else volumeOutline?.set(v);
}

/** The volume as the engine takes it (header key `volume`), or null. */
function volumeArg(): Record<string, any> | null {
  return volumeToWire(selVolume);
}

/**
 * The selection as an undo step: the mask and the volume.
 *
 * Volumes are immutable and shared between snapshots, so a snapshot of a
 * painted selection is a handful of references. The per-splat mask is kept
 * only where the volume cannot stand for it -- a selection that was not
 * painted, or the size-aware reach being on -- and then as sorted indices,
 * not one byte per splat.
 */
export interface SelectionSnapshot {
  n: number;
  count: number;
  volume: Volume | null;
  view: [number, number, number] | null;
  indices: Uint32Array | null;
}

function snapshotSelection(): SelectionSnapshot {
  // The index list, always, and the very array the selection caches: it is
  // replaced on every change, never written into, so steps share it. A
  // painted selection used to be kept as its volume alone and worked out
  // again on undo, against the splats as they are *then* -- after a crop
  // preview was switched off, a refine swapped the fill, or an in-place edit
  // moved the selection to new indices, that gave another selection than the
  // one the step had.
  return {
    n: doc.nTotal, count: selCount, volume: selVolume, view: selView,
    indices: selectionIndices() as Uint32Array
  };
}

function restoreSelection(s: SelectionSnapshot | null) {
  stroke = [];
  forgetStroke();
  strokeStart = null;
  lassoPts = [];
  boxStart = boxNow = null;
  if (!s) { clearSelection(); return; }
  let mask: Uint8Array<ArrayBuffer>;
  let exact: Uint32Array | null = null;
  if (s.indices) {
    mask = new Uint8Array(doc.nTotal);
    let inside = true;
    for (let k = 0; k < s.indices.length; k++) {
      if (s.indices[k] < mask.length) mask[s.indices[k]] = 1; else inside = false;
    }
    if (inside) exact = s.indices;
  } else if (s.volume && s.volume.ops.length) {
    mask = evaluateVolume(s.volume, doc.centres, doc.nTotal, visibleMask(), opSize());
  } else {
    mask = new Uint8Array(doc.nTotal);
  }
  selMask = mask;
  selCount = 0;
  for (let i = 0; i < mask.length; i++) selCount += mask[i];
  // the step's own list stands for the mask: no new array, so the step
  // after this one shares it too
  if (exact && exact.length === selCount) { selection = exact; selDirty = false; }
  else selDirty = true;
  selView = s.view;
  setVolume(s.volume);
  afterSelectionChanged();
}

function brushSize(): SizeReach {
  return { covA: doc.covA, covB: doc.covB, k: sizeReach.k, cap: sizeReach.cap };
}

/** The reach brush and lasso/box select with, to evaluate a volume the same way. */
function opSize(): OpSize | undefined {
  if (sizeReach.k <= 0) return undefined;
  return { brush: brushSize(), region: { ...brushSize(), cap: REGION_SIZE_CAP } };
}

/* ----------------------------------------------------------------- history */

/**
 * One history for selections and edits, as in Photoshop (ui/history.ts).
 * Selection changes are steps; an edit step restores the selection it was
 * made from when it is undone. The selection part is the snapshot above:
 * mask and painted volume together. Hooks: afterSelectionChanged and
 * clearSelection report changes, setHistory reports edits, setBusy says
 * when an engine command starts and ends.
 */
const history = new History({
  engineUndo: async () => { await doUndo(); return null; },
  engineRedo: async () => { await doRedo(); return null; },
  busy: () => engineBusy,
  // Ctrl+Z during a command did nothing and said nothing
  refused: () => status('still working, try again in a moment')
});
history.addPart<SelectionSnapshot>('selection', {
  snapshot: snapshotSelection,
  restore: (s) => restoreSelection(s),
  equal: sameSelection,
  bytes: (s) => (s.indices ? s.indices.byteLength : 0) + 256
});
// "Retry" follows the history: a step on top of the fill switches it off,
// undoing that step switches it on again
history.onChange(() => refreshReroll());

/**
 * The original file is already on disk. An edited revision is safe to
 * discard once an export of that revision succeeds.
 */
function hasUnexportedEdits(): boolean {
  const revision = history.revision;
  return revision !== 0 && revision !== exportedRevision;
}
window.host.onUnexportedQuery(hasUnexportedEdits);
/** Menus, view cube, options bar; created at boot (ui/shell.ts). */
let shell: Shell | null = null;

function sameSelection(a: SelectionSnapshot, b: SelectionSnapshot): boolean {
  if (a.count !== b.count) return false;
  if (a.count === 0) return true;
  if (a.volume !== b.volume) return false;
  if (!a.indices || !b.indices) return a.indices === b.indices;
  if (a.indices === b.indices) return true;
  if (a.indices.length !== b.indices.length) return false;
  for (let i = 0; i < a.indices.length; i++) if (a.indices[i] !== b.indices[i]) return false;
  return true;
}

const SEL_LABEL: Record<string, string> = {
  brush: 'Brush', lasso: 'Lasso', box: 'Rectangle', clone: 'Clone Stamp', xform: 'Move', orbit: 'Selection'
};

/** What the history panel calls a selection change made with the active tool. */
function selectionLabel(): string {
  if (!selCount) return 'Deselect';
  let base = SEL_LABEL[tool] ?? 'Selection';
  if ((tool === 'lasso' || tool === 'box') && selDepth === 'through') base += ' through';
  return selectMode === 'add' ? `${base} add` : selectMode === 'subtract' ? `${base} subtract` : base;
}

/** Ctrl+A: every splat on screen. */
function selectAll() {
  if (!doc.nTotal) return;
  history.labelNext('Select All');
  selectMode = 'replace';
  forgetStroke();
  selView = null;
  setVolume(null);
  commitHits(maskToIndices(visibleMask()));
}

/** F: the selection into view; everything when nothing is selected. */
function frameSelection() {
  if (!doc.nTotal) return;
  if (!selCount) { viewer.frameAll(); return; }
  const b = robustBounds(doc.centres, visibleMask(), selMask, 0.01, 0.99);
  if (b) viewer.frameSphere(b.centre, b.radius);
}

/* ------------------------------------------------------------------ status */

function status(msg: string, kind: 'ok' | 'err' | '' = '') {
  const el = $<HTMLSpanElement>('status-msg');
  el.textContent = msg;
  el.className = 'rec' + (kind ? ' ' + kind : '');
}

function counts() {
  const el = $<HTMLSpanElement>('status-counts');
  if (!doc.nTotal) { el.textContent = '–'; return; }
  const parts = [`${fmt(doc.nAlive)} splats`];
  if (doc.nGenerated) parts.push(`${fmt(doc.nGenerated)} new`);
  if (selCount) parts.push(`${fmt(selCount)} selected`);
  el.textContent = parts.join('  /  ');
}

const fmt = (n: number) => n.toLocaleString('en-US');
const num = (n: number, d = 2) => n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

/**
 * Is a command in flight?
 *
 * The overlay alone was decoration: nothing read it, so a second Ctrl+Z
 * during the first one queued a second undo behind it. The engine applies
 * both, correctly and invisibly, and the document ends up two steps back
 * from where the user expected.
 */
let engineBusy = false;

function setBusy(on: boolean, text = 'Working …', frac = 0) {
  if (on && !engineBusy) history.noteBusy(text);     // names the step it may become
  engineBusy = on;
  busyText.textContent = text;
  busyFill.style.setProperty('width', `${Math.round(frac * 100)}%`);
  busy.hidden = !on;
  if (!on) history.flush();     // a selection made by a command (floaters) becomes a step
}

/* ------------------------------------------------------------------ engine */

/**
 * The engine has ended after it had been running (see onExit at the bottom):
 * the sentence that says so. It held the document and is not started a
 * second time, so every command from then on fails with this, whatever the
 * call itself was told. (`engineUp`: it has answered at least once. One that
 * never came up is another matter, and its first command is told why.)
 */
let engineGone: string | null = null;
let engineUp = false;

async function engine(cmd: string, args: Record<string, any> = {}, payload?: ArrayBuffer) {
  const r = await window.host.engine.call(cmd, args, payload);
  if (!r.header.ok) {
    const e = new Error(engineGone ?? r.header.error ?? `${cmd} failed`);
    (e as any).kind = r.header.kind;        // 'missing': the file is gone (openFile)
    throw e;
  }
  engineUp = true;
  return r;
}

/* -------------------------------------------------------------------- open */

/**
 * Back to "nothing open". For an open that failed after the engine had
 * already taken the new file: the viewport then still showed the old one,
 * and every edit went to a document the user was not looking at.
 */
function closeDocument() {
  viewer.clear();
  // an empty document: the same arrays a load builds, with no splats in them
  doc.load({ numSplats: 0, elements: [{ name: 'vertex', properties: [] }] });
  filePath = null;
  lastGeneratedCount = 0;
  lastFill = null;
  lastFillVolume = null;
  exportedRevision = 0;
  gpuSel?.destroy();
  gpuSel = null;
  cloneSource = cloneOrigin = cloneAlignedOffset = null;
  cloneView = null;
  cloneOffset = cloneScreenOff = cloneCarry = null;
  clonePreview?.cancel();
  $('clone-state').textContent = 'Alt + click sets the source.';
  forgetFileNotes();
  history.clear();
  resetCrop();
  clearSelection();
  setHistory({ canUndo: false, canRedo: false });
  $('file-name').textContent = 'No file open';
  hint.hidden = false;
  for (const id of ['btn-export', 'btn-floaters', 'btn-invert', 'btn-up']) {
    $<HTMLButtonElement>(id).disabled = true;
  }
  projDirty = true;
  counts();
}

/**
 * What the panel says about the file before this one: how much debris the
 * last search found, and what the shadow brush last did ("Ctrl+Z undoes
 * it"). Both stayed over the next file, where neither was true.
 */
function forgetFileNotes() {
  $('floater-count').textContent = '–';
  shadowNote(null);
}

/** The "unexported edits" question is open (openFile). */
let openAsking = false;

async function openFile(path: string) {
  if (!viewer.app) {
    status('Nothing can be opened without a viewport. Graphics did not start at launch.', 'err');
    return;
  }
  // One document at a time, on both sides. A file opened while another load
  // or a command was still in flight left the viewport on one file and the
  // engine on the other.
  if (engineBusy) {
    status('not opened: something is still running, try again when it is done', 'err');
    return;
  }
  if (hasUnexportedEdits()) {
    if (openAsking) return;       // one question at a time
    openAsking = true;
    let discard = false;
    try { discard = await window.host.confirmDiscard(); } finally { openAsking = false; }
    if (!discard) return;
    if (engineBusy) return;       // something started while the question was open
  }
  setBusy(true, 'Loading splat …');
  status('loading …');
  let engineHasIt = false;
  try {
    const t0 = performance.now();
    // The engine goes first because it is the one that identifies the format,
    // and because PlayCanvas parses PLY but not .splat or .spz -- for those
    // the engine hands back a PLY of what it read. Still no geometry over the
    // wire: both sides end up reading a file.
    const open = await engine('open', { path });
    engineHasIt = true;
    sourceFormat = String(open.header.format ?? 'ply');
    const viewPath = String(open.header.viewPath ?? path);
    // What the engine names for the viewport is PLY: the file itself when it
    // read it as one, else the PLY it wrote of it. PlayCanvas picks its parser
    // by the name's ending, and a PLY saved as .splat or .spz found none.
    let viewName = viewPath.split(/[\\/]/).pop() ?? 'splat.ply';
    if ((open.header.viewPath || sourceFormat === 'ply') && !/\.ply$/i.test(viewName)) {
      viewName += '.ply';
    }
    // the old file's textures go with it; the GPU brush must not keep them
    gpuSel?.destroy();
    gpuSel = null;
    const info = await viewer.load(window.host.splatUrl(viewPath), viewName);
    doc.load(viewer.gsplatData);
    filePath = path;
    exportedRevision = 0;
    lastFill = null;
    lastFillVolume = null;
    // Forget the previous file's selection, crop and stamp source first.
    // `applyUpAxis` refreshes the gizmo and the source marker, and it used
    // to do that with the old file's selection mask over the new file's
    // centres -- a NaN pivot -- and the old source still on screen.
    cloneSource = cloneOrigin = cloneAlignedOffset = null;
    cloneView = null;
    cloneOffset = cloneScreenOff = cloneCarry = null;
    clonePreview?.cancel();
    $('clone-state').textContent = 'Alt + click sets the source.';
    forgetFileNotes();
    resetCrop();
    clearSelection();
    // Upright before the first frame is shown: the axis this file was last
    // viewed with, else the 3DGS convention.
    applyUpAxis(savedUpAxis(path), false);
    if (viewer.splat && viewer.editState) {
      // What the GPU brush needs to see a splat's size. Written once: the
      // loaded file never changes on the GPU, only its flags do.
      viewer.writeStream(viewer.splat, 'editCovA',
        new Uint32Array(doc.covA.buffer, doc.covA.byteOffset, doc.nBase * 4));
      viewer.writeStream(viewer.splat, 'editCovB',
        new Uint32Array(doc.covB.buffer, doc.covB.byteOffset, doc.nBase * 2));
    }

    $('file-name').textContent = path;
    hint.hidden = true;
    $<HTMLButtonElement>('btn-export').disabled = false;
    $<HTMLButtonElement>('btn-floaters').disabled = false;
    $<HTMLButtonElement>('btn-invert').disabled = false;
    $<HTMLButtonElement>('btn-up').disabled = false;
    // A fresh document has an empty history; the buttons used to stay lit
    // from the previous file and then do nothing.
    setHistory({ canUndo: false, canRedo: false });
    resetCrop();
    projDirty = true;
    lastGeneratedCount = 0;
    gpuSel = viewer.splat && viewer.editState
      ? new GpuSelector(viewer.device, viewer.splat) : null;
    if (gpuSel && !gpuSel.ready) {
      console.warn('GPU selection unavailable:', gpuSel.reason);
      gpuSel = null;
    }
    clearSelection();
    history.reset(`Opened: ${path.split(/[\\/]/).pop() ?? path}`);
    shell?.opened(path);
    // Splats whose position is NaN or infinite: the engine counts them and
    // leaves them in (`nonFinite`). Nobody sees them, but they are exported
    // again and the search for debris refuses the file while they are there.
    const bad = Number(open.header.nonFinite ?? 0);
    status(`loaded in ${num((performance.now() - t0) / 1000)} s · `
           + `${sourceFormat} · SH ${info.shBands}`
           + (sourceFormat === 'splat' ? ' · .splat carries no SH bands' : '')
           + (bad ? ` · ${fmt(bad)} ${bad === 1 ? 'splat' : 'splats'} without a valid position`
                    + ' · remove them in the Filter menu' : ''),
           bad ? '' : 'ok');
    counts();
  } catch (e: any) {
    const msg = String(e.message ?? e);
    const name = path.split(/[\\/]/).pop() ?? path;
    if (engineHasIt) {
      // The engine holds the new file, the viewport could not show it:
      // nothing is open then, rather than the old file over the new document.
      closeDocument();
      status(`${name} cannot be displayed (${msg}). No file is open now.`, 'err');
    } else if (e?.kind === 'missing' && shell?.forget(path)) {
      status(`${msg} The entry was removed from Open Recent.`, 'err');
    } else {
      status(msg, 'err');
    }
  } finally {
    setBusy(false);
  }
}

/* ------------------------------------------------------------------ up axis */

const UP_KEY = 'splatRepair.up:';
const UP_LABEL: Record<UpAxis, string> = { '-y': '−Y', '+y': '+Y', '+z': '+Z', '-z': '−Z' };

/** The up axis this file was last viewed with, or the 3DGS default. */
function savedUpAxis(path: string): UpAxis {
  try {
    const v = localStorage.getItem(UP_KEY + path) as UpAxis | null;
    if (v && UP_AXES.includes(v)) return v;
  } catch { /* storage unavailable: fall through */ }
  return DEFAULT_UP;
}

/** Turn the view to an up axis; `remember` stores it for this file. */
function applyUpAxis(up: UpAxis, remember = true) {
  viewer.setUpAxis(up);
  projDirty = true;
  $('btn-up').textContent = `Up ${UP_LABEL[up]}`;
  if (remember && filePath) {
    try { localStorage.setItem(UP_KEY + filePath, up); } catch { /* not persistent */ }
  }
  // The crop sliders work in the file's own axes, which the view no longer
  // shows upright once it has been turned. Say which way the up axis runs.
  const ax = up[1].toUpperCase();
  $('crop-hint').textContent = `Sliders run along the file's axes. Up is ${UP_LABEL[up]}: `
    + (up[0] === '-' ? `on the ${ax} slider 0 is top, 1 is bottom.` : `on the ${ax} slider 0 is bottom, 1 is top.`);
  for (const a of ['x', 'y', 'z']) {
    $(`crop-ax-${a}`).textContent = a.toUpperCase() + (a === up[1] ? (up[0] === '-' ? '↓' : '↑') : '');
  }
  refreshGizmo();
  showCloneSource();
}

function cycleUpAxis() {
  if (!doc.nTotal) return;
  const i = UP_AXES.indexOf(viewer.upAxis);
  applyUpAxis(UP_AXES[(i + 1) % UP_AXES.length]);
  status(`up is now ${UP_LABEL[viewer.upAxis]} · remembered for this file`, 'ok');
}

/* --------------------------------------------------------------- selection */

/**
 * Projection and the surface-depth map behind the brush.
 *
 * Both depend only on the camera and which splats are alive, so they survive a
 * whole stroke. Rebuilding them per frame was the last linear-in-splat-count
 * work left on the hot path.
 */
function ensureProjection() {
  if (!doc.nTotal) return null;
  if (!projDirty && proj) return proj;
  const r = canvas.getBoundingClientRect();
  const vp = viewer.viewProjection();
  // kept, so a lasso or box volume records the camera it was really made in
  projMeta = { vp: new Float32Array(vp), w: r.width, h: r.height };
  proj = project(doc.centres, vp, r.width, r.height, proj ?? undefined);
  grid = depthGrid(proj, r.width, r.height,
                   { alive: visibleMask(), alpha: doc.alpha, minAlpha: doc.surfaceAlpha });
  projDirty = false;
  return proj;
}

/**
 * What is on screen: alive, and not hidden by the crop preview.
 *
 * Only the GPU brush preview honoured the crop -- it reads the deleted
 * flags the preview writes. The depth grid, the exact brush, the lasso and
 * box and the stamp's picking all used `alive`, so a stroke over a cropped
 * area found its surface in the hidden part and selected splats nobody
 * could see. One mask for all of them.
 */
let visibleCache: Uint8Array | null = null;
function visibleMask(): Uint8Array {
  if (!cropActive || !cropOut.length) return doc.alive;
  if (!visibleCache || visibleCache.length !== doc.alive.length) {
    const v = doc.alive.slice();
    for (let k = 0; k < cropOut.length; k++) {
      const i = cropOut[k];
      if (i < v.length) v[i] = 0;
    }
    visibleCache = v;
  }
  return visibleCache;
}

/** The view the selection was made from, else the camera now. */
function paintView(): [number, number, number] {
  return selView ?? viewer.viewDirection();
}

/**
 * The brush's depth window as a share of its radius (`segmentRows`) for a
 * stroke through `caps`, painted in `mode` from `view` on top of `base`.
 *
 * Painting: half the depth setting. Un-painting reaches deeper (volume.ts,
 * SUBTRACT_REACH): twice that, and at least twice as deep as anything in
 * `base` was painted (`paintedReach`) -- a small brush in mode "Subtract" over
 * a large one's stroke otherwise took the surface and left the stroke's
 * volume, and what it had selected deeper down, standing around its own
 * thin window. One function for the live preview, the GPU brush and the
 * exact result, so they cannot disagree.
 */
function strokeSquash(depth: number, mode: SelectMode, base: Volume | null,
                      view: [number, number, number], caps: CapsulePoint[]): number {
  const own = depth * 0.5;
  if (mode !== 'subtract') return own;
  let s = own * SUBTRACT_REACH;
  const reach = paintedReach(base, view);
  let rMin = Infinity;
  for (const c of caps) if (c.r < rMin) rMin = c.r;
  if (reach > 0 && rMin > 0 && isFinite(rMin)) s = Math.max(s, SUBTRACT_REACH * reach / rMin);
  return s;
}

/** Drop the stroke the depth slider would redo (another tool changed the selection). */
function forgetStroke() {
  strokeCaps = [];
  strokeBase = null;
}

/**
 * Build the stroke's capsules and show the selection.
 *
 * While the stroke is live only the GPU runs: it writes the tint straight into
 * the splat stream, so nothing has to come back across the bus. The exact index
 * list is worked out once when the stroke ends, where the cost does not matter.
 */
function previewSelection() {
  const p = ensureProjection();
  if (!p || !stroke.length) return;
  const depth = Number($<HTMLInputElement>('brush-depth').value);
  lastCaps = strokeToCapsules(
    grid!, stroke,
    (x, y, d) => viewer.unproject(x, y, d),
    (d) => viewer.worldPerPixelAt(d));
  // un-painting reaches deeper than painting (strokeSquash)
  const squash = strokeSquash(depth, selectMode, strokeStartVolume, viewer.viewDirection(), lastCaps);
  if (gpuSel?.ready
      && gpuSel.run(lastCaps, viewer.viewDirection(), squash, selectMode, sizeReach)) {
    // the outline follows the stroke; built when volumeview.ts gets to it
    const caps = lastCaps, mode = selectMode, base = strokeStartVolume;
    const view = viewer.viewDirection();
    volumeOutline?.preview(() => caps.length
      ? withOp(base, { kind: 'brush', add: true, view, rows: segmentRows(caps, squash) }, mode, false)
      : mode === 'replace' ? EMPTY_VOLUME : base);
    return;
  }
  recomputeSelection(strokeStart ?? undefined);   // no GPU path: the exact one
}

/**
 * The exact selection for the stroke, once it is finished.
 *
 * `base` is the selection the stroke started from (taken at pointerdown);
 * without it, the selection as it stands now.
 */
function recomputeSelection(base?: Uint8Array) {
  const p = ensureProjection();
  if (!p || !stroke.length) return;
  // Lift the stroke onto the surface it touches, then select the 3-D volume it
  // sweeps. `depth` squashes that volume towards the camera, so a wide brush
  // covers more surface without reaching further into the object.
  strokeCaps = strokeToCapsules(
    grid!, stroke,
    (x, y, d) => viewer.unproject(x, y, d),
    (d) => viewer.worldPerPixelAt(d));
  strokeBase = base ?? selMask.slice();
  strokeVolumeBase = base ? (strokeStartVolume ?? selVolume) : selVolume;
  selView = viewer.viewDirection();
  applyStroke();
}

/**
 * Select what the last stroke sweeps at the depth the slider says.
 *
 * World-space capsules and the view they were painted from, so moving the
 * slider after orbiting away still redoes the stroke that was painted. It
 * used to re-run only the GPU tint: the selection the buttons send stayed at
 * the old depth, and in lasso or box mode the old brush stroke reappeared.
 */
function applyStroke() {
  if (!strokeBase || !selView) return;
  const depth = Number($<HTMLInputElement>('brush-depth').value);
  const squash = strokeSquash(depth, selectMode, strokeVolumeBase, selView, strokeCaps);
  const hits = selectInCapsules(doc.centres, strokeCaps, selView,
                                { depth: squash, alive: visibleMask(), size: brushSize() });
  // the stroke as volume: exactly the capsules just tested
  let baseEmpty = true;
  for (let i = 0; i < strokeBase.length && baseEmpty; i++) if (strokeBase[i]) baseEmpty = false;
  setVolume(strokeCaps.length || selectMode !== 'replace'
    ? withOp(strokeVolumeBase, { kind: 'brush', add: true, view: selView,
                                 rows: segmentRows(strokeCaps, squash) },
             selectMode, baseEmpty)
    : EMPTY_VOLUME);
  commitHits(hits, strokeBase);
}

/** What the last lasso or box cost to find its surface (autotest). */
let lastRegion: { entries: number; gridMs: number } | null = null;

/** Lasso and box share everything but the shape they fill. */
function commitRegion(mask: Uint8Array, w: number, h: number,
                      poly: { x: number; y: number }[] = []) {
  const p = ensureProjection();
  if (!p) return;
  // The same reach the brush uses: a depth window measured in pixels at the
  // surface's own distance, so "surface only" means the same thing close up
  // and far away, and at any zoom.
  const depth = Number($<HTMLInputElement>('brush-depth').value);
  let tolPx = 6 * depth * (selectMode === 'subtract' ? SUBTRACT_REACH : 1);
  const ppu = viewer.worldPerPixelPerUnit();
  // "Surface": the surface as the renderer blends it, footprints and all
  // (selection.ts, surfaceGrid), worked out for the outline's cells only.
  // "Through" keeps the centre grid: it only sets the volume's depth range.
  let surface: DepthGrid | null = grid;
  lastRegion = null;
  if (selDepth === 'surface' && poly.length >= 3 && projMeta) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const q of poly) {
      if (q.x < x0) x0 = q.x; if (q.x > x1) x1 = q.x;
      if (q.y < y0) y0 = q.y; if (q.y > y1) y1 = q.y;
    }
    const t0 = performance.now();
    const R = viewer.camera.right, U = viewer.camera.up;
    const sg = surfaceGrid(p, projMeta.w, projMeta.h, {
      alive: visibleMask(), alpha: doc.alpha, covA: doc.covA, covB: doc.covB,
      right: [R.x, R.y, R.z], up: [U.x, U.y, U.z], perPixelPerUnit: ppu,
      box: [x0 - 1, y0 - 1, x1 + 1, y1 + 1]
    });
    surface = sg;
    lastRegion = { entries: sg.entries, gridMs: +(performance.now() - t0).toFixed(1) };
  }
  // holes inside the outline get the surrounding surface (fillRegionHoles)
  const filled = surface && poly.length >= 3
    ? fillRegionHoles(surface, mask, w, h, poly, regionHoleFill) : null;
  if (selectMode === 'subtract' && selDepth === 'surface' && filled) {
    // Un-painting takes all of what was painted there, not this tool's six
    // pixels of it: at least twice as deep as the volume reaches along this
    // view (volume.ts, paintedReach), at the outline's nearest surface.
    const reach = paintedReach(selVolume, viewer.viewDirection());
    const [gx0, gy0, gw, gh] = filled.crop;
    let nearest = Infinity;
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        const d = filled.grid.surf[(gy0 + y) * filled.grid.cw + gx0 + x];
        if (d < nearest) nearest = d;
      }
    }
    if (reach > 0 && isFinite(nearest) && nearest > 0) {
      tolPx = Math.max(tolPx, SUBTRACT_REACH * reach / (ppu * nearest));
    }
  }
  const hits = selectInRegion(p, mask, w, h, {
    depthMode: selDepth, tolPx,
    perPixelPerUnit: ppu,
    alive: visibleMask(), grid: filled?.grid ?? surface ?? undefined,
    size: { ...brushSize(), cap: sizeReach.k > 0 ? REGION_SIZE_CAP : 0 },
    viewDir: viewer.viewDirection()
  });
  forgetStroke();
  selView = viewer.viewDirection();
  if (filled && projMeta) {
    // the region as volume: the outline, the camera, the cells it kept
    const [gx0, gy0, gw, gh] = filled.crop;
    const depthCells = new Float32Array(gw * gh);
    let near = Infinity, far = 0;
    const back = tolPx * ppu;
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        const d = filled.grid.surf[(gy0 + y) * filled.grid.cw + gx0 + x];
        depthCells[y * gw + x] = d;
        if (isFinite(d)) {
          near = Math.min(near, d - back * d * 1.2);
          far = Math.max(far, d + back * d);
        }
      }
    }
    if (selDepth === 'through') {
      for (let k = 0; k < hits.length; k++) {
        const d = p.dz[hits[k]];
        if (d < near) near = d;
        if (d > far) far = d;
      }
    }
    let baseEmpty = selCount === 0;
    if (isFinite(near) && far > 0) {
      setVolume(withOp(selVolume, {
        kind: 'region', add: true, view: selView,
        vp: projMeta.vp, width: projMeta.w, height: projMeta.h,
        polygon: poly.map(q => ({ x: q.x, y: q.y })), through: selDepth === 'through',
        crop: filled.crop, cw: filled.grid.cw, ch: filled.grid.ch, depth: depthCells,
        backPerDepth: back, frontScale: 1.2,
        near: Math.max(near, 1e-3), far
      }, selectMode, baseEmpty));
    } else if (selectMode === 'replace') {
      setVolume(EMPTY_VOLUME);       // drawn over nothing at all
    }
  }
  commitHits(hits);
}

function commitLasso() {
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, r.width | 0), h = Math.max(1, r.height | 0);
  if (lassoPts.length >= 3) commitRegion(fillPolygon(lassoPts, w, h), w, h, lassoPts);
  lassoPts = [];
  drawGuides();
}

function commitBox() {
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, r.width | 0), h = Math.max(1, r.height | 0);
  if (boxStart && boxNow) {
    const x0 = Math.min(boxStart.x, boxNow.x), x1 = Math.max(boxStart.x, boxNow.x);
    const y0 = Math.min(boxStart.y, boxNow.y), y1 = Math.max(boxStart.y, boxNow.y);
    if (x1 - x0 > 2 && y1 - y0 > 2) {
      const corners = [{ x: x0, y: y0 }, { x: x1, y: y0 }, { x: x1, y: y1 }, { x: x0, y: y1 }];
      commitRegion(fillPolygon(corners, w, h), w, h, corners);
    }
  }
  boxStart = boxNow = null;
  drawGuides();
}

/** Materialise the index list the engine needs, only when it is needed. */
function selectionIndices(): Uint32Array<ArrayBufferLike> {
  if (selDirty) {
    selection = maskToIndices(selMask);
    selDirty = false;
  }
  return selection;
}

/**
 * Fold a hit set into the selection under the active modifier -- into
 * `base` instead of the current selection when one is given.
 */
function commitHits(hits: Uint32Array<ArrayBufferLike>, base?: Uint8Array | null) {
  if (base) {
    const next = new Uint8Array(doc.nTotal);
    next.set(base.subarray(0, Math.min(base.length, next.length)));
    selMask = next;
  } else if (selMask.length !== doc.nTotal) {
    const next = new Uint8Array(doc.nTotal);
    next.set(selMask.subarray(0, Math.min(selMask.length, next.length)));
    selMask = next;
  }
  applyMode(selMask, hits, selectMode);
  selCount = 0;
  for (let i = 0; i < selMask.length; i++) selCount += selMask[i];
  selDirty = true;
  afterSelectionChanged();
}

function afterSelectionChanged() {
  applyStates();
  counts();
  if (tool === 'xform' && gizmo && !gizmo.isDragging) refreshGizmo();
  const n = selCount;
  $('sel-count').textContent = n ? `${fmt(n)} splats selected.` : 'Nothing selected.';
  $<HTMLButtonElement>('btn-repair').disabled = n === 0;
  $<HTMLButtonElement>('btn-delete').disabled = n === 0;
  $<HTMLButtonElement>('btn-unshadow').disabled = n === 0;
  $<HTMLButtonElement>('btn-solidify').disabled = n === 0;
  // the shadow note speaks of the last press; a new selection makes it stale
  // (an edit keeps its selection while the engine is busy, and its note)
  if (!engineBusy) shadowNote(null);
  // a selection change is a history step, as in Photoshop -- but not the
  // live preview of a stroke, only its result
  if (!painting) history.noteSelection(selectionLabel());
  if (!painting) scheduleObjectPreview();
}

/* ----------------------------------------------------------- object select */

/**
 * "Detect object", the brush's select-subject.
 *
 * After a stroke the engine works out which object it covers (objsel.py):
 * whatever stands proud of the surface or differs from its material, grown
 * through touching splats to the object's sides, its faint edge and its drop
 * shadow. The outline shows that at once, so what the repair will take is
 * visible before anything changes. The repair takes it; an object standing
 * clear of a whole surface (the pole before the hut's wall) is only deleted,
 * and the wall the brush touched behind it stays.
 *
 * The selection mask is left as painted: undo, the depth slider and the
 * other buttons keep working on the stroke itself. Off, the repair takes
 * exactly what was painted.
 */
let objectSerial = 0;
let objectTimer = 0;
/** The last answer, for the autotest: mode, how many splats, how long. */
let lastObject: { mode: string; count: number; object: number; seconds: number;
                  info: Record<string, any> } | null = null;
/** Painted selections larger than this are not looked at (debris, invert). */
const OBJECT_MAX = 400_000;

function objectSelectOn(): boolean {
  const el = document.getElementById('obj-select') as HTMLInputElement | null;
  return !el || el.checked;
}

/** Was something taken out of the painted selection again (mode "Subtract", Ctrl)? */
function selectionCut(): boolean {
  return !!selVolume && selVolume.ops.some(o => !o.add);
}

/**
 * Grow this selection to the object it covers? Only as painted in one go.
 * The engine grows from the selected splats through whatever touches them
 * (objsel.py), and knows nothing of what was un-painted: a part of the
 * object taken out of the selection touches the rest and would come back
 * with the grow -- in the outline, and in the repair. Whoever takes
 * something away says they do not want the whole thing; from then on the
 * repair takes the selection as it is. (Said in the checkbox's tooltip, not
 * in the selection line: there it added four lines with every stroke taken
 * away, and moved the whole panel.)
 */
function objectGrow(): boolean {
  return objectSelectOn() && !selectionCut();
}

function scheduleObjectPreview() {
  objectSerial++;
  clearTimeout(objectTimer);
  lastObject = null;
  volumeOutline?.setObject(null);
  if (!objectGrow() || !selCount || selCount > OBJECT_MAX || !selVolume
      || !doc.nTotal || tool === 'clone') return;
  const serial = objectSerial;
  objectTimer = window.setTimeout(() => { void objectPreview(serial); }, 150);
}

async function objectPreview(serial: number) {
  if (serial !== objectSerial) return;
  if (engineBusy) {             // an edit is running: ask again once it is done
    objectTimer = window.setTimeout(() => { void objectPreview(serial); }, 400);
    return;
  }
  try {
    const r = await engine('object_select', { view: paintView(), volume: volumeArg() },
                           selectionPayload());
    if (serial !== objectSerial) return;        // the selection moved on meanwhile
    const h = r.header;
    const ids = (h.object ?? []) as number[];
    const all = (h.indices ?? []) as number[];
    lastObject = { mode: String(h.mode), count: all.length, object: ids.length,
                   seconds: Number(h.seconds ?? 0), info: h.info ?? {} };
    if (h.mode === 'none') return;
    // the outline: everything the repair will take (lift: the object alone)
    const show = h.mode === 'lift' ? ids : all;
    const c = doc.centres;
    const pts = new Float32Array(show.length * 3);
    let k = 0;
    for (const i of show) {
      if (i >= doc.nTotal) continue;
      pts[k++] = c[i * 3]; pts[k++] = c[i * 3 + 1]; pts[k++] = c[i * 3 + 2];
    }
    volumeOutline?.setObject(pts.subarray(0, k), 2 * Number(h.info?.spacing ?? 0));
    const grown = Math.max(0, all.length - selCount);
    $('sel-count').textContent = `${fmt(selCount)} splats selected · ` + (h.mode === 'lift'
      ? `Object stands free: Repair only deletes the object (${fmt(ids.length)} splats), the surface behind it stays.`
      : `Object detected: Repair takes ${fmt(grown)} more splats with it (sides, edges, shadow).`);
  } catch {
    /* only a preview: never in the way of the brush */
  }
}

function clearSelection() {
  objectSerial++;               // an object preview on its way is stale now
  lastObject = null;
  selMask = new Uint8Array(doc.nTotal);
  selCount = 0;
  selection = new Uint32Array(0);
  selDirty = false;
  stroke = [];
  forgetStroke();
  strokeStart = null;
  selView = null;
  setVolume(EMPTY_VOLUME);
  lassoPts = [];
  boxStart = boxNow = null;
  applyStates();
  counts();
  // The handles go with the selection. They used to stay after Esc, Ctrl+D,
  // Del or a repair, and dragging them moved the next selection on screen.
  if (gizmo && !gizmo.isDragging) refreshGizmo();
  $('sel-count').textContent = 'Nothing selected.';
  $<HTMLButtonElement>('btn-repair').disabled = true;
  $<HTMLButtonElement>('btn-unshadow').disabled = true;
  $<HTMLButtonElement>('btn-solidify').disabled = true;
  $<HTMLButtonElement>('btn-delete').disabled = true;
  history.noteSelection('Deselect');
}

/* ------------------------------------------------------------------ guides */

/**
 * The lasso path and the box, drawn as thin 2-D lines.
 *
 * This is the only thing left on a 2-D layer: it is the tool's own outline, not
 * a stand-in for the selection. What is selected is tinted by the splat shader,
 * where it occludes correctly.
 */
function drawGuides() {
  const r = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio, 2);
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (guides.width !== w || guides.height !== h) { guides.width = w; guides.height = h; }
  const ctx = guides.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(229, 162, 98, 0.95)';
  ctx.fillStyle = 'rgba(229, 162, 98, 0.13)';

  if (tool === 'lasso' && lassoPts.length > 1) {
    ctx.beginPath();
    ctx.moveTo(lassoPts[0].x, lassoPts[0].y);
    for (let i = 1; i < lassoPts.length; i++) ctx.lineTo(lassoPts[i].x, lassoPts[i].y);
    if (!painting) ctx.closePath();
    ctx.fill();
    ctx.stroke();
    if (painting) {
      // show where it will close, so the shape is never a surprise
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(lassoPts[lassoPts.length - 1].x, lassoPts[lassoPts.length - 1].y);
      ctx.lineTo(lassoPts[0].x, lassoPts[0].y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  } else if (tool === 'box' && boxStart && boxNow) {
    const x = Math.min(boxStart.x, boxNow.x), y = Math.min(boxStart.y, boxNow.y);
    const bw = Math.abs(boxNow.x - boxStart.x), bh = Math.abs(boxNow.y - boxStart.y);
    ctx.fillRect(x, y, bw, bh);
    ctx.strokeRect(x, y, bw, bh);
  }
}

/* -------------------------------------------------------------------- edit */

function selectionPayload(): ArrayBuffer {
  const a = new Uint32Array(selectionIndices());
  return a.buffer.slice(0) as ArrayBuffer;
}

/**
 * A reason from the engine, fit for the status line: its first line, spaces
 * folded, no full stop, cut at `max` characters. (What torch says when it
 * runs out of memory is five lines long.)
 */
function oneLine(s: string, max = 110): string {
  const t = (s.split(/\r?\n/).find(l => l.trim()) ?? '').replace(/\s+/g, ' ').trim().replace(/\.$/, '');
  return t.length > max ? t.slice(0, max - 1).trimEnd() + '…' : t;
}

async function doRepair(reroll?: { idx: ArrayBuffer; deep: boolean; seed: number;
                                   view: [number, number, number]; objects: boolean }) {
  // One command at a time. The overlay only covers the viewport: buttons,
  // menus and keys stay live, and a second click on "Repair" queued a
  // second repair that became an undo step of its own, with no selection to
  // bring back.
  if (engineBusy) return;
  if (!reroll && !selCount) return;
  const deep = reroll ? reroll.deep : depthMode === 'deep';
  const seed = reroll ? reroll.seed : 0;
  const idx = reroll ? reroll.idx : selectionPayload();
  // which side of the surface is the front: the one facing the camera the
  // selection was made from (a reroll keeps the first fill's)
  const view = reroll ? reroll.view : paintView();
  // the painted volume, for masks and holes without splats (a reroll keeps its own)
  const volume = reroll ? lastFillVolume : volumeArg();
  // grown to the object or as painted (a reroll: as the first fill was, the
  // selection it could be read from is gone by then)
  const objects = reroll ? reroll.objects : objectGrow();
  setBusy(true, deep ? 'Deep Fill …' : 'Repairing …');
  status(deep ? 'deep fill …' : 'repairing …');
  try {
    const params = {
      mode: $<HTMLSelectElement>('mode').value,
      density: Number($<HTMLInputElement>('density').value),
      roughness: Number($<HTMLInputElement>('roughness').value),
      color_smooth: Number($<HTMLInputElement>('colorsmooth').value),
      view,
      seed
    };
    const t0 = performance.now();
    const r = deep
      ? await engine('deep_repair', {
          params, fill: fillChoice(), volume, objects,
          deep: { detail: Number($<HTMLInputElement>('deep-detail').value), seed }
        }, idx.slice(0))
      // `engine`: which fill closes the hole, for this call ("Fill" under
      // Advanced). A reroll takes what the menu says now, so the same
      // stroke can be filled once each way.
      : await engine('repair', { params, volume, objects, engine: fillEngine() },
                     idx.slice(0));
    const h = r.header;
    lastFillInfo = h.info ?? null;
    doc.setDead(Uint32Array.from(h.removed as number[]));
    if (h.addedCount > 0 && r.payload) {
      doc.appendGenerated(new Float32Array(r.payload), h.layout);
    }
    // The engine has the edit on its undo stack from here on, so the history
    // gets its step now. It used to come last, after the rebuild and the
    // status line: anything failing in between left the two stacks apart.
    // (The selection is cleared below; the history folds that into this step.)
    setHistory(h);
    {
      const top = history.entries[history.index];
      lastFill = { idx, deep, seed, serial: editSerial, view, objects,
                   step: top?.kind === 'edit' ? top : null };
    }
    lastFillVolume = volume;
    refreshReroll();
    rebuild();
    clearSelection();
    // The picture fill takes a further picture from the side where the
    // surface turns away from the view: say so, it explains the extra second.
    const views = Number(h.info?.passes ?? 1);
    const how = h.info?.mode === 'image'
      ? (views > 1 ? `filled in a picture of the surface, ${views} views` : 'filled in a picture of the surface')
      : h.info?.mode === 'patch' ? 'patch transplanted'
      : h.info?.mode === 'delete' ? 'only deleted, the surface behind is intact'
      : 'seeded from the rim';
    // The picture fill was tried on this spot and gave up, and the copying
    // fill ran in its place (engine: Document._image_fill, `imageFallback`).
    // Nothing said so: the result just looked unlike the fill before it.
    const fallback = h.info?.imageFallback
      ? ' · picture fill could not do this spot'
        + (h.info?.mode === 'delete' ? '' : ', the copying fill took over')
        + ` (${oneLine(String(h.info.imageFallback))})`
      : '';
    let note = how;
    if (deep && h.info?.mode !== 'delete') {
      note = h.info?.deep
        // What the number means, in words: how far the splats ended up from
        // the inpainted picture, and whether the patch holds from the side.
        ? `${how} + ${h.info.fill_source ?? h.info.inpainter}`
          + (h.info.fill_fallback ? ` (Generative Fill: ${h.info.fill_fallback})` : '')
          + ` · ${num(100 * (h.info.fit ?? 0), 1)}% off the fill picture`
          + formatSideError(h.info)
        : `${how} · Deep Fill discarded`
          + (h.info?.fill_source ? ` (${h.info.fill_source})` : '')
          + `: ${h.info?.rejected ?? 'unclear'}`;
    }
    status(`repaired: −${fmt(h.removed.length)} / +${fmt(h.addedCount)} · ${note} · ` +
           `${num(h.seconds)} s + ${num((performance.now() - t0) / 1000 - h.seconds)} s rebuild`
           + fallback
           + (h.info?.warning ? ` · note: ${h.info.warning}` : ''),
           h.info?.warning || fallback || (deep && !h.info?.deep) ? '' : 'ok');
    // The quick fill is there at once; where a GPU and LaMa are, the deep
    // fill refines it behind the user's back and swaps in as the same step.
    // The picture fill (info.engine 'image') is already what a refine aims for.
    if (!deep && caps.viewFill && fillChoice() === 'lama' && h.addedCount > 0
        && h.info?.mode !== 'delete' && h.info?.engine !== 'image') {
      // (the refine writes the status line; the fallback note stays on it)
      refineLast(editSerial, `${how} · ${num(h.seconds)} s`, fallback);
    }
  } catch (e: any) {
    status(String(e.message ?? e), 'err');
  } finally {
    setBusy(false);
  }
}

/**
 * Refine the last quick fill with the deep fill, like a generative preview.
 *
 * The engine recolours the quick fill's own splats (`refine`): same count,
 * same indices, one undo step. The viewport swaps the tail when the reply
 * comes. Nothing is locked meanwhile; an edit made in between queues behind
 * the refine in the engine and lands on the refined fill, which is why the
 * swap is still right. Only an edit that has already come back means the
 * reply is stale.
 *
 * The engine swaps the refine in only when it looks better than the quick
 * fill (`splatcore/judge.py`); otherwise nothing changes here and the
 * status line says which one stayed and why (`refineKept`, `info.judge.why`).
 */
async function refineLast(serial: number, quickNote: string, tail = '') {
  // `tail`: what the repair's own status line ended on and this one must
  // not lose (the picture fill's fallback note); with it the line is no "ok"
  const kind = tail ? '' : 'ok';
  status(`repaired · ${quickNote} · refining in the background (LaMa) …${tail}`);
  try {
    const r = await engine('refine', { fill: 'lama', deep: { seed: 0 } });
    const h = r.header;
    if (editSerial !== serial) return;
    if (h.refined && r.payload) {
      doc.truncateGenerated(h.addedCount);
      doc.appendGenerated(new Float32Array(r.payload), h.layout);
      // Same number of splats, other colours: rebuild() uploads only when the
      // count changed, so the refined fill never reached the viewport.
      lastGeneratedCount = -1;
      rebuild();
      status(`refined · ${num(h.seconds)} s${tail}`, kind);
    } else if (h.refineKept === false) {
      status(`Fast Fill kept, ${h.info?.judge?.why ?? 'it looked better'}${tail}`, kind);
    } else {
      status(`repaired · ${quickNote} · refinement discarded: `
        + `${h.info?.refine?.rejected ?? 'no gain'}${tail}`);
    }
  } catch (e: any) {
    if (editSerial === serial) {
      status(`repaired · could not refine: ${String(e.message ?? e)}${tail}`);
    }
  }
}

/**
 * The shadow brush, in one of two ways (the "Shadow" menu):
 *
 * - "Find Shadow" (auto): the engine finds the shadow's edge in the
 *   stroke and takes out exactly the shadow. Precise where it works -- but it
 *   refuses whenever it cannot tell shadow from material, and on real scans
 *   that is most strokes: soft shadows on grained surfaces, sparse walls
 *   where a stroke holds a hundred splats. Its refusal used to be one line
 *   in the status bar and nothing else, which read as "does nothing".
 * - "Lighten" (lift): brightens what was painted by the slider's amount,
 *   one factor for all channels, soft towards the stroke's rim. Always does
 *   something visible, like a dodge in an image editor.
 *
 * Either way the engine hands back the same splats with corrected colour, so
 * the viewport swaps them like any other edit and one Ctrl+Z undoes it. The
 * selection stays on the lifted splats, so a second click lifts again.
 */
type ShadowMode = 'auto' | 'lift';
const shadowVals: Record<ShadowMode, number> = { auto: 1, lift: 0.5 };

function shadowMode(): ShadowMode {
  return $<HTMLSelectElement>('shadow-mode').value === 'lift' ? 'lift' : 'auto';
}

function showShadowValue() {
  const v = Number($<HTMLInputElement>('shadow-strength').value);
  $('shadow-strength-v').textContent = shadowMode() === 'lift'
    ? `+${Math.round(v * 100)}%` : `${Math.round(v * 100)}%`;
}

function applyShadowMode() {
  const lift = shadowMode() === 'lift';
  const input = $<HTMLInputElement>('shadow-strength');
  input.min = lift ? '0.1' : '0';
  input.max = lift ? '3' : '1';
  input.step = lift ? '0.1' : '0.05';
  input.value = String(shadowVals[shadowMode()]);
  $('shadow-strength-l').textContent = lift ? 'Lighten by' : 'Strength';
  $('btn-unshadow').textContent = lift ? 'Lighten' : 'Remove Shadow';
  $('shadow-hint').textContent = lift
    ? 'Paint over the spot. Its middle gets lighter by the set amount, dark parts '
      + 'more, light parts less. White stays white, pattern and hue stay.'
    : 'Paint generously over the shadow, past its edge too. Where the shadow reaches beyond '
      + 'the stroke, the brush follows it a little and lightens that as well.';
  showShadowValue();
  shadowNote(null);
}

/** What the last press did, or why not -- in the tile, where the user looks. */
function shadowNote(text: string | null, offerLift = false) {
  const el = $<HTMLParagraphElement>('shadow-note');
  el.hidden = !text;
  el.textContent = text ?? '';
  $<HTMLDivElement>('shadow-instead').hidden = !(text && offerLift);
}

/** The engine's reasons read "a -- b"; in the tile "a – b". */
const reasonText = (s: string) => s.replace(/\s+--\s+/g, ' – ').replace(/\.$/, '');

/**
 * Why "Find Shadow" found nothing, said plainly. The two refusals real
 * strokes hit almost always get their own sentence; the rest keep the
 * engine's reason.
 */
function whyNoShadow(msg: string): string {
  const m = msg.replace(/;\s+no reflection either\s*$/, '');
  const grain = m.match(/no shadow found in the stroke \(step (-?[\d.]+) times the grain\)/);
  if (grain) {
    return 'The spot is not clearly darker than the pattern of the surface itself '
      + `(step ${grain[1]} of the grain), no shadow edge can be made out.`;
  }
  // the picture's refusals (engine: _picture_shadow)
  const flat = m.match(/nothing in it is clearly darker than the surface around it \(step ([\d.]+), needed ([\d.]+)\)/);
  if (flat) {
    return `Nothing in the stroke is darker than the surface around it (step ${flat[1]}, needed ${flat[2]}).`;
  }
  if (/continues the pattern or the shape around it/.test(m)) {
    return 'What is dark in the stroke continues the pattern or shape around it. '
      + 'The picture without the stroke shows it just as dark, so it is not a shadow.';
  }
  if (/nothing in it is darker than the surface around it\s*$/.test(m)) {
    return 'Nothing in the stroke is darker than the surface around it.';
  }
  if (/nowhere brighter in the image/.test(m)) {
    return 'Nothing in the stroke is darker than the surface around it, and the same '
      + 'surface is nowhere lighter in the picture.';
  }
  if (/the light falls off across the whole part/.test(m)) {
    return 'The surface around it is just as dark; more likely the light falls off across the '
      + 'whole part. If it is a shadow after all, paint past its edge into the light.';
  }
  if (/more than four times darker/.test(m)) {
    return 'The dark part is more than four times darker than the same surface in the light. '
      + 'That is more likely a gap or another part than a shadow.';
  }
  if (/^no shadow found in the stroke\s*$/.test(m)) {
    return 'No shadow edge can be made out in the stroke. Often the stroke hits too few '
      + 'splats or lies entirely in shadow; painting past the edge helps.';
  }
  return reasonText(m) + '.';
}

/**
 * Keep the selection on the splats an in-place edit replaced. The engine
 * appends the corrected copies in the order of `removed`, from `first` on;
 * whatever else was selected stays as it was.
 */
function keepSelectionAfterEdit(removed: number[], first: number) {
  const next = new Uint8Array(doc.nTotal);
  next.set(selMask.subarray(0, Math.min(selMask.length, next.length)));
  for (let k = 0; k < removed.length; k++) {
    const i = removed[k];
    if (next[i]) {
      next[i] = 0;
      if (first + k < next.length) next[first + k] = 1;
    }
  }
  selMask = next;
  selCount = 0;
  for (let i = 0; i < next.length; i++) selCount += next[i];
  selDirty = true;
  forgetStroke();
  afterSelectionChanged();
}

async function doUnshadow() {
  if (!selCount || engineBusy) return;
  const mode = shadowMode();
  const v = Number($<HTMLInputElement>('shadow-strength').value);
  setBusy(true, mode === 'lift' ? 'Lightening …' : 'Finding the shadow …');
  try {
    const params = mode === 'lift' ? { mode, lift: 1 + v } : { mode, strength: v };
    const r = await engine('unshadow', { view: paintView(), params, volume: volumeArg() },
                           selectionPayload());
    const h = r.header;
    doc.setDead(Uint32Array.from(h.removed as number[]));
    if (h.addedCount > 0 && r.payload) {
      doc.appendGenerated(new Float32Array(r.payload), h.layout);
    }
    setHistory(h);              // now, not after the notes below (see doRepair)
    rebuild();
    keepSelectionAfterEdit(h.removed as number[], h.addedFirstIndex as number);
    lastInfo = h.info;
    const i = h.info;
    const warn = i.warning ? ` Note: ${reasonText(i.warning)}.` : '';
    let note: string;
    let offer = false;
    if (i.mode === 'lift') {
      // a dodge curve (engine: _lift_map): the stroke's middle by the
      // factor, darker parts more, lighter ones less; say so, or the lighter
      // result in the light parts reads as the brush not doing what was set
      note = (i.curve
        ? `Lightened by ×${num(i.factor, 2)}, dark parts more, light parts less `
        : `Lightened by ×${num(i.factor, 2)} `)
        + `(${fmt(i.lifted)} splats). Click again to lighten further. Ctrl+Z undoes it.`
        + warn;
      status(`lightened ×${num(i.factor, 2)} · ${fmt(i.lifted)} splats · ${num(h.seconds)} s`
             + (i.warning ? ` · note: ${i.warning}` : ''), i.warning ? '' : 'ok');
    } else {
      const parts: string[] = [];
      // the lift, not the shadow's gain: "×0.42" read as darker
      const lift = i.shadow ? (i.gain as number[]).map(x => 1 / x) : [];
      if (i.shadow) parts.push(`shadow lightened: ×${lift.map(x => num(x, 2)).join(' / ')} (R/G/B)`);
      if (i.reflex) parts.push(`reflection removed (${fmt(i.reflexCells)} cells)`);
      if (i.shadow) {
        note = `Shadow found and lightened by ×${num(Math.max(...lift), 2)}`
          + (i.edge === 'picture' || i.picture ? ` (${fmt(i.lifted)} splats in the picture)` : '')
          // the region grew past the stroke (engine: _grow_shadow): say so,
          // or the lifted area beyond the stroke looks like a slip
          + (i.grown ? ', also where it reaches beyond the stroke' : '')
          + '. Ctrl+Z undoes it.' + warn;
      } else {
        // Only a reflection went. Say so: a dark patch that stayed under a
        // status line marked "ok" is the brush "doing nothing".
        note = 'Only removed a coloured reflection, no shadow. '
          + whyNoShadow(i.shadowWhyNot ?? 'unclear')
          + ' You can lighten the spot yourself instead.';
        offer = true;
      }
      status(`${parts.join(' · ')} · ${fmt(i.lifted)} splats · ${num(h.seconds)} s`
             + (i.warning ? ` · note: ${i.warning}` : ''),
             i.warning || !i.shadow ? '' : 'ok');
    }
    shadowNote(note, offer);
  } catch (e: any) {
    const msg = String(e.message ?? e);
    status(msg, 'err');
    // A hole is for the repair brush; anything else the user can lift by hand.
    const hole = /hole/.test(msg);
    shadowNote(mode === 'lift'
      ? `Not lightened: ${reasonText(msg)}.`
      : whyNoShadow(msg)
        + (hole ? '' : ' You can lighten the spot yourself instead.'),
      mode === 'auto' && !hole);
  } finally {
    setBusy(false);
  }
}

/**
 * Make a see-through patch as solid as the surface around it. The view the
 * selection was painted from goes along -- the view in which the patch showed
 * through, which the engine measures from. Captured when the stroke, lasso or
 * box was committed, not when the button is pressed: by then the user may
 * well have orbited away to look at what they selected.
 */
async function doSolidify() {
  if (!selCount || engineBusy) return;
  setBusy(true, 'Solidifying …');
  try {
    const r = await engine('solidify', {
      view: paintView(), volume: volumeArg(),
      params: { colour: $<HTMLSelectElement>('solid-colour').value }
    }, selectionPayload());
    const h = r.header;
    doc.setDead(Uint32Array.from(h.removed as number[]));
    if (h.addedCount > 0 && r.payload) {
      doc.appendGenerated(new Float32Array(r.payload), h.layout);
    }
    setHistory(h);              // now, not after the status line (see doRepair)
    rebuild();
    clearSelection();
    const b = h.info.coverageBefore as number[], a = h.info.coverageAfter as number[];
    status(`solidified: ${fmt(h.info.raised)} splats · coverage ${num(b[0], 3)} → ${num(a[0], 3)}`
           + ` (head-on), ${num(b[1], 3)} → ${num(a[1], 3)} (at an angle)`
           + (h.info.colour === 'halten' ? ' · look preserved' : '')
           + ` · ${num(h.seconds)} s`
           + (h.info.warning ? ` · note: ${h.info.warning}` : ''),
           h.info.warning ? '' : 'ok');
  } catch (e: any) {
    status(String(e.message ?? e), 'err');
  } finally {
    setBusy(false);
  }
}

async function doDelete() {
  if (!selCount || engineBusy) return;
  setBusy(true, 'Deleting …');
  try {
    const r = await engine('delete', { volume: volumeArg() }, selectionPayload());
    doc.setDead(Uint32Array.from(r.header.removed as number[]));
    setHistory(r.header);       // now, not after the rebuild (see doRepair)
    rebuild();
    const n = r.header.removed.length;
    clearSelection();
    status(`${fmt(n)} splats deleted`, 'ok');
  } catch (e: any) {
    status(String(e.message ?? e), 'err');
  } finally {
    setBusy(false);
  }
}

/** Enable or disable both history buttons from one engine reply. */
function setHistory(h: any) {
  $<HTMLButtonElement>('btn-undo').disabled = !h.canUndo;
  $<HTMLButtonElement>('btn-redo').disabled = !h.canRedo;
  editSerial++;
  refreshReroll();
  history.noteEdit(h);      // an edit becomes a step; the history's own undo/redo do not
}

/**
 * Re-filling is only offered while the last edit is still that fill, and its
 * step is the one on top of the history: "Retry" undoes whatever is on top.
 * After a new selection step it took that back instead and then repaired on
 * top of a fill that was never undone.
 */
function rerollReady(): boolean {
  return !!lastFill && lastFill.serial === editSerial && !!lastFill.step
    && history.entries[history.index] === lastFill.step;
}

function refreshReroll() {
  $<HTMLButtonElement>('btn-reroll').disabled = !rerollReady();
}

async function doUndo() {
  if (engineBusy) return;
  setBusy(true, 'Undoing …');
  try {
    const r = await engine('undo');
    if (!r.header.applied) { status('nothing to undo'); setHistory(r.header); return; }
    doc.setAlive(Uint32Array.from(r.header.restored as number[]));
    if (r.header.droppedTail) doc.truncateGenerated(r.header.droppedTail);
    // The engine has taken the step back from here on, so the history hears
    // of it now (see doRepair). It used to come last: a rebuild or a status
    // line that failed left the history a step ahead of the engine.
    setHistory(r.header);
    rebuild();
    clearSelection();
    refreshGizmo();
    status('undone', 'ok');
  } catch (e: any) {
    status(String(e.message ?? e), 'err');
  } finally {
    setBusy(false);
  }
}

async function doRedo() {
  if (engineBusy) return;
  setBusy(true, 'Redoing …');
  try {
    const r = await engine('redo');
    if (!r.header.applied) { status('nothing to redo'); setHistory(r.header); return; }
    const h = r.header;
    doc.setDead(Uint32Array.from(h.removed as number[]));
    // The engine kept the exact splats this edit had appended, so they come
    // back as they were -- a repair is never re-run and never differs.
    if (h.addedCount > 0 && r.payload) {
      doc.appendGenerated(new Float32Array(r.payload), h.layout);
    }
    setHistory(h);              // now, not after the status line (see doUndo)
    rebuild();
    clearSelection();
    refreshGizmo();
    status('redone', 'ok');
  } catch (e: any) {
    status(String(e.message ?? e), 'err');
  } finally {
    setBusy(false);
  }
}

async function doExport() {
  if (!filePath) return;
  if (engineBusy) { status('Export is available once the current job is done'); return; }
  // Suggest the format it came in as; the extension the user ends up with is
  // what the engine writes, so the save dialog is the format picker.
  const stem = filePath.replace(/\.(ply|splat|spz)$/i, '');
  const out = await window.host.saveSplatDialog(`${stem}_repaired.${sourceFormat}`);
  if (!out) return;
  // a key can have started something while the dialog was open
  if (engineBusy) { status('not exported: something is still running, try again when it is done'); return; }
  const shCap = $<HTMLSelectElement>('export-sh').value;
  // what this export writes: the document as the history has it now
  const written = history.revision;
  setBusy(true, 'Exporting …');
  try {
    const r = await engine('save', {
      path: out,
      options: shCap === '' ? {} : { sh_degree: Number(shCap) }
    });
    const h = r.header;
    exportedRevision = written;
    // Say what the format had to give up. "Exported" on its own reads the
    // same whether or not half the radiance field went with it.
    const cost: string[] = [];
    if (h.shBandsCapped) cost.push(`${h.shBandsCapped} SH bands capped`);
    if (h.shBandsDropped) cost.push(`${h.shBandsDropped} SH bands dropped`);
    if (h.coloursClipped) cost.push(`${fmt(h.coloursClipped)} colours clipped`);
    if (h.positionsClipped) cost.push(`${fmt(h.positionsClipped)} positions clipped`);
    // What the engine warns of (`warning`: .spz stores a very wide scene's
    // positions with fewer than 12 fractional bits). Such an export, or one
    // that clipped positions, is written but is no plain "ok".
    const warn = h.warning ? oneLine(String(h.warning), 160) : '';
    status(`exported as ${h.format}: ${fmt(h.written)} splats, `
           + `${num(h.bytes / 1048576, 1)} MB`
           + (cost.length ? ` · ${cost.join(', ')}` : '')
           + (warn ? ` · note: ${warn}` : ''),
           warn || h.positionsClipped ? '' : 'ok');
  } catch (e: any) {
    status(String(e.message ?? e), 'err');
  } finally {
    setBusy(false);
  }
}

/** Push the current deleted/selected flags to the GPU. Microseconds, no upload. */
function applyStates() {
  if (!viewer.splat) return;
  // While a stroke is live the GPU owns editSel; writing it here would
  // overwrite the preview with a frame-old CPU result.
  // The crop preview is folded into the deleted flags here, so it survives
  // the next selection change -- it used to vanish the moment anything was
  // selected, while the crop button stayed armed.
  if (painting && gpuSel?.ready && tool === 'brush') {
    viewer.writeStream(viewer.splat, 'editDel', cropHidden(doc.baseDeleted(), 0));
    if (viewer.generated) {
      viewer.writeStream(viewer.generated, 'editDel', cropHidden(doc.genDeleted(), doc.nBase));
    }
    return;
  }
  const baseSel = doc.baseSelectedMask(selMask);
  viewer.writeStream(viewer.splat, 'editDel', cropHidden(doc.baseDeleted(), 0));
  viewer.writeStream(viewer.splat, 'editBase', baseSel);
  viewer.writeStream(viewer.splat, 'editSel', baseSel);
  if (viewer.generated) {
    const genSel = doc.genSelectedMask(selMask);
    viewer.writeStream(viewer.generated, 'editDel', cropHidden(doc.genDeleted(), doc.nBase));
    viewer.writeStream(viewer.generated, 'editBase', genSel);
    viewer.writeStream(viewer.generated, 'editSel', genSel);
  }
}

/**
 * Re-upload only what a repair generated. The loaded file stays on the GPU
 * exactly as it arrived; deleting from it is a flag, not a rebuild.
 */
function rebuild() {
  const t0 = performance.now();
  if (doc.nGenerated !== lastGeneratedCount) {
    viewer.setGenerated(doc.buildGenerated());
    lastGeneratedCount = doc.nGenerated;
  }
  // The crop preview follows the document: an undo brings back splats it
  // must hide, a repair adds some. It used to be worked out only when a
  // slider moved, and went stale. (updateCrop pushes the flags.)
  updateCrop();
  projDirty = true;
  counts();
  return performance.now() - t0;
}
let lastGeneratedCount = 0;

/* ------------------------------------------------------------ transformieren */

function selectionPivot(): [number, number, number] | null {
  if (!selCount) return null;
  const c = doc.centres;
  let x = 0, y = 0, z = 0, n = 0;
  for (let i = 0; i < selMask.length; i++) {
    if (!selMask[i]) continue;
    x += c[i * 3]; y += c[i * 3 + 1]; z += c[i * 3 + 2]; n++;
  }
  return n ? [x / n, y / n, z / n] : null;
}

function refreshGizmo() {
  if (!gizmo) return;
  const pivot = selectionPivot();
  if (tool !== 'xform' || !pivot) {
    gizmo.show(null);
    $('xform-state').textContent = selCount
      ? 'Handles appear in the view.' : 'Select something first.';
    return;
  }
  gizmo.show(gizmoMode, pivot);
  $('xform-state').textContent =
    `${fmt(selCount)} Splats · Griffe am Schwerpunkt.`;
}

async function commitTransform(d: GizmoDelta) {
  if (!selCount || engineBusy) {
    // Nothing is sent, so the drag's preview has to come off again. It used
    // to stay in the shader (after Esc mid-drag, or a drag that ended while
    // a command ran) and moved whatever was selected next, on screen only.
    viewer.clearTransformPreview();
    refreshGizmo();
    if (selCount) status('not moved: something is still running, drag again when it is done');
    return;
  }
  setBusy(true, 'Applying …');
  try {
    const r = await engine('transform', {
      rotation: toEngineQuat(d.quat),
      translation: d.translation,
      scale: d.scale,
      pivot: d.pivot
    }, selectionPayload());
    const h = r.header;
    doc.setDead(Uint32Array.from(h.removed as number[]));
    if (h.addedCount > 0 && r.payload) {
      doc.appendGenerated(new Float32Array(r.payload), h.layout);
    }
    setHistory(h);              // now, not after the status line (see doRepair)
    viewer.clearTransformPreview();
    // the transformed copies are the selection now
    const first = h.addedFirstIndex as number;
    forgetStroke();
    setVolume(null);            // the splats moved; the painted volume did not
    selMask = new Uint8Array(doc.nTotal);
    for (let i = 0; i < h.addedCount; i++) selMask[first + i] = 1;
    selCount = h.addedCount;
    selDirty = true;
    rebuild();
    afterSelectionChanged();
    refreshGizmo();
    const i = h.info;
    status(`bewegt: ${fmt(i.count)} Splats · ${num(i.rotation, 1)}° · `
           + `Faktor ${num(i.scale)}`
           + (i.shRotated ? ' · SH mitgedreht' : '') + ` · ${num(h.seconds)} s`, 'ok');
  } catch (e: any) {
    viewer.clearTransformPreview();
    status(String(e.message ?? e), 'err');
  } finally {
    setBusy(false);
  }
}

/* -------------------------------------------------------------- klonstempel */

/**
 * The world point under a pixel, taken from the same depth map the brush uses.
 *
 * Reading the GPU's depth buffer would be exact but means waiting for the card
 * mid-gesture; this map is already built for the brush and is accurate to a
 * few pixels, which is well inside what a hand can aim.
 */
function pickWorld(x: number, y: number): [number, number, number] | null {
  const p = ensureProjection();
  if (!p || !grid) return null;
  const CELL = 6;
  const cx = (x / CELL) | 0, cy = (y / CELL) | 0;
  let best = Infinity;
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      const ux = cx + dx, uy = cy + dy;
      if (ux < 0 || ux >= grid.cw || uy < 0 || uy >= grid.ch) continue;
      const v = grid.surf[uy * grid.cw + ux];
      if (v < best) best = v;
    }
  }
  if (!isFinite(best)) return null;
  return viewer.unproject(x, y, best);
}

/** Canvas pixel of a world point, or null behind the camera. */
function toPixel(p: [number, number, number]) {
  const s = viewer.worldToViewport(...p);
  return s.onScreen ? { x: s.x, y: s.y } : null;
}

/**
 * The crosshair: where the stamp takes from. A ring the size of the brush
 * around it shows how much it takes, like Photoshop's clone source overlay.
 * While the brush hovers, the brush itself shows what lies under the
 * crosshair (ui/clonepreview.ts).
 */
function showCloneSource(at?: { x: number; y: number } | null) {
  const el = $<HTMLDivElement>('clone-src');
  const p = at !== undefined ? at : (cloneSource ? toPixel(cloneSource) : null);
  if (tool !== 'clone' || !p) {
    el.classList.remove('on');
    clonePreview?.hover(null, null, 0, 0);
    ring.classList.remove('peek');
    return;
  }
  el.style.setProperty('left', `${p.x}px`);
  el.style.setProperty('top', `${p.y}px`);
  el.style.setProperty('--d', `${brushRadius() * 2}px`);
  el.classList.add('on');
  const peek = !clonePainting && !engineBusy && !!cloneCursor && !!cloneOrigin;
  clonePreview?.hover(peek ? cloneCursor : null, p, brushRadius(), cloneFeather());
  ring.classList.toggle('peek', peek);
}

/** The soft edge as a fraction of the radius, as the engine gets it. */
function cloneFeather() {
  return Math.max(0.05, 1 - Number($<HTMLInputElement>('clone-hard').value));
}

/** Set the source, as Alt + click or Ctrl + click does. */
function setCloneSourceAt(w: [number, number, number]) {
  cloneSource = w;
  cloneOrigin = w;
  cloneAlignedOffset = null;
  showCloneSource();
  $('clone-state').textContent = 'Source set. Now paint over the spot to remove.';
  status('source set');
}

const dist3 = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/**
 * The surface point closest to `w`: the visible splat, opaque enough to be
 * surface, nearest to it within `maxD`. Null when there is none -- `w` is
 * off the object.
 */
function nearestSurface(w: [number, number, number], maxD: number): [number, number, number] | null {
  const c = doc.centres, a = doc.alpha, vis = visibleMask();
  const bar = Math.min(doc.surfaceAlpha, 0.2);
  let best = maxD * maxD, bi = -1;
  for (let i = 0; i < doc.nTotal; i++) {
    if (!vis[i] || a[i] < bar) continue;
    const dx = c[i * 3] - w[0], dy = c[i * 3 + 1] - w[1], dz = c[i * 3 + 2] - w[2];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < best) { best = d2; bi = i; }
  }
  return bi < 0 ? null : [c[bi * 3], c[bi * 3 + 1], c[bi * 3 + 2]];
}

/**
 * `w` put back on the surface the camera sees: what lies under its pixel
 * if that is near it, else the nearest surface point if that one is in
 * view. Null when `w` is off the object or hidden from this view.
 */
let nearestMemo: { w: number[]; r: number; t: number; s: [number, number, number] | null } | null = null;
function onVisibleSurface(w: [number, number, number], r: number, hover = false): [number, number, number] | null {
  const px = toPixel(w);
  const hit = px ? pickWorld(px.x, px.y) : null;
  if (hit && dist3(hit, w) <= 1.5 * r) return hit;
  // a full search of the splats; while hovering, at most ten times a second
  const now = performance.now();
  const m = nearestMemo;
  const s = hover && m && now - m.t < 100 && dist3(m.w, w) < 0.5 * r
    ? m.s : nearestSurface(w, 3 * r);
  nearestMemo = { w: w.slice(), r, t: now, s };
  if (!s) return null;
  const sp = toPixel(s);
  const seen = sp ? pickWorld(sp.x, sp.y) : null;
  return seen && dist3(seen, s) <= 1.5 * r ? seen : null;
}

/**
 * Where the next stroke would take from, for a brush at `target`.
 *
 * Aligned, the source keeps its offset from the brush -- but the offset is
 * a step across the surface, not a vector in the air. After an orbit the
 * world vector could end beside the object or behind it; the crosshair then
 * sat on the background or on whatever was in front, and the stroke's dabs
 * were skipped. So the offset point goes back onto the surface the camera
 * sees, and where there is none within reach (off the object, round the
 * back) the source starts again at the point that was set, and says so.
 */
function strokeSource(target: [number, number, number], hover = false):
    { p: [number, number, number]; note: string } | null {
  if (!cloneOrigin) return null;
  const aligned = $<HTMLInputElement>('clone-aligned').checked;
  if (!aligned || !cloneAlignedOffset) return { p: cloneOrigin, note: '' };
  const guess: [number, number, number] = [target[0] - cloneAlignedOffset[0],
    target[1] - cloneAlignedOffset[1], target[2] - cloneAlignedOffset[2]];
  const on = onVisibleSurface(guess, cloneRadiusAt(target), hover);
  if (on) return { p: on, note: '' };
  return { p: cloneOrigin,
           note: 'The source was beside or behind the object. It starts again at the set point.' };
}

/**
 * Start a painted stroke at canvas pixel (x, y) over world point `target`.
 *
 * The offset is held in screen pixels for the length of the stroke, as in
 * Photoshop: the crosshair keeps its distance from the brush, and each dab
 * takes from the surface that is under the crosshair. It used to be a world
 * offset that the engine carried along the surface from dab to dab, turned by
 * the change of the target's normal. Over relief smaller than the brush -- a
 * sign on a wall, a tread -- those normals swing by 20 to 40 degrees from one
 * dab to the next, and the source swung with them, a long way from where the
 * crosshair (which did not move at all) said it was.
 */
function beginCloneStroke(target: [number, number, number], x?: number, y?: number) {
  // A stroke started while the last one is still being stamped would take
  // its offset from a source the engine is about to move.
  if (engineBusy) return false;
  if (!cloneOrigin) {
    status('set a source with Alt + click first');
    return false;
  }
  const at = x !== undefined && y !== undefined ? { x, y } : toPixel(target);
  const from = strokeSource(target);
  const src = from?.p ?? null;
  const sp = src ? toPixel(src) : null;
  if (!at || !src || !sp) { status('the source is behind the camera'); return false; }
  if (from!.note) {
    // re-anchored: the next strokes keep this new distance
    cloneAlignedOffset = null;
    $('clone-state').textContent = from!.note;
    status(from!.note);
  }
  cloneScreenOff = { x: at.x - sp.x, y: at.y - sp.y };
  cloneOffset = [target[0] - src[0], target[1] - src[1], target[2] - src[2]];
  cloneCarry = null;
  cloneStartedAligned = !cloneAlignedOffset && $<HTMLInputElement>('clone-aligned').checked;
  // the side of the surface that faces the camera the stroke is painted from
  cloneView = viewer.viewDirection();
  { const e = viewer.camera.getPosition(); cloneEye = [e.x, e.y, e.z]; }
  cloneDabs = [];
  cloneRadii = [];
  clonePicked = [];
  cloneLast = null;
  clonePainting = true;
  showCloneSource();
  clonePreview?.begin();
  addCloneDab(target, at.x, at.y);
  return true;
}

/**
 * Add a dab if the brush has travelled far enough.
 *
 * Spacing is a quarter of the brush, which is what gives a continuous band
 * rather than a row of discs; closer than that and neighbouring dabs mostly
 * delete and re-lay the same splats for nothing.
 *
 * The source is what lies under the crosshair -- if that is the surface the
 * stroke has been taking from. Off the object, or onto something in front
 * of it or behind it (more than two brush radii from where the last offset
 * says the source is), the dab takes the offset point put back onto the
 * surface instead, and the engine carries it along from there.
 */
function addCloneDab(target: [number, number, number], x: number, y: number) {
  if (!cloneOffset || !cloneScreenOff) return;
  const sx = x - cloneScreenOff.x, sy = y - cloneScreenOff.y;
  showCloneSource({ x: sx, y: sy });
  const r = cloneRadiusAt(target);
  if (cloneLast) {
    const d = dist3(target, cloneLast);
    if (d < r * 0.25) return;
  }
  const off = cloneCarry ?? cloneOffset;
  const expect: [number, number, number] = [target[0] - off[0],
    target[1] - off[1], target[2] - off[2]];
  let hit = pickWorld(sx, sy);
  if (hit && dist3(hit, expect) > 2 * r) hit = null;
  const src: [number, number, number] = hit ?? nearestSurface(expect, 2 * r) ?? expect;
  cloneLast = target;
  cloneDabs.push([src, target]);
  clonePicked.push(!!hit);
  cloneRadii.push(r);
  cloneSource = src;
  // where the source really is carries on to the next dab
  if (hit) cloneCarry = [target[0] - hit[0], target[1] - hit[1], target[2] - hit[2]];
  clonePreview?.dab({ x, y }, { x: sx, y: sy }, brushRadius(), cloneFeather());
  $('clone-state').textContent = `${cloneDabs.length} dabs …`;
}

function cloneRadiusAt(p: [number, number, number]) {
  return brushRadius() * viewer.worldPerPixelAt(viewer.worldToViewport(...p).depth);
}

/**
 * The status line for a stamped stroke, from the engine's reply.
 *
 * `info.engine` says which stamp did it. When the picture stamp was tried
 * and gave up, the copying stamp did the stroke and `info.imageFallback`
 * holds the reason (engine: Document._image_stamp). As for the fill
 * (doRepair): the result looks unlike the stroke before it, and nothing
 * said why. With it the line is no "ok".
 */
function stampStatus(h: any): { text: string; kind: 'ok' | '' } {
  const i = h.info ?? {};
  const skipped = i.skipped
    ? ` · ${i.skipped} skipped (${i.skippedWhy ?? 'no surface'})` : '';
  const fallback = i.imageFallback
    ? ' · picture stamp could not do this stroke, the copying stamp took over'
      + ` (${oneLine(String(i.imageFallback))})`
    : '';
  return {
    text: `stamped: ${fmt(i.dabs ?? 1)} dabs, `
      + `+${fmt(h.addedCount)} / −${fmt(h.removed.length)}`
      + skipped + ` · ${num(h.seconds)} s` + fallback,
    kind: fallback ? '' : 'ok'
  };
}

/** Send the whole stroke as one edit, so one Ctrl+Z takes it all back. */
async function endCloneStroke() {
  clonePainting = false;
  clonePreview?.end();
  const dabs = cloneDabs;
  const radii = cloneRadii;
  const picked = clonePicked;
  cloneDabs = [];
  cloneRadii = [];
  clonePicked = [];
  if (!dabs.length || !cloneOrigin) { clonePreview?.cancel(); return; }
  const r = radii.length ? radii[radii.length - 1] : cloneRadiusAt(dabs[dabs.length - 1][1]);
  setBusy(true, dabs.length > 1 ? `Stamping ${dabs.length} dabs …` : 'Stamping …');
  // the provisional picture stays visible while the engine works
  busy.classList.add('busy--quiet');
  let ok = false;
  try {
    // What the crop preview hides is not there for the stamp either: neither
    // copied nor counted as surface.
    const hidden = cropActive && cropOut.length
      ? new Uint32Array(cropOut).buffer.slice(0) as ArrayBuffer : undefined;
    const rep = await engine('clone', {
      // Each dab keeps the radius the ring showed when it was laid, and the
      // camera direction says which side of the surface is the front --
      // which geometry alone cannot tell on a floor or a wall.
      dabs, radius: r, radii, picked, view: cloneView ?? viewer.viewDirection(),
      eye: cloneEye ?? undefined,
      volume: volumeArg(),          // the selection's painted volume, if any
      exclude: hidden ? 'payload' : undefined,
      params: {
        color_match: Number($<HTMLInputElement>('clone-match').value),
        feather: cloneFeather()
      }
    }, hidden);
    const h = rep.header;
    lastInfo = h.info;
    doc.setDead(Uint32Array.from(h.removed as number[]));
    if (h.addedCount > 0 && rep.payload) {
      doc.appendGenerated(new Float32Array(rep.payload), h.layout);
    }
    setHistory(h);              // now, not after the status line (see doRepair)
    rebuild();
    clearSelection();
    ok = true;
    // Aligned: the offset of the first stroke holds for every later one.
    if (cloneStartedAligned && cloneOffset) cloneAlignedOffset = cloneOffset;
    const line = stampStatus(h);
    status(line.text, line.kind);
    $('clone-state').textContent = $<HTMLInputElement>('clone-aligned').checked
      ? 'The next stroke continues at the same offset.'
      : 'The next stroke starts at the set source again.';
  } catch (e: any) {
    status(String(e.message ?? e), 'err');
    $('clone-state').textContent = 'Nothing stamped. Set the source again or paint on a surface.';
  } finally {
    if (ok) clonePreview?.done(); else clonePreview?.cancel();
    busy.classList.remove('busy--quiet');
    cloneStartedAligned = false;
    cloneScreenOff = null;
    cloneSource = cloneOrigin && !$<HTMLInputElement>('clone-aligned').checked
      ? cloneOrigin : cloneSource;
    setBusy(false);
    showCloneSource();
  }
}

/**
 * The crosshair while the brush only hovers: aligned, it rides along at the
 * offset the next stroke will use -- on the surface, see `strokeSource` --
 * otherwise it stays on the clicked source.
 */
function hoverCloneSource(x: number, y: number) {
  cloneCursor = { x, y };
  if (!cloneOrigin) { showCloneSource(null); return; }
  if (!cloneAlignedOffset || !$<HTMLInputElement>('clone-aligned').checked) {
    cloneSource = cloneOrigin;
    showCloneSource();
    return;
  }
  const w = pickWorld(x, y);
  if (!w) { showCloneSource(); return; }
  const s = strokeSource(w, true);
  if (s) { cloneSource = s.p; showCloneSource(); }
}

/* ------------------------------------------------------------------ clean up */

/**
 * The crop box, previewed by hiding what falls outside it.
 *
 * Nothing is deleted until it is committed: the preview writes the same
 * `editDel` flags the real deletion uses, so what you see while dragging the
 * sliders is exactly what you would get, and letting go of the idea costs
 * nothing.
 */
/**
 * What the crop sliders span, per axis.
 *
 * The full extent made them useless on real scans: a handful of floaters far
 * out set the ends, and on the hut 99 % of the splats sat in x 0.11-0.97 and
 * y 0.20-0.86 of the travel. So the middle of each slider covers the
 * 0.5-99.5 percentile padded by 10 %, and only the last few per cent at each
 * end run on out to the true extremes -- the outliers stay reachable, and 0
 * and 1 still mean "crop nothing". (Actual extent per axis, not the bounding
 * sphere: a shoe is long and thin.)
 *
 * Measured once per crop, not per slider move: measuring the live splats
 * every time meant a deletion mid-crop silently moved every boundary.
 */
interface CropAxis { min: number; lo: number; hi: number; max: number; p0: number; p1: number; }
let cropRange: CropAxis[] | null = null;
const CROP_TAIL = 0.04;        // slider travel given to the outliers at each end

function measureCropRange(): CropAxis[] | null {
  const c = doc.centres;
  const n = doc.nTotal;
  // A strided sample is plenty for percentiles and keeps this to a few ms
  // at a million splats; min and max come from every splat.
  const step = Math.max(1, Math.floor(n / 200000));
  const out: CropAxis[] = [];
  for (let a = 0; a < 3; a++) {
    let min = Infinity, max = -Infinity;
    const sample: number[] = [];
    for (let i = 0; i < n; i++) {
      if (!doc.alive[i]) continue;
      const v = c[i * 3 + a];
      if (v < min) min = v;
      if (v > max) max = v;
      if (i % step === 0) sample.push(v);
    }
    if (!isFinite(min)) return null;
    sample.sort((p, q) => p - q);
    const q = (f: number) => sample[Math.min(sample.length - 1, Math.floor(f * sample.length))];
    const p0 = q(0.005), p1 = q(0.995);
    const pad = (p1 - p0) * 0.1;
    out.push({ min, lo: Math.max(min, p0 - pad), hi: Math.min(max, p1 + pad), max, p0, p1 });
  }
  return out;
}

/** Slider travel at each end that runs out to the outliers; none if there are none. */
function cropTails(r: CropAxis): [number, number] {
  return [r.lo > r.min ? CROP_TAIL : 0, r.hi < r.max ? CROP_TAIL : 0];
}

/** Slider position 0..1 to a coordinate on one axis. */
function cropCoord(r: CropAxis, t: number): number {
  const [t0, t1] = cropTails(r);
  if (t < t0) return r.min + (r.lo - r.min) * (t / t0);
  if (t > 1 - t1) return r.hi + (r.max - r.hi) * ((t - (1 - t1)) / t1);
  return r.lo + (r.hi - r.lo) * ((t - t0) / Math.max(1 - t0 - t1, 1e-9));
}

function cropBounds(): { lo: number[]; hi: number[] } | null {
  if (!viewer.info || !doc.nTotal) return null;
  if (!cropRange) cropRange = measureCropRange();
  const range = cropRange;
  if (!range) return null;
  const f = (id: string) => Number($<HTMLInputElement>(id).value);
  const g = ['x', 'y', 'z'].map((ax, a) => {
    const t0 = Math.min(f(`crop-${ax}0`), f(`crop-${ax}1`));
    const t1 = Math.max(f(`crop-${ax}0`), f(`crop-${ax}1`));
    // the very ends are open, so a slider at 0 or 1 never cuts anything
    return [t0 <= 0 ? -Infinity : cropCoord(range[a], t0),
            t1 >= 1 ? Infinity : cropCoord(range[a], t1)];
  });
  return { lo: g.map(v => v[0]), hi: g.map(v => v[1]) };
}

/** Mark what the crop preview hides in a deleted-flags stream starting at `offset`. */
function cropHidden(del: Uint32Array<ArrayBufferLike>, offset: number) {
  if (!cropActive) return del;
  for (let k = 0; k < cropOut.length; k++) {
    const i = cropOut[k] - offset;
    if (i >= 0 && i < del.length) del[i] = 1;
  }
  return del;
}

function cropOutside(): Uint32Array<ArrayBufferLike> {
  const b = cropBounds();
  if (!b) return new Uint32Array(0);
  const c = doc.centres;
  const out: number[] = [];
  for (let i = 0; i < doc.nTotal; i++) {
    if (!doc.alive[i]) continue;
    const x = c[i * 3], y = c[i * 3 + 1], z = c[i * 3 + 2];
    if (x < b.lo[0] || x > b.hi[0] || y < b.lo[1] || y > b.hi[1]
        || z < b.lo[2] || z > b.hi[2]) out.push(i);
  }
  return Uint32Array.from(out);
}

let cropOut: Uint32Array<ArrayBufferLike> = new Uint32Array(0);

/** All six sliders at their ends: nothing can be outside. */
function cropIsOpen() {
  for (const ax of ['x', 'y', 'z']) {
    if (Number($<HTMLInputElement>(`crop-${ax}0`).value) > 0
        || Number($<HTMLInputElement>(`crop-${ax}1`).value) < 1) return false;
  }
  return true;
}

function updateCrop() {
  if (!doc.nTotal) return;
  cropOut = cropIsOpen() ? new Uint32Array(0) : cropOutside();
  cropActive = cropOut.length > 0;
  // what the brush, lasso, box and stamp can reach changed with it
  visibleCache = null;
  projDirty = true;
  $('crop-count').textContent = cropActive
    ? `${fmt(cropOut.length)} outside` : 'off';
  $<HTMLButtonElement>('btn-crop').disabled = !cropActive;
  // preview by hiding, using the same flags a real deletion writes
  applyStates();
}

function resetCrop() {
  for (const ax of ['x', 'y', 'z']) {
    $<HTMLInputElement>(`crop-${ax}0`).value = '0';
    $<HTMLInputElement>(`crop-${ax}1`).value = '1';
  }
  cropOut = new Uint32Array(0);
  cropActive = false;
  cropRange = null;
  visibleCache = null;
  projDirty = true;
  $('crop-count').textContent = 'off';
  $<HTMLButtonElement>('btn-crop').disabled = true;
  applyStates();
}

async function doCrop() {
  if (!cropOut.length || engineBusy) return;
  setBusy(true, 'Cropping …');
  try {
    const a = new Uint32Array(cropOut);
    const r = await engine('delete', {}, a.buffer.slice(0) as ArrayBuffer);
    doc.setDead(Uint32Array.from(r.header.removed as number[]));
    setHistory(r.header);       // now, not after the rebuild (see doRepair)
    resetCrop();
    rebuild();
    clearSelection();
    status(`cropped: ${fmt(r.header.removed.length)} splats removed`, 'ok');
  } catch (e: any) {
    status(String(e.message ?? e), 'err');
  } finally {
    setBusy(false);
  }
}

/**
 * Live splats whose position is no number (NaN or infinite): what the
 * engine's `drop_nonfinite` removes. Counted here, from the same positions,
 * for the menu entry; kept until an edit or another file changes what is
 * alive (`alive` is a new array per file, `editSerial` moves with every edit).
 */
let invalidMemo: { alive: unknown; serial: number; n: number } | null = null;
function invalidCount(): number {
  const a = doc.alive, c = doc.centres;
  if (invalidMemo && invalidMemo.alive === a && invalidMemo.serial === editSerial) return invalidMemo.n;
  let n = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] && !(isFinite(c[i * 3]) && isFinite(c[i * 3 + 1]) && isFinite(c[i * 3 + 2]))) n++;
  }
  invalidMemo = { alive: a, serial: editSerial, n };
  return n;
}

/**
 * Delete the splats without a valid position. They cannot be seen or
 * painted, so there is no selecting them: the engine finds them itself and
 * its reply is a delete's, applied like one (doCrop) and one step to undo.
 */
async function doDropInvalid() {
  if (!doc.nTotal || engineBusy) return;
  setBusy(true, 'Removing invalid splats …');
  try {
    const r = await engine('drop_nonfinite');
    const removed = r.header.removed as number[];
    // none: the engine made no step, so the history gets none either
    if (!removed.length) { status('no splats without a valid position found'); return; }
    doc.setDead(Uint32Array.from(removed));
    setHistory(r.header);       // now, not after the rebuild (see doRepair)
    rebuild();
    clearSelection();
    status(`${fmt(removed.length)} ${removed.length === 1 ? 'splat' : 'splats'} `
           + 'without a valid position removed', 'ok');
  } catch (e: any) {
    status(String(e.message ?? e), 'err');
  } finally {
    setBusy(false);
  }
}

/** Ask the engine which splats look like debris, and select them. */
async function doFloaters() {
  if (engineBusy) return;
  setBusy(true, 'Finding floaters …');
  try {
    const thr = Number($<HTMLInputElement>('floater-thr').value);
    // The haze test splits its cameras into above and below by the up axis,
    // so it has to be the one the view uses -- it was fixed at -y.
    const r = await engine('floaters', { threshold: thr, up: UP_VECTOR[viewer.upAxis] });
    const idx = Uint32Array.from(r.header.indices as number[]);
    selectMode = 'replace';
    forgetStroke();
    selView = null;               // not made from a view
    // not painted: the volume no longer is the whole story, or none of it
    setVolume(selectMode === 'replace' || !selVolume ? null : { ...selVolume, exact: false });
    commitHits(idx);
    $('floater-count').textContent = `${fmt(idx.length)} splats`;
    // Haze the engine left in place: without it, a surface no photo saw
    // would show (the ground under a hut, seen from below).
    const kept = Number(r.header.kept ?? 0);
    status(`${fmt(idx.length)} floaters selected `
           + `(threshold ${num(thr)}, ${num(r.header.seconds)} s)`
           + (kept ? ` · ${fmt(kept)} haze splats stay, otherwise unscanned areas would show beneath` : ''),
           'ok');
  } catch (e: any) {
    status(String(e.message ?? e), 'err');
  } finally {
    setBusy(false);
  }
}

function invertSelection() {
  if (!doc.nTotal) return;
  if (selMask.length !== doc.nTotal) selMask = new Uint8Array(doc.nTotal);
  forgetStroke();
  selView = null;
  setVolume(null);              // the inverse of a volume is not one we keep
  selCount = 0;
  // what is on screen, like "Select All" and the brushes: with `alive`
  // the inverse took in the splats the crop preview hides
  const vis = visibleMask();
  for (let i = 0; i < selMask.length; i++) {
    const v = vis[i] && !selMask[i] ? 1 : 0;
    selMask[i] = v;
    selCount += v;
  }
  selDirty = true;
  afterSelectionChanged();
}

/* --------------------------------------------------------------- interaction */

const TOOL_HINT: Record<Tool, string> = {
  orbit: 'Drag to orbit around the point under the pointer. Double-click centres it.',
  brush: 'Paint over the spot to remove. Shift adds, Ctrl subtracts.',
  lasso: 'Draw a line around the area. It closes on release.',
  box: 'Drag out a rectangle. Shift adds, Ctrl subtracts.',
  clone: 'Alt + click sets the source, then paint. The cross shows where the stamp samples.',
  xform: 'Drag the handles. Only the selection moves.'
};

/** The tool's hint; lasso and box say when they take the back too. */
function toolHint(t: Tool): string {
  if ((t === 'lasso' || t === 'box') && selDepth === 'through') {
    return 'Through: you select everything inside the outline, the back too. Alt+Z switches back.';
  }
  return TOOL_HINT[t];
}

function setTool(next: Tool) {
  if (painting) {
    // A tool key in the middle of a stroke. The stroke ends here, with the
    // tool that began it. Left running, the brush's live tint stayed on the
    // splats with nothing selected under it, and B during a lasso drag made
    // the button's release finish "a brush stroke": the one before, again.
    painting = false;
    stroke = [];
    strokeStart = null;
    strokeStartVolume = null;
    volumeOutline?.set(selVolume);      // the live outline back to the selection's
    applyStates();                      // and the tint: what is selected, nothing else
    // (without the GPU brush the stroke is already in the selection: a step then)
    history.noteSelection(selectionLabel());
  }
  tool = next;
  for (const b of document.querySelectorAll<HTMLButtonElement>('.tool[data-tool]')) {
    b.classList.toggle('active', b.dataset.tool === next);
  }
  canvas.classList.toggle('brushing', next === 'brush' || next === 'clone');
  canvas.classList.toggle('outlining', next === 'lasso' || next === 'box');
  ring.classList.toggle('on', next === 'brush' || next === 'clone');
  $('tool-hint').textContent = toolHint(next);
  // Lasso and box use the same depth reach as the brush, so the slider that
  // sets it has to be reachable from them too -- it was hidden, which left
  // the user with no way to see or change the value they were getting.
  $<HTMLElement>('brush-tile').hidden = next === 'orbit' || next === 'xform';
  $<HTMLElement>('brush-size-field').hidden = next !== 'brush' && next !== 'clone';
  $<HTMLElement>('clone-field').hidden = next !== 'clone';
  // The stamp copies a surface: how deep a selection reaches and how it is
  // combined with the last one do not apply to it.
  for (const id of ['brush-depth-field', 'opt-selmode', 'objsel-field']) {
    const el = document.getElementById(id);
    if (el) el.hidden = next === 'clone';
  }
  $<HTMLElement>('xform-field').hidden = next !== 'xform';
  $<HTMLElement>('depth-mode-field').hidden =
    next === 'orbit' || next === 'brush' || next === 'clone';
  refreshDepthLabel();
  showCloneSource();
  lassoPts = [];
  boxStart = boxNow = null;
  drawGuides();
  refreshGizmo();
  // "Orbit" used to clear the selection; switching to look at it should not
  // throw it away (Ctrl+D does that now). Its left button turns the view.
  viewer.leftNavigates = next === 'orbit';
  shell?.onTool(next);
}

function brushRadius() { return Number($<HTMLInputElement>('brush-size').value) / 2; }

function moveRing(e: PointerEvent) {
  const r = canvas.getBoundingClientRect();
  const d = brushRadius() * 2;
  ring.style.setProperty('left', `${e.clientX - r.left}px`);
  ring.style.setProperty('top', `${e.clientY - r.top}px`);
  ring.style.setProperty('width', `${d}px`);
  ring.style.setProperty('height', `${d}px`);
}

/** Modifier keys pick replace / add / subtract, like every other editor. */
function modeFrom(e: PointerEvent | MouseEvent): SelectMode {
  if (e.shiftKey) return 'add';
  if (e.ctrlKey || e.metaKey) return 'subtract';
  return shell?.selMode ?? 'replace';      // the options bar's "Mode"
}

function localPoint(e: PointerEvent) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

canvas.addEventListener('pointerdown', (e) => {
  // Alt + click sets the stamp's source, as in Photoshop. Alt + drag still
  // turns the view (the viewer takes the drag), so only a press that comes
  // up where it went down counts -- see pointerup.
  cloneAltDown = tool === 'clone' && e.button === 0 && e.altKey && !e.shiftKey
    ? { x: e.clientX, y: e.clientY } : null;
  if (e.button !== 0 || e.altKey || !doc.nTotal) return;
  if (tool === 'xform') return;     // the gizmo handles its own dragging
  if (tool === 'clone') {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const w = pickWorld(x, y);
    if (!w) { status('no surface there'); return; }
    if (e.ctrlKey || e.metaKey) setCloneSourceAt(w);
    else if (beginCloneStroke(w, x, y)) canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
    return;
  }
  if (tool !== 'brush' && tool !== 'lasso' && tool !== 'box') return;
  selectMode = modeFrom(e);
  painting = true;
  canvas.setPointerCapture(e.pointerId);
  if (tool === 'brush') {
    stroke = [];
    // what was selected before this stroke: the depth slider redoes the
    // stroke on top of it, not on top of the stroke's own first answer
    strokeStart = selMask.slice();
    strokeStartVolume = selVolume;
    addSample(e);
  } else if (tool === 'lasso') {
    lassoPts = [localPoint(e)];
  } else {
    boxStart = boxNow = localPoint(e);
  }
  drawGuides();
  e.preventDefault();
});

/**
 * Record a stroke sample, but only if the pointer actually went somewhere.
 *
 * Chromium delivers one pointermove per frame, so an unthinned stroke piled
 * up 60-144 samples a second whether the cursor moved a pixel or a hundred.
 * Every consumer walks the whole stroke, so the cost grew with how long the
 * user held the button down -- which is what the stalling was. A quarter of
 * the brush radius is finer than the brush can resolve anyway.
 */
function addSample(e: PointerEvent) {
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top, rad = brushRadius();
  const last = stroke[stroke.length - 1];
  if (last && Math.hypot(x - last.x, y - last.y) < Math.max(2, rad * 0.25)) return;
  stroke.push({ x, y, r: rad });
}

canvas.addEventListener('pointermove', (e) => {
  if (tool === 'brush' || tool === 'clone') moveRing(e);
  if (clonePainting) {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const w = pickWorld(x, y);
    if (w) addCloneDab(w, x, y);
    return;
  }
  if (tool === 'clone' && !viewer.isNavigating && doc.nTotal && !engineBusy) {
    const r = canvas.getBoundingClientRect();
    hoverCloneSource(e.clientX - r.left, e.clientY - r.top);
  }
  if (!painting) return;
  if (tool === 'brush') {
    addSample(e);
    scheduleSelect();
    return;
  }
  const p = localPoint(e);
  if (tool === 'lasso') {
    const last = lassoPts[lassoPts.length - 1];
    // thin the path: a point every few pixels is plenty and keeps the fill fast
    if (!last || Math.abs(p.x - last.x) + Math.abs(p.y - last.y) > 3) lassoPts.push(p);
  } else {
    boxNow = p;
  }
  drawGuides();
});

const endPaint = (e: PointerEvent) => {
  if (cloneAltDown && tool === 'clone' && doc.nTotal) {
    const moved = Math.hypot(e.clientX - cloneAltDown.x, e.clientY - cloneAltDown.y);
    cloneAltDown = null;
    if (moved < 4 && e.type === 'pointerup') {
      const r = canvas.getBoundingClientRect();
      const w = pickWorld(e.clientX - r.left, e.clientY - r.top);
      if (w) setCloneSourceAt(w); else status('no surface there');
    }
    return;
  }
  cloneAltDown = null;
  if (clonePainting) {
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    endCloneStroke();
    return;
  }
  if (!painting) return;
  painting = false;
  try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
  if (tool === 'brush') {                       // exact, once, stroke finished
    recomputeSelection(strokeStart ?? undefined);
    strokeStart = null;
  } else if (tool === 'lasso') commitLasso();
  else if (tool === 'box') commitBox();
};
canvas.addEventListener('pointerup', endPaint);
// the stamp's source preview follows the brush; off the canvas there is none
canvas.addEventListener('pointerleave', () => {
  if (cloneCursor) { cloneCursor = null; showCloneSource(); }
});
canvas.addEventListener('pointercancel', endPaint);

// Reselecting on every mouse move at 550k splats would fight the renderer for
// the main thread; once per frame is responsive and leaves the viewport smooth.
let selectQueued = false;
function scheduleSelect() {
  if (selectQueued) return;
  selectQueued = true;
  // Only while the stroke is still down: on a slow frame the preview queued
  // by the last move ran after the button came up, over the exact result --
  // its live outline replaced the finished one.
  requestAnimationFrame(() => { selectQueued = false; if (painting) previewSelection(); });
}

/* -------------------------------------------------------------------- wiring */

function bindRange(id: string, out: string, digits = 2, suffix = '') {
  const input = $<HTMLInputElement>(id);
  const label = $<HTMLSpanElement>(out);
  const show = () => {
    label.textContent = suffix
      ? `${Math.round(Number(input.value))}${suffix}`
      : num(Number(input.value), digits);
  };
  input.addEventListener('input', show);
  show();
  return input;
}

bindRange('brush-size', 'brush-size-v', 0, ' px');

/**
 * The depth slider is a bare factor, and what it means differs by tool: the
 * brush reaches depth/2 brush radii behind the surface, lasso and box 6 px per
 * step. Say it in pixels at the surface, which is what the user can see.
 */
function refreshDepthLabel() {
  const depth = Number($<HTMLInputElement>('brush-depth').value);
  const px = tool === 'brush' || tool === 'clone'
    ? depth * 0.5 * brushRadius() : 6 * depth;
  $('brush-depth-v').textContent = `${num(depth, 1)} · ${Math.round(px)} px`;
}
$<HTMLInputElement>('brush-depth').addEventListener('input', refreshDepthLabel);
$<HTMLInputElement>('brush-size').addEventListener('input', refreshDepthLabel);
refreshDepthLabel();
bindRange('density', 'density-v');
bindRange('roughness', 'roughness-v');
bindRange('colorsmooth', 'colorsmooth-v');
bindRange('clone-match', 'clone-match-v');
bindRange('clone-hard', 'clone-hard-v');
// the source ring follows the brush size
$<HTMLInputElement>('brush-size').addEventListener('input', () => showCloneSource());
{
  // Left of zero is "auto": the engine measures the share itself (detail < 0)
  const input = $<HTMLInputElement>('deep-detail');
  const label = $<HTMLSpanElement>('deep-detail-v');
  const show = () => {
    const v = Number(input.value);
    label.textContent = v < 0 ? 'auto' : num(v, 2);
  };
  input.addEventListener('input', show);
  show();
}
// The shadow slider means "how much of the shadow" or "how much brighter",
// with its own value for each (see applyShadowMode).
$<HTMLInputElement>('shadow-strength').addEventListener('input', () => {
  shadowVals[shadowMode()] = Number($<HTMLInputElement>('shadow-strength').value);
  showShadowValue();
});
$('shadow-mode').addEventListener('change', applyShadowMode);
applyShadowMode();

// After a stroke the slider redoes that stroke, exactly, at the new depth.
// Only for the brush: lasso and box take the value for their next outline.
let restrokeQueued = false;
$<HTMLInputElement>('brush-depth').addEventListener('input', () => {
  if (tool !== 'brush' || painting || !strokeBase || restrokeQueued) return;
  restrokeQueued = true;
  requestAnimationFrame(() => {
    restrokeQueued = false;
    // one step for the whole drag, after the stroke's own (ui/history.ts)
    history.labelNext('Depth');
    history.mergeNext('depth');
    applyStroke();
  });
});

for (const b of document.querySelectorAll<HTMLButtonElement>('.tool[data-tool]')) {
  b.addEventListener('click', () => setTool(b.dataset.tool as 'orbit' | 'brush'));
}

const FILL_HINT: Record<string, string> = {
  lama: 'Renders the surface head-on with plenty of surroundings, LaMa fills the picture, '
    + 'and the result is projected back onto the splats. Local, about 6 s. '
    + 'Paint generously: everything unwanted belongs in the selection.',
  photoshop: 'The same, but your Photoshop fills the picture with Generative Fill '
    + '(Adobe cloud, costs credits, also on every Retry; without credits '
    + 'content-aware). About 15–20 s, and Photoshop opens while it runs.',
  patch: 'The old variant: only a tight crop around the spot, with grain '
    + 'from the copy. Weaker on large holes.'
};

function fillChoice(): string {
  return $<HTMLSelectElement>('deep-fill').value;
}

function refreshFill() {
  const f = fillChoice();
  $('deep-fill-hint').textContent = FILL_HINT[f] ?? '';
  // The grain slider only means something where grain is mixed in.
  $('deep-detail-field').hidden = depthMode !== 'deep' || f !== 'patch';
}

/**
 * Which fill closes a quick repair ("Fill" under Advanced): the picture
 * fill, or the copying fill that did it before -- kept to set the two side
 * by side. The engine takes it per call (`engine` in the repair header);
 * the menu starts at the picture fill with every launch.
 */
type FillEngine = 'image' | 'copy';
const QUICK_HINT: Record<FillEngine, string> = {
  image: 'Fills in a picture of the surface, in a few seconds.',
  copy: 'Copies splats from the surroundings into the spot, in a few seconds.'
};

function fillEngine(): FillEngine {
  return $<HTMLSelectElement>('fill-engine').value === 'copy' ? 'copy' : 'image';
}

/** "Fast Fill" says what it does with the fill that is picked. */
function refreshEngineHint() {
  const hint = QUICK_HINT[fillEngine()];
  $('depth-quick').title = hint.replace(/\.$/, '');
  // The four settings below belong to the copying fill; the picture fill
  // reads none of them. They moved nothing and nothing said so.
  $('fill-engine-hint').textContent = fillEngine() === 'image'
    ? 'Method, Density, Texture and Color Smoothing only apply here when the copying fill steps in.'
    : 'Method, Density, Texture and Color Smoothing adjust this fill.';
  // only over its own words: the line also says why "Deep Fill" cannot be had
  const el = $('depth-hint');
  if (depthMode === 'quick' && Object.values(QUICK_HINT).includes(el.textContent ?? '')) {
    el.textContent = hint;
  }
}
$('fill-engine').addEventListener('change', refreshEngineHint);

function setDepth(next: 'quick' | 'deep') {
  if (next === 'deep' && !caps.deep) return;
  depthMode = next;
  $('deep-fill-field').hidden = next !== 'deep';
  // the deep repair has its own fills ("Fill with")
  $('fill-engine-field').hidden = next === 'deep';
  for (const b of document.querySelectorAll<HTMLButtonElement>('.tool[data-depth]')) {
    b.classList.toggle('active', b.dataset.depth === next);
  }
  $('depth-hint').textContent = next === 'deep'
    ? 'A 2D picture of the spot is filled and transferred onto the new Gaussians. '
      + (caps.gpu ? '' : 'Several minutes without a GPU.')
    : QUICK_HINT[fillEngine()];
  refreshFill();
}
$('deep-fill').addEventListener('change', refreshFill);
for (const b of document.querySelectorAll<HTMLButtonElement>('.tool[data-depth]')) {
  b.addEventListener('click', () => setDepth(b.dataset.depth as 'quick' | 'deep'));
}

const openHandler = async () => {
  // before the dialog, not after the file is picked (openFile checks again)
  if (engineBusy) {
    status('Open is available once the current job is done');
    return;
  }
  const p = await window.host.openSplatDialog();
  if (p) openFile(p);
};
$('btn-open').addEventListener('click', openHandler);
$('btn-open-2').addEventListener('click', openHandler);
$('btn-repair').addEventListener('click', () => doRepair());
// "Detect object" on or off: show what the repair will take now
$<HTMLInputElement>('obj-select').addEventListener('change', () => scheduleObjectPreview());
// The same stroke, filled again with another seed: undo, then repair with
// the stroke that was kept aside. Redo of the first fill is given up, which
// is what asking for a different one means.
$('btn-reroll').addEventListener('click', async () => {
  const last = lastFill;
  if (!last || engineBusy) return;
  if (!rerollReady()) { refreshReroll(); return; }
  // through the history, so the fill it replaces leaves the list too --
  // and only on if the fill really came off
  if (!(await history.undo())) return;
  await doRepair({ idx: last.idx, deep: last.deep, seed: last.seed + 1, view: last.view,
                   objects: last.objects });
});
$('btn-floaters').addEventListener('click', doFloaters);
$('btn-crop').addEventListener('click', doCrop);
$('btn-crop-reset').addEventListener('click', resetCrop);
$('btn-invert').addEventListener('click', invertSelection);
for (const ax of ['x', 'y', 'z']) {
  for (const end of ['0', '1']) {
    $(`crop-${ax}${end}`).addEventListener('input', updateCrop);
  }
}
$('btn-delete').addEventListener('click', doDelete);
$('btn-unshadow').addEventListener('click', doUnshadow);
$('btn-shadow-instead').addEventListener('click', () => {
  $<HTMLSelectElement>('shadow-mode').value = 'lift';
  applyShadowMode();
  void doUnshadow();
});
$('btn-solidify').addEventListener('click', doSolidify);
// one history for selections and edits (ui/history.ts), not the engine's alone
$('btn-undo').addEventListener('click', () => { void history.undo(); });
$('btn-redo').addEventListener('click', () => { void history.redo(); });
$('btn-export').addEventListener('click', doExport);
$('btn-up').addEventListener('click', cycleUpAxis);

for (const b of document.querySelectorAll<HTMLButtonElement>('.tool[data-xf]')) {
  b.addEventListener('click', () => {
    gizmoMode = b.dataset.xf as GizmoMode;
    for (const o of document.querySelectorAll<HTMLButtonElement>('.tool[data-xf]')) {
      o.classList.toggle('active', o === b);
    }
    refreshGizmo();
  });
}

/**
 * Lasso and box: only the visible surface (and the depth window behind it),
 * or everything inside the outline, the back of the object included. One
 * setter for the buttons, the menu, Alt+Z and the autotest, so what the
 * buttons show is what the next outline does.
 */
function setReach(r: DepthMode) {
  selDepth = r;
  for (const o of document.querySelectorAll<HTMLButtonElement>('.tool[data-reach]')) {
    o.classList.toggle('active', o.dataset.reach === r);
    o.setAttribute('aria-pressed', String(o.dataset.reach === r));
  }
  if (tool === 'lasso' || tool === 'box') $('tool-hint').textContent = toolHint(tool);
}
for (const b of document.querySelectorAll<HTMLButtonElement>('.tool[data-reach]')) {
  b.addEventListener('click', () => setReach(b.dataset.reach as DepthMode));
}

// Keys live in one place, the shell's command list (ui/shell.ts). This
// handler used to sit behind it and see whatever the shell let through --
// a key pressed while a dropdown had focus, after picking "Lighten" or a
// fill. Its Ctrl+Z went straight to the engine's undo, past the history:
// the selection never came back, and an edit was undone that the history
// still listed.

// A window `resize` fires for the window, not for the viewport. Going
// fullscreen does fire it, but the sidebar's own media-query breakpoints
// change the canvas width without one, and nothing noticed. Watch the box
// the canvas actually lives in.
new ResizeObserver(() => {
  viewer.resize(canvas);
  projDirty = true;
  drawGuides();
}).observe(canvas.parentElement ?? canvas);

window.addEventListener('resize', () => {
  viewer.resize(canvas);
  projDirty = true;
  drawGuides();
});

// The panel as the starting tool wants it. The markup shows the brush tile,
// which belongs to the brush, lasso and box, so "Orbit" opened with a depth
// slider that did nothing until another tool had been picked once.
setTool(tool);

/* ------------------------------------------------------------------- shell */

// What the camera picks and frames: the splats on screen, the surface the
// brush would find (opacity threshold), deleted and cropped ones left out.
viewer.navData = {
  centres: () => doc.centres,
  visible: () => doc.nTotal ? visibleMask() : null,
  alpha: () => doc.nTotal ? doc.alpha : null,
  minAlpha: () => doc.surfaceAlpha
};
// Menu bar, shortcuts, options bar, view cube, history panel (ui/shell.ts).
shell = new Shell({
  viewer, history,
  hasDoc: () => doc.nTotal > 0,
  selCount: () => selCount,
  tool: () => tool,
  setTool: (t) => setTool(t as Tool),
  selectAll,
  deselect: () => { if (selCount) clearSelection(); },
  frameSelection,
  openPath: (p) => { void openFile(p); },
  brushStep: (dir) => {
    const el = $<HTMLInputElement>('brush-size');
    const v = Number(el.value);
    el.value = String(Math.round(Math.max(8, Math.min(260, v * (dir > 0 ? 1.15 : 1 / 1.15)))));
    el.dispatchEvent(new Event('input'));
  },
  status,
  busy: () => engineBusy,
  unexported: hasUnexportedEdits,
  invalidCount,
  dropInvalid: () => { void doDropInvalid(); },
  through: () => selDepth === 'through',
  setThrough: (on) => setReach(on ? 'through' : 'surface')
});
shell.onTool(tool);
// The depth slider re-runs the last stroke on every frame it moves: one
// step for the drag, folded where the restroke runs (history.mergeNext).

/* ---------------------------------------------------------------- bootstrap */

(async () => {
  try {
    let backend: string;
    try {
      backend = await viewer.init(canvas);
    } catch (e: any) {
      // PlayCanvas reports this in English and in its own terms
      console.error('Graphics:', e);
      throw new Error('Graphics could not start, neither with WebGPU nor with WebGL 2. '
        + 'Without them Splat Repair cannot display anything. An up-to-date graphics driver usually helps.');
    }
    $('renderer-badge').textContent = backend;
    gizmo = new SelectionGizmo(viewer.app, viewer.camera, () => viewer.upRotation());
    clonePreview = new ClonePreview(canvas, canvas.parentElement!, ring, viewer.app as any);
    gizmo.onPreview = (d) => viewer.setTransformPreview(
      d.quat, d.translation, d.pivot, d.scale);
    gizmo.onCommit = (d) => { commitTransform(d); };
    viewer.onRender(() => {
      // Selection is drawn by the splat shader now, so a camera move needs no
      // repaint here -- only the cached projection used for the next stroke.
      // Not just while dragging: the wheel zooms without a drag, and the
      // cached projection would stay at the old zoom until the next one.
      if (viewer.consumeCameraMoved()) { projDirty = true; showCloneSource(); }
    });
    // the painted volume's outline, redrawn when the camera or the volume move
    volumeOutline = new VolumeOutline(canvas, guides, () => (doc.nTotal ? viewer.navData : null));
    volumeOutline.set(selVolume);
    viewer.addFrameListener(() => volumeOutline?.frame(viewer.viewProjection()));
    const r = await engine('ping');
    // The engine has answered: "ready" from here on, not from the first
    // paint (the markup says "starting …"). Not over a file's load report.
    if (!filePath && !engineBusy) status(`ready · Python ${r.header.python}`);
    // The engine probes the GPU on a thread of its own; until that is done
    // "capabilities" answers {pending}. Ask again in the background, so
    // opening a file never waits for torch to load.
    try {
      caps = (await engine('capabilities')).header;
      for (let i = 0; caps.pending && i < 300; i++) {
        $('depth-hint').textContent = 'Deep Fill: checking GPU …';
        await new Promise((res) => setTimeout(res, 1000));
        caps = (await engine('capabilities')).header;
      }
    } catch { caps = { deep: false }; }
    if (caps.pending) caps = { deep: false, gpuDetail: 'GPU check is taking too long' };
    const deepBtn = $<HTMLButtonElement>('depth-deep');
    deepBtn.disabled = !caps.deep;
    if (!caps.deep) {
      deepBtn.title = caps.inpainterDetail ?? caps.gpuDetail ?? 'unavailable';
      $('depth-hint').textContent =
        'Deep Fill unavailable: ' + (caps.inpainterDetail ?? caps.gpuDetail ?? '');
    } else {
      // "checking GPU …" off the line again, now that it has been: it
      // stayed there until the first click on "Fast Fill" or "Deep Fill".
      setDepth(depthMode);
    }
    // The wide view needs the GPU (and LaMa, or Photoshop). Offer only what
    // works here; where neither does, the tight view is the default.
    const fillSel = $<HTMLSelectElement>('deep-fill');
    const psOpt = fillSel.querySelector<HTMLOptionElement>('option[value="photoshop"]');
    const lamaOpt = fillSel.querySelector<HTMLOptionElement>('option[value="lama"]');
    if (psOpt && !caps.photoshop) {
      psOpt.disabled = true;
      psOpt.textContent = `Photoshop (${caps.photoshopDetail ?? 'not found'})`;
    }
    if (lamaOpt && !caps.viewFill) {
      lamaOpt.disabled = true;
      lamaOpt.textContent = `LaMa, wide view (${caps.viewFillDetail ?? 'unavailable'})`;
      // The engine's own default where LaMa cannot run. Never Photoshop
      // unasked: it costs credits and opens a window.
      fillSel.value = 'patch';
    }
    if (!caps.gpu) {
      // The picture fill renders on the GPU. Without one the copying fill is
      // what runs, so "Fill" offers that, as "Fill with" does above.
      const engineSel = $<HTMLSelectElement>('fill-engine');
      const imageOpt = engineSel.querySelector<HTMLOptionElement>('option[value="image"]');
      if (imageOpt) {
        imageOpt.disabled = true;
        imageOpt.textContent = 'In picture (needs a GPU)';
      }
      engineSel.value = 'copy';
      refreshEngineHint();
      // Without a GPU only "cut off from everything" is measured, and that
      // is yes or no: every score is 0 or 0.95.
      $('floater-hint').textContent = 'Splats cut off from everything. Without a GPU '
        + 'it is only yes or no: the threshold changes nothing then. '
        + 'Select first, then look, then delete.';
    }
    const gpuNote = caps.gpu ? caps.gpuDetail.split(',')[0] : 'no GPU';
    // Only while nothing is open: a file opened before the engine answered
    // (the castle loads in under a second) had its load report replaced by
    // "ready", and the autotest's GPU/CPU picture compare tripped over it.
    if (!filePath) {
      status(`ready · Python ${r.header.python} · ${gpuNote}`
             + (caps.deep ? ` · ${caps.inpainter}` : ''));
    }
  } catch (e: any) {
    // No engine (or no viewport): "Deep Fill" stays off, as the markup has it,
    // and says why. It used to look ready and do nothing.
    $<HTMLButtonElement>('depth-deep').disabled = true;
    $('depth-deep').title = 'Unavailable. The status bar below says what was missing at launch.';
    status(String(e.message ?? e), 'err');
  }
})();

// The engine held the document, and it is not started a second time: say
// what that means. "Engine exited (3221225620)" was all it said, and the
// command it died in replaced even that with its own error a moment later.
window.host.engine.onExit((code) => {
  // It never answered: the first command is told why it did not start
  // (sidecar.ts, startFailure), and that sentence must stand.
  if (!engineUp) { status(`Engine exited (${code})`, 'err'); return; }
  engineGone = `The engine has exited${code === null || code === undefined ? '' : ` (code ${code})`}. `
    + 'Restart Splat Repair. Anything not exported is lost.';
  status(engineGone, 'err');
});
window.host.engine.onEvent((e) => {
  if (e.event === 'progress') setBusy(true, e.stage, e.frac ?? 0);
});

/*
 * Automation hook. The main process drives these to run the whole flow --
 * open, paint, repair, export -- without a human, so the app itself is covered
 * by a test and not just the engine underneath it.
 */
(window as any).__test = {
  async open(path: string) {
    await openFile(path);
    return { splats: doc.nTotal, alive: doc.nAlive, path: filePath };
  },
  /** Paint a disc at viewport-relative coordinates and select under it. */
  paint(cx: number, cy: number, radius: number, depth?: number) {
    return (window as any).__test.paintStroke([[cx, cy]], radius, depth);
  },
  /** Paint a stroke through viewport-relative points. */
  paintStroke(points: [number, number][], radius: number, depth?: number) {
    const r = canvas.getBoundingClientRect();
    if (depth !== undefined) $<HTMLInputElement>('brush-depth').value = String(depth);
    setTool('brush');
    stroke = points.map(([x, y]) => ({ x: x * r.width, y: y * r.height, r: radius }));
    recomputeSelection();
    return selCount;
  },
  /** Viewport size in CSS pixels, so a test can place a stroke in real units. */
  viewportSize() {
    const r = canvas.getBoundingClientRect();
    return { width: r.width, height: r.height };
  },
  /**
   * Paint through world-space points. Aiming a test stroke at coordinates in
   * the scene rather than at fractions of the window keeps it pointed at the
   * artifact no matter how the camera is framed or the window is sized.
   */
  /** Why a stroke selected what it did: the state the brush read (autotest). */
  paintDiag() {
    const vis = visibleMask();
    let nVis = 0, nOpaque = 0;
    for (let i = 0; i < vis.length; i++) {
      if (!vis[i]) continue;
      nVis++;
      if (doc.alpha[i] >= doc.surfaceAlpha) nOpaque++;
    }
    let finite = 0, under = 0;
    if (grid) {
      for (let k = 0; k < grid.surf.length; k++) if (isFinite(grid.surf[k])) finite++;
      for (const s of stroke) {
        const k = Math.floor(s.y / 6) * grid.cw + Math.floor(s.x / 6);
        if (k >= 0 && k < grid.surf.length && isFinite(grid.surf[k])) under++;
      }
    }
    const r = canvas.getBoundingClientRect();
    return {
      mode: selectMode, tool, painting, busy: engineBusy, projDirty,
      n: doc.nTotal, alive: doc.nAlive, visible: nVis, opaque: nOpaque, surfaceAlpha: doc.surfaceAlpha,
      crop: { active: cropActive, out: cropOut.length },
      grid: grid ? { cw: grid.cw, ch: grid.ch, finite, under } : null,
      projMeta: projMeta ? { w: projMeta.w, h: projMeta.h, same: projMeta.vp.every((v, i) => Math.abs(v - viewer.viewProjection()[i]) < 1e-6) } : null,
      stroke: stroke.length, caps: strokeCaps.length,
      rect: [r.width, r.height], buffer: [canvas.width, canvas.height],
      pose: viewer.pose(), animating: viewer.isAnimating, navigating: viewer.isNavigating
    };
  },
  paintWorld(points: [number, number, number][], radius: number, depth?: number) {
    const r = canvas.getBoundingClientRect();
    if (depth !== undefined) $<HTMLInputElement>('brush-depth').value = String(depth);
    setTool('brush');
    stroke = [];
    for (const [x, y, z] of points) {
      const p = viewer.worldToViewport(x, y, z);
      if (p.onScreen) stroke.push({ x: p.x, y: p.y, r: radius });
    }
    recomputeSelection();
    return {
      selected: selCount,
      points: stroke.map(s => [Math.round(s.x), Math.round(s.y)]),
      viewport: [Math.round(r.width), Math.round(r.height)]
    };
  },
  setDepth(m: 'quick' | 'deep') { setDepth(m); return depthMode; },
  /** Pick what fills a deep repair; returns what the menu now says. */
  setFill(f: string) {
    const sel = $<HTMLSelectElement>('deep-fill');
    const opt = sel.querySelector<HTMLOptionElement>(`option[value="${f}"]`);
    if (opt && !opt.disabled) sel.value = f;
    refreshFill();
    return sel.value;
  },
  /** Move the depth slider as a user would, after a stroke. */
  slideDepth(v: number) {
    const el = $<HTMLInputElement>('brush-depth');
    el.value = String(v);
    el.dispatchEvent(new Event('input'));
    return new Promise(r => requestAnimationFrame(() => requestAnimationFrame(
      () => r({ selected: selCount, depth: Number(el.value) }))));
  },
  /** The view the selection was made from, and the camera now. */
  views() { return { selView, now: viewer.viewDirection() }; },
  /** Selected splats the crop preview hides (must be none), and how many it hides. */
  cropCheck() {
    let hidden = 0;
    for (let k = 0; k < cropOut.length; k++) if (selMask[cropOut[k]]) hidden++;
    return { selectedHidden: hidden, outside: cropOut.length, active: cropActive,
             label: $('crop-count').textContent };
  },
  async caps() {
    // the GPU probe may still be running; wait for the final answer
    for (let i = 0; caps.pending && i < 300; i++) await new Promise((r) => setTimeout(r, 500));
    return caps;
  },
  async repair() {
    await doRepair();
    const i = lastFillInfo ?? {};
    return { alive: doc.nAlive, generated: doc.nGenerated,
             mode: depthMode, status: $('status-msg').textContent,
             fill: { engine: i.engine, views: i.passes ?? 1, seconds: i.seconds, picture: i.picture,
                     holePixels: i.holePixels, flank: i.flank, sideViews: i.sideViews } };
  },
  /**
   * "Detect object": switch it (when `on` is given), then wait for the
   * engine's answer for the current selection and return it with how many
   * object points the outline now holds.
   */
  async objectSelection(on?: boolean) {
    const el = $<HTMLInputElement>('obj-select');
    if (on !== undefined && el.checked !== on) {
      el.checked = on;
      scheduleObjectPreview();
    }
    const serial = objectSerial;
    for (let i = 0; i < 200 && !lastObject && serial === objectSerial && on !== false; i++) {
      await new Promise(r => setTimeout(r, 50));
    }
    return { on: el.checked, grow: objectGrow(), answered: !!lastObject,
             ...(lastObject ?? { mode: 'none', count: 0, object: 0, seconds: 0, info: {} }),
             outline: volumeOutline?.objectCount ?? 0, selected: selCount,
             label: $('sel-count').textContent };
  },
  /** Delete the selection, as "Delete". */
  async deleteSelection() {
    await doDelete();
    return { alive: doc.nAlive, status: $('status-msg').textContent };
  },
  /**
   * Undo back over the last edit, through the history (selection steps on
   * top go first), and leave nothing selected -- what the engine-only undo
   * did, which the older checks build on. historyUndo() is the user's Ctrl+Z.
   */
  async undo() {
    for (let k = 0; k < 64 && history.canUndo; k++) {
      const wasEdit = history.entries[history.index].kind === 'edit';
      if (!(await history.undo())) break;
      if (wasEdit) break;
    }
    clearSelection();
    return { alive: doc.nAlive, generated: doc.nGenerated };
  },
  async reroll() {
    const before = doc.nGenerated;
    $<HTMLButtonElement>('btn-reroll').click();
    // the click runs undo and repair in turn; wait for both to finish. 20 s
    // was the shoe's budget -- the hut's repair alone takes 16.
    for (let i = 0; i < 2400 && (engineBusy || doc.nGenerated === 0); i++) {
      await new Promise(r => setTimeout(r, 50));
    }
    return { before, generated: doc.nGenerated, status: $('status-msg').textContent };
  },
  /** Press the shadow button; `mode` / `value` set the menu and slider first. */
  async unshadow(mode?: 'auto' | 'lift', value?: number) {
    if (mode) { $<HTMLSelectElement>('shadow-mode').value = mode; applyShadowMode(); }
    if (value !== undefined) {
      const el = $<HTMLInputElement>('shadow-strength');
      el.value = String(value);
      el.dispatchEvent(new Event('input'));
    }
    lastInfo = null;
    await doUnshadow();
    return { alive: doc.nAlive, generated: doc.nGenerated, selected: selCount,
             status: $('status-msg').textContent, info: lastInfo,
             note: $<HTMLParagraphElement>('shadow-note').hidden ? null
               : $('shadow-note').textContent,
             offersLift: !$<HTMLDivElement>('shadow-instead').hidden };
  },
  /** Press "Lighten instead". */
  async shadowInstead() {
    lastInfo = null;
    $<HTMLButtonElement>('btn-shadow-instead').click();
    for (let i = 0; i < 400 && (engineBusy || !lastInfo); i++) {
      await new Promise(r => setTimeout(r, 50));
      if (!engineBusy && $('status-msg').className.includes('err')) break;
    }
    return { info: lastInfo, mode: shadowMode(), status: $('status-msg').textContent,
             note: $('shadow-note').textContent };
  },
  /**
   * Look at `target` along `dir` from `dist` world units: for aiming a test
   * at a spot on the scan. Returns how far the camera ended up off `dir`.
   */
  aim(target: [number, number, number], dir: [number, number, number], dist: number) {
    const v = viewer as any;
    viewer.setView();             // stops a running camera animation, perspective
    v.target.set(target[0], target[1], target[2]);
    const n = Math.hypot(...dir);
    const f = dir.map(x => x / n);
    let best: [number, number, number] = [1e9, 0, 0];
    // yaw/pitch for that forward; the Euler convention is checked by
    // reading the camera back rather than trusted
    for (const sy of [1, -1]) for (const sp of [1, -1]) for (const fy of [1, -1]) {
      const fl = [f[0], fy * f[1], fy * f[2]];
      const pitch = sp * Math.asin(Math.max(-1, Math.min(1, -fl[1]))) * 180 / Math.PI;
      const yaw = sy * Math.atan2(-fl[0], -fl[2]) * 180 / Math.PI;
      v.yaw = yaw; v.pitch = pitch; v.distance = dist;
      v.applyCamera();
      const g = viewer.viewDirection();
      const err = Math.hypot(g[0] - f[0], g[1] - f[1], g[2] - f[2]);
      if (err < best[0]) best = [err, yaw, pitch];
    }
    v.yaw = best[1]; v.pitch = best[2]; v.distance = dist;
    v.applyCamera();
    projDirty = true;
    return { err: best[0], worldPerPixel: viewer.worldPerPixelAt(dist) };
  },
  async solidify() {
    await doSolidify();
    return { alive: doc.nAlive, generated: doc.nGenerated, status: $('status-msg').textContent };
  },
  rebuildTiming() { return rebuild(); },
  forceTint() { if (viewer.splat) viewer.forceTintAll(viewer.splat); return true; },
  /** Mark bands of splats deleted/selected straight in the GPU stream. */
  probeState() {
    if (!viewer.splat) return { ok: false, why: 'no splat loaded' };
    if (!viewer.editState) return { ok: false, why: 'editState unavailable' };
    const n = doc.nBase;
    const del = new Uint32Array(n);
    const sel = new Uint32Array(n);
    const c = doc.centres;
    for (let i = 0; i < n; i++) {
      const x = c[i * 3];
      if (x > 2 && x < 8) del[i] = 1;
      else if (x > -4 && x <= 2) sel[i] = 1;
    }
    const a = viewer.writeStream(viewer.splat, 'editDel', del);
    const b = viewer.writeStream(viewer.splat, 'editSel', sel);
    return { ok: a && b,
             deleted: del.reduce((p, v) => p + v, 0),
             selected: sel.reduce((p, v) => p + v, 0) };
  },
  /**
   * Run the stroke through both paths and report whether they agree.
   * Two implementations of one rule drift silently; this is what catches it.
   */
  async transform(rot: [number, number, number, number],
                  trans: [number, number, number], scale = 1) {
    setTool('xform');
    const pivot = selectionPivot() ?? [0, 0, 0];
    await commitTransform({ quat: rot, translation: trans, pivot, scale });
    return { total: doc.nTotal, alive: doc.nAlive,
             selected: selCount, status: $('status-msg').textContent };
  },
  gizmoState() {
    return { active: !!gizmo?.active, mode: gizmoMode,
             pivot: selectionPivot(), tool };
  },
  setCloneSource(x: number, y: number, z: number) {
    setTool('clone');
    setCloneSourceAt([x, y, z]);
    return { source: cloneSource };
  },
  /** Set the source at a viewport-relative position, as Alt + click does. */
  setCloneSourceAtScreen(fx: number, fy: number) {
    setTool('clone');
    const r = canvas.getBoundingClientRect();
    projDirty = true;
    const w = pickWorld(fx * r.width, fy * r.height);
    if (w) setCloneSourceAt(w);
    return { source: w };
  },
  async stamp(x: number, y: number, z: number, radiusPx = 60) {
    return (window as any).__test.stampStroke([[x, y, z]], radiusPx);
  },
  /**
   * Paint a stamp stroke through world points, as a drag would: each point
   * goes to the surface under its pixel, as the pointer's would.
   */
  async stampStroke(points: Array<[number, number, number]>, radiusPx = 60, match?: number) {
    if (match !== undefined) $<HTMLInputElement>('clone-match').value = String(match);
    const size = $<HTMLInputElement>('brush-size');
    size.value = String(radiusPx * 2);
    size.dispatchEvent(new Event('input'));
    setTool('clone');
    projDirty = true;
    const px = points.map(p => toPixel(p));
    const hit = px.map(p => (p ? pickWorld(p.x, p.y) : null));
    if (!px[0] || !hit[0] || !beginCloneStroke(hit[0], px[0].x, px[0].y)) {
      return { status: $('status-msg').textContent };
    }
    for (let k = 1; k < points.length; k++) {
      if (px[k] && hit[k]) addCloneDab(hit[k]!, px[k]!.x, px[k]!.y);
    }
    const sent = cloneDabs.map(d => d.map(v => v.slice()));
    const picked = clonePicked.slice();
    const dabs = cloneDabs.length;
    // a frame or two, as a drag has: the provisional picture is drawn
    for (let k = 0; k < 3; k++) await new Promise(r => requestAnimationFrame(() => r(null)));
    const preview = clonePreview?.state() ?? null;
    lastInfo = null;
    await endCloneStroke();
    return { dabs, sent, picked, preview, info: lastInfo, total: doc.nTotal, alive: doc.nAlive,
             generated: doc.nGenerated, status: $('status-msg').textContent };
  },
  /** The stamp's status line for an engine reply, as endCloneStroke writes it. */
  stampStatusFor(h: Record<string, any>) { return stampStatus(h); },
  /**
   * "Beta Tools" (ui/shell.ts, setBeta): switch it when `on` is given --
   * for this run only, nothing is stored, the autotest shares the user's
   * profile -- and say what shows of the paused tools (shadow, "Solidify"):
   * their controls that have a box on the page, their menu entries, the ones
   * with a key, their lines in the help card, and the panel's heading.
   */
  beta(on?: boolean) {
    if (on !== undefined) shell!.setBeta(on, false);
    const theirs = /Shadow|Lighten|Solidif/;
    const shown = ['retouch-tile', 'shadow-mode', 'shadow-strength', 'btn-unshadow', 'btn-shadow-instead',
                   'solid-colour', 'btn-solidify']
      .filter(id => (document.getElementById(id)?.getClientRects().length ?? 0) > 0);
    const cmds = shell!.commands().filter(c => theirs.test(c.label));
    const helpWas = $('help').hidden;
    shell!.showHelp(true);
    const help = [...document.querySelectorAll('#help-body .keys__row')]
      .map(r => r.textContent ?? '').filter(t => theirs.test(t));
    if (helpWas) shell!.showHelp(false);
    const tile = $<HTMLDetailsElement>('retouch-tile');
    return { on: shell!.beta, panel: !tile.hidden, open: tile.open, shown,
             menu: cmds.map(c => `${c.menu}: ${c.label}`), keys: cmds.filter(c => c.bound || c.keys).map(c => c.label),
             help, heading: tile.querySelector('summary')?.textContent?.replace(/\s+/g, ' ').trim() ?? '',
             setting: shell!.commands().some(c => c.menu === 'Edit' && c.label === 'Beta Tools') };
  },
  /** The brush hovering at a viewport-relative position, as a mouse would. */
  async hoverClone(fx: number, fy: number) {
    const r = canvas.getBoundingClientRect();
    projDirty = true;
    hoverCloneSource(fx * r.width, fy * r.height);
    for (let k = 0; k < 3; k++) await new Promise(res => requestAnimationFrame(() => res(null)));
    return { preview: clonePreview?.state() ?? null, source: cloneSource,
             state: $('clone-state').textContent };
  },
  /** Look at `p` along `fwd` from `dist`, for probes that need a set view. */
  lookAt(p: [number, number, number], fwd: [number, number, number], dist: number) {
    const v = viewer as any;
    v.target.set(p[0], p[1], p[2]);
    const n = Math.hypot(...fwd);
    let best = -2, by = 0, bp = 0;
    for (let yaw = 0; yaw < 360; yaw += 1) {
      for (let pitch = -89; pitch <= 89; pitch += 1) {
        v.yaw = yaw; v.pitch = pitch; v.applyCamera();
        const f = viewer.viewDirection();
        const d = (f[0] * fwd[0] + f[1] * fwd[1] + f[2] * fwd[2]) / n;
        if (d > best) { best = d; by = yaw; bp = pitch; }
      }
    }
    v.yaw = by; v.pitch = bp; v.distance = dist; v.applyCamera();
    projDirty = true;
    return { yaw: by, pitch: bp, dot: best };
  },
  cloneState() {
    const el = $<HTMLDivElement>('clone-src');
    return { source: cloneSource, origin: cloneOrigin, aligned: cloneAlignedOffset,
             dabs: cloneDabs.length, painting: clonePainting,
             marker: el.classList.contains('on')
               ? [parseFloat(el.style.getPropertyValue('left')),
                  parseFloat(el.style.getPropertyValue('top'))] : null,
             state: $('clone-state').textContent };
  },
  /** World point under a viewport-relative position, and back. */
  pick(fx: number, fy: number) {
    const r = canvas.getBoundingClientRect();
    projDirty = true;
    return pickWorld(fx * r.width, fy * r.height);
  },
  toScreen(x: number, y: number, z: number) {
    const r = canvas.getBoundingClientRect();
    const p = viewer.worldToViewport(x, y, z);
    return { fx: p.x / r.width, fy: p.y / r.height, onScreen: p.onScreen };
  },
  /** Ids of the splats around a world point, alive or generated, for checks. */
  splatsNear(x: number, y: number, z: number, radius: number) {
    const c = doc.centres;
    let base = 0, gen = 0;
    for (let i = 0; i < doc.nTotal; i++) {
      if (!doc.alive[i]) continue;
      const dx = c[i * 3] - x, dy = c[i * 3 + 1] - y, dz = c[i * 3 + 2] - z;
      if (dx * dx + dy * dy + dz * dz > radius * radius) continue;
      if (i < doc.nBase) base++; else gen++;
    }
    return { base, generated: gen };
  },
  async floaters(threshold = 0.3) {
    $<HTMLInputElement>('floater-thr').value = String(threshold);
    await doFloaters();
    return { selected: selCount, status: $('status-msg').textContent };
  },
  crop(lo: [number, number, number], hi: [number, number, number]) {
    const ax = ['x', 'y', 'z'];
    for (let a = 0; a < 3; a++) {
      $<HTMLInputElement>(`crop-${ax[a]}0`).value = String(lo[a]);
      $<HTMLInputElement>(`crop-${ax[a]}1`).value = String(hi[a]);
    }
    updateCrop();
    return { outside: cropOut.length, active: cropActive };
  },
  async commitCrop() {
    await doCrop();
    return { alive: doc.nAlive, status: $('status-msg').textContent };
  },
  invert() { invertSelection(); return { selected: selCount }; },
  /** Drag a lasso through viewport-relative points. */
  lasso(points: [number, number][], mode: SelectMode = 'replace',
        reach: DepthMode = 'surface') {
    const r = canvas.getBoundingClientRect();
    setTool('lasso');
    selectMode = mode;
    setReach(reach);
    lassoPts = points.map(([x, y]) => ({ x: x * r.width, y: y * r.height }));
    const t0 = performance.now();
    commitLasso();
    return { selected: selCount, ms: +(performance.now() - t0).toFixed(2),
             mode, reach, surface: lastRegion };
  },
  /** Drag a box between two viewport-relative corners. */
  box(x0: number, y0: number, x1: number, y1: number,
      mode: SelectMode = 'replace', reach: DepthMode = 'surface') {
    const r = canvas.getBoundingClientRect();
    setTool('box');
    selectMode = mode;
    setReach(reach);
    boxStart = { x: x0 * r.width, y: y0 * r.height };
    boxNow = { x: x1 * r.width, y: y1 * r.height };
    const t0 = performance.now();
    commitBox();
    return { selected: selCount, ms: +(performance.now() - t0).toFixed(2),
             mode, reach, surface: lastRegion };
  },
  /** Build the stroke and show only the GPU's answer. */
  selectGpu(points: [number, number, number][], radius: number, depth = 1.5) {
    const r = canvas.getBoundingClientRect();
    $<HTMLInputElement>('brush-depth').value = String(depth);
    setTool('brush');
    stroke = points.map(([x, y, z]) => {
      const p = viewer.worldToViewport(x, y, z);
      return { x: p.x, y: p.y, r: radius };
    }).filter((_, i) => viewer.worldToViewport(...points[i]).onScreen);
    ensureProjection();
    // Wipe the stream first. Otherwise a GPU pass that quietly did nothing
    // leaves the previous CPU answer sitting there and the comparison passes
    // without having tested anything.
    selMask = new Uint8Array(doc.nTotal); selCount = 0; selDirty = true;
    if (viewer.splat) viewer.writeStream(viewer.splat, 'editSel', new Uint32Array(doc.nBase));
    if (viewer.generated) {
      viewer.writeStream(viewer.generated, 'editSel', new Uint32Array(doc.nGenerated));
    }
    const t0 = performance.now();
    previewSelection();
    return { ok: !!gpuSel?.ready, ms: +(performance.now() - t0).toFixed(2),
             capsules: lastCaps.length, viewport: [r.width | 0, r.height | 0],
             reason: gpuSel?.reason || undefined };
  },
  /** Same stroke, exact CPU answer, written into the same stream. */
  selectCpu() {
    const t0 = performance.now();
    recomputeSelection();
    return { selected: selCount, ms: +(performance.now() - t0).toFixed(2) };
  },
  /** Time the selection path, separating projection from the capsule test. */
  timeSelect(points: [number, number, number][], radius: number, runs = 10) {
    setTool('brush');
    stroke = [];
    for (const [x, y, z] of points) {
      const p = viewer.worldToViewport(x, y, z);
      if (p.onScreen) stroke.push({ x: p.x, y: p.y, r: radius });
    }
    let tProj = 0, tSel = 0, n = 0;
    for (let k = 0; k < runs; k++) {
      let t = performance.now();
      projDirty = true;
      ensureProjection();
      tProj += performance.now() - t;
      t = performance.now();
      const depth = Number($<HTMLInputElement>('brush-depth').value);
      const caps = strokeToCapsules(grid!, stroke,
        (x, y, d) => viewer.unproject(x, y, d),
        (d) => viewer.worldPerPixelAt(d));
      n = selectInCapsules(doc.centres, caps, viewer.viewDirection(),
                           { depth: depth * 0.5, alive: visibleMask(), size: brushSize() }).length;
      tSel += performance.now() - t;
    }
    let tState = performance.now();
    applyStates();
    tState = performance.now() - tState;
    return { splats: doc.nTotal, selected: n, runs,
             projectMs: +(tProj / runs).toFixed(2),
             capsuleMs: +(tSel / runs).toFixed(2),
             uploadStateMs: +tState.toFixed(2) };
  },
  /** World-space extent of the current selection, to compare against a known-good one. */
  selectionStats() {
    const selection = selectionIndices();
    if (!selection.length) return null;
    const c = doc.centres;
    const xs: number[] = [], ys: number[] = [], zs: number[] = [];
    for (let k = 0; k < selection.length; k++) {
      const i = selection[k] * 3;
      xs.push(c[i]); ys.push(c[i + 1]); zs.push(c[i + 2]);
    }
    const pct = (a: number[], q: number) => {
      const s = a.slice().sort((p, r) => p - r);
      return +s[Math.min(s.length - 1, Math.floor(q * s.length))].toFixed(2);
    };
    const mean = (a: number[]) => +(a.reduce((p, r) => p + r, 0) / a.length).toFixed(2);
    return {
      n: selection.length,
      centroid: [mean(xs), mean(ys), mean(zs)],
      x: [pct(xs, 0.05), pct(xs, 0.5), pct(xs, 0.95)],
      y: [pct(ys, 0.05), pct(ys, 0.5), pct(ys, 0.95)],
      z: [pct(zs, 0.05), pct(zs, 0.5), pct(zs, 0.95)]
    };
  },
  /** Orbit angles are in the file's up frame (see Viewer.setUpAxis). */
  frame(yaw?: number, pitch?: number, dist?: number) {
    viewer.setView(yaw, pitch, dist);
    projDirty = true;
  },
  upAxis(up?: UpAxis) {
    if (up) applyUpAxis(up, false);
    return viewer.upAxis;
  },
  /**
   * What the size-aware depth test adds to a world-space stroke: select only
   * those splats, and describe them. sigma and the distance behind are along
   * the view ray, the distance in units of the depth window.
   */
  reachDiff(points: [number, number, number][], radius: number, depth = 1.5, k = sizeReach.k) {
    $<HTMLInputElement>('brush-depth').value = String(depth);
    setTool('brush');
    stroke = [];
    for (const [x, y, z] of points) {
      const p = viewer.worldToViewport(x, y, z);
      if (p.onScreen) stroke.push({ x: p.x, y: p.y, r: radius });
    }
    ensureProjection();
    const capsules = strokeToCapsules(grid!, stroke,
      (x, y, d) => viewer.unproject(x, y, d), (d) => viewer.worldPerPixelAt(d));
    const v = viewer.viewDirection();
    const run = (kk: number) => selectInCapsules(doc.centres, capsules, v, {
      depth: depth * 0.5, alive: visibleMask(),
      size: { covA: doc.covA, covB: doc.covB, k: kk, cap: sizeReach.cap }
    });
    const before = run(0), after = run(k);
    const had = new Uint8Array(doc.nTotal);
    for (const i of before) had[i] = 1;
    const added = Array.from(after).filter(i => !had[i]);
    const surfAt = (i: number) => {
      // depth of the stroke surface nearest this splat, from the capsules
      let best = Infinity, d0 = 0;
      for (const c of capsules) {
        const dx = doc.centres[i * 3] - c.x, dy = doc.centres[i * 3 + 1] - c.y,
              dz = doc.centres[i * 3 + 2] - c.z;
        const along = dx * v[0] + dy * v[1] + dz * v[2];
        const lat = dx * dx + dy * dy + dz * dz - along * along;
        if (lat < best) { best = lat; d0 = along / (c.r * depth * 0.5); }
      }
      return d0;
    };
    const sig: number[] = [], alpha: number[] = [], behind: number[] = [];
    const rMean = capsules.reduce((p, c) => p + c.r, 0) / Math.max(capsules.length, 1);
    for (const i of added) {
      const a = i * 4, b = i * 2, A = doc.covA, B = doc.covB;
      const s2 = A[a] * v[0] * v[0] + A[a + 3] * v[1] * v[1] + B[b + 1] * v[2] * v[2]
        + 2 * (A[a + 1] * v[0] * v[1] + A[a + 2] * v[0] * v[2] + B[b] * v[1] * v[2]);
      sig.push(Math.sqrt(Math.max(s2, 0)));
      alpha.push(doc.alpha[i]);
      behind.push(surfAt(i));
    }
    const med = (a: number[]) => a.length
      ? +a.slice().sort((p, q) => p - q)[a.length >> 1].toFixed(3) : null;
    selectMode = 'replace';
    forgetStroke();
    commitHits(Uint32Array.from(added));
    return { k, before: before.length, after: after.length, added: added.length,
             brushRadius: +rMean.toFixed(3), window: +(rMean * depth * 0.5).toFixed(3),
             sigmaMed: med(sig), alphaMed: med(alpha), behindWindowsMed: med(behind),
             inFront: behind.filter(b => b < 0).length };
  },
  /**
   * Share of screen cells holding any splat that the brush finds a surface
   * in, at the scan's own opacity threshold and at the old fixed 0.25.
   */
  surfaceCoverage() {
    const p = ensureProjection();
    if (!p || !grid) return null;
    const r = canvas.getBoundingClientRect();
    const any = depthGrid(p, r.width, r.height, { alive: doc.alive, alpha: doc.alpha, minAlpha: 0 });
    const old = depthGrid(p, r.width, r.height, { alive: doc.alive, alpha: doc.alpha, minAlpha: 0.25 });
    let nAny = 0, nNow = 0, nOld = 0;
    for (let c = 0; c < any.surf.length; c++) {
      if (!isFinite(any.surf[c])) continue;
      nAny++;
      if (isFinite(grid.surf[c])) nNow++;
      if (isFinite(old.surf[c])) nOld++;
    }
    return { surfaceAlpha: +doc.surfaceAlpha.toFixed(3), cells: nAny,
             found: +(nNow / Math.max(nAny, 1)).toFixed(3),
             foundAt025: +(nOld / Math.max(nAny, 1)).toFixed(3) };
  },
  /** Hide everything but the selection, to look at what a stroke took. */
  isolate(on: boolean) {
    if (!viewer.splat) return false;
    if (!on) { applyStates(); return true; }
    const del = doc.baseDeleted();
    for (let i = 0; i < del.length; i++) if (!selMask[i]) del[i] = 1;
    viewer.writeStream(viewer.splat, 'editDel', del);
    if (viewer.generated) {
      const g = doc.genDeleted();
      for (let i = 0; i < g.length; i++) if (!selMask[doc.nBase + i]) g[i] = 1;
      viewer.writeStream(viewer.generated, 'editDel', g);
    }
    return true;
  },
  /** Brush size reach; k = 0 is the old point test. Returns the old value. */
  setSizeReach(k: number, cap = SIZE_CAP) {
    const old = { ...sizeReach };
    sizeReach = { k, cap };
    return old;
  },
  /** The crop sliders' robust range, and where the 0.5/99.5 % box sits on them. */
  cropRange() {
    cropRange = measureCropRange();
    const r = cropRange;
    if (!r) return null;
    const pos = (a: CropAxis, v: number) => {
      const [t0, t1] = cropTails(a);
      return t0 + (1 - t0 - t1) * (v - a.lo) / Math.max(a.hi - a.lo, 1e-9);
    };
    return r.map(a => ({
      min: +a.min.toFixed(2), lo: +a.lo.toFixed(2), hi: +a.hi.toFixed(2), max: +a.max.toFixed(2),
      p005: +pos(a, a.p0).toFixed(3), p995: +pos(a, a.p1).toFixed(3)
    }));
  },
  clearSelection() { clearSelection(); },
  /** Paint through world points with a modifier, as Shift / Ctrl would. */
  paintWorldMode(points: [number, number, number][], radius: number, depth: number,
                 mode: SelectMode) {
    const t = (window as any).__test;
    selectMode = mode;
    const r = t.paintWorld(points, radius, depth);
    return { ...r, mode };
  },
  /** The painted volume: ops, whether it is exact, and the outline's last draw. */
  volume() {
    const w = volumeArg();
    return {
      ops: selVolume ? selVolume.ops.length : null, exact: selVolume ? selVolume.exact : null,
      capsules: w ? w.capsules.length : 0, subtract: w ? w.subtract.length : 0,
      regions: w ? w.regions.length : 0, bytes: w ? JSON.stringify(w).length : 0,
      shell: volumeOutline?.pointCount ?? 0, outline: volumeOutline?.stats ?? null
    };
  },
  /** The volume worked out from scratch must select what the selection holds. */
  volumeCheck() {
    if (!selVolume) return { ok: null, why: 'not painted' };
    const t0 = performance.now();
    const m = evaluateVolume(selVolume, doc.centres, doc.nTotal, visibleMask(), opSize());
    let diff = 0, n = 0;
    for (let i = 0; i < doc.nTotal; i++) { n += m[i]; if (m[i] !== (selMask[i] ?? 0)) diff++; }
    return { ok: diff === 0, diff, volume: n, selected: selCount, exact: selVolume.exact,
             ms: +(performance.now() - t0).toFixed(1) };
  },
  /**
   * What a check outside the app needs: the wire volume, the selection, what
   * was hidden. `volume` is the volume as painted, which selects exactly the
   * selection (the engine's reference test, core/tests/test_volume.py, holds
   * it to that); `sent` is what the commands send, where the rows of a
   * stroke follow what was taken away again (engine/volume.ts, cutRows).
   */
  volumeDump() {
    const vis = visibleMask();
    const hidden: number[] = [];
    for (let i = 0; i < doc.nBase; i++) if (!vis[i]) hidden.push(i);
    return { file: filePath, nBase: doc.nBase, volume: volumeToWire(selVolume, false),
             sent: volumeArg(), selected: Array.from(selectionIndices()), hidden };
  },
  /** Take the selection as an undo step, drop it, put it back. */
  snapshotRoundTrip() {
    const before = selMask.slice(), count = selCount;
    const t0 = performance.now();
    const s = snapshotSelection();
    const t1 = performance.now();
    clearSelection();
    restoreSelection(s);
    const t2 = performance.now();
    let same = selMask.length === before.length;
    for (let i = 0; same && i < before.length; i++) if (before[i] !== selMask[i]) same = false;
    return { count, after: selCount, same, stored: s.indices ? `indices ${s.indices.length}` : 'volume',
             snapshotMs: +(t1 - t0).toFixed(2), restoreMs: +(t2 - t1).toFixed(1) };
  },
  snapshotSelection() { return snapshotSelection(); },
  restoreSelection(s: SelectionSnapshot | null) { restoreSelection(s); return { selected: selCount }; },
  /** Close holes inside lasso and box, or not (to measure). Returns the old value. */
  holeFill(on: boolean) {
    const old = regionHoleFill;
    regionHoleFill = on;
    strokeHoles.on = on;            // the brush's look through holes, too
    return old;
  },
  /** Hide or show the volume outline (screenshots with and without it). */
  outline(on: boolean) {
    if (volumeOutline) {
      volumeOutline.visible = on;
      volumeOutline.invalidate();
      // now, not on the next frame: a window behind others gets few frames,
      // and a screenshot taken before one still showed the outline
      volumeOutline.frame(viewer.viewProjection());
    }
    return on;
  },
  /**
   * The outline's last draw (cells, segments, how many of them hidden) and
   * its live rebuilds. `redraw`: draw it for the camera as it is first.
   */
  outlineState(redraw = false) {
    if (redraw && volumeOutline) {
      volumeOutline.invalidate();
      volumeOutline.frame(viewer.viewProjection());
    }
    return { ...(volumeOutline?.stats ?? {}), liveBuilds: volumeOutline?.liveBuilds ?? 0,
             liveMs: +(volumeOutline?.liveMs ?? 0).toFixed(1), shell: volumeOutline?.pointCount ?? 0,
             painting, selected: selCount, tool };
  },
  /**
   * The renderer's limits for small splats (engine/viewer.ts,
   * setSmallSplatCull): set them when both are given, and return what holds.
   * PlayCanvas' own are (2, 3), the app's (0, 0).
   */
  smallSplats(minPixelSize?: number, minContribution?: number) {
    if (minPixelSize !== undefined && minContribution !== undefined) {
      viewer.setSmallSplatCull(minPixelSize, minContribution);
    }
    return viewer.smallSplatCull;
  },
  /**
   * Time between frames from `frameTimes(true)` until `frameTimes(false)`;
   * `frameTimes(null)` reads without stopping.
   */
  frameTimes(start: boolean | null) {
    const w = window as any;
    if (start) {
      const t: number[] = [];
      if (w.__frames) w.__frames.on = false;
      const rec = { t, on: true };
      w.__frames = rec;
      const tick = (ts: number) => { t.push(ts); if (rec.on) requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
      return true;
    }
    const f = w.__frames;
    if (!f) return null;
    if (start === false) f.on = false;
    const d = f.t.slice(1).map((x: number, i: number) => x - f.t[i]).sort((a: number, b: number) => a - b);
    const q = (p: number) => d.length ? +d[Math.min(d.length - 1, Math.floor(d.length * p))].toFixed(1) : null;
    return { frames: d.length, median: q(0.5), p95: q(0.95), max: q(1) };
  },
  /**
   * Drag files onto the window and drop them, as the OS would. A page cannot
   * make a File that knows its path, so the paths stand in for webUtils;
   * everything after that is the real drop handler.
   */
  async dropFiles(paths: string[]) {
    const dt = new DataTransfer();
    for (const p of paths) dt.items.add(new File([new Uint8Array(1)], p.split(/[\\/]/).pop() ?? p));
    const real = shell!.pathOf;
    shell!.pathOf = (f) => paths.find(p => (p.split(/[\\/]/).pop() ?? p) === f.name) ?? '';
    const at = $('viewport');
    const ev = (type: string) => new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true });
    try {
      at.dispatchEvent(ev('dragenter'));
      at.dispatchEvent(ev('dragover'));
      const overlay = !$('drop').hidden;
      const line = $('drop-line').textContent;
      at.dispatchEvent(ev('drop'));
      const gone = $('drop').hidden;
      // as long as the open takes: the engine may still be starting its GPU
      while (engineBusy) await new Promise(r => setTimeout(r, 50));
      let recent: string | null = null;
      try { recent = JSON.parse(localStorage.getItem('splatRepair.recent') ?? '[]')[0] ?? null; } catch { /* */ }
      return { overlay, line, gone, path: filePath, splats: doc.nTotal, alive: doc.nAlive,
               status: $('status-msg').textContent, statusKind: $('status-msg').className, recent };
    } finally {
      shell!.pathOf = real;
    }
  },
  /* ---- navigation, view cube, history (ui/) ---- */
  /** Camera state. */
  nav() {
    const p = viewer.pose();
    return { ...p, ortho: viewer.isOrtho, animating: viewer.isAnimating,
             fwd: viewer.viewDirection(), up: UP_VECTOR[viewer.upAxis], preset: viewer.preset };
  },
  /** Wait until a view transition has finished. */
  async settle() {
    for (let i = 0; i < 100 && viewer.isAnimating; i++) await new Promise(r => setTimeout(r, 20));
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    return !viewer.isAnimating;
  },
  /**
   * Press Home (or "."), wait, and say whether the bulk of the scan is centred
   * and inside the view: its robust centre's pixel and its sphere's radius in
   * pixels against the viewport.
   */
  async home(key: 'Home' | 'Period' = 'Home') {
    viewer.setView(80, -60, 6);                       // somewhere else first
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: key === 'Home' ? 'Home' : '.', code: key, bubbles: true }));
    await (window as any).__test.settle();
    const b = robustBounds(doc.centres, visibleMask());
    const r = canvas.getBoundingClientRect();
    if (!b) return null;
    const c = viewer.worldToViewport(...b.centre);
    const rPx = b.radius / viewer.worldPerPixelAt(c.depth);
    return { pose: viewer.pose(), centre: [Math.round(c.x), Math.round(c.y)],
             viewport: [Math.round(r.width), Math.round(r.height)], radiusPx: Math.round(rPx),
             centred: Math.abs(c.x - r.width / 2) < 3 && Math.abs(c.y - r.height / 2) < 3,
             fits: rPx < Math.min(r.width, r.height) / 2 && rPx > Math.min(r.width, r.height) * 0.3 };
  },
  /** Click the view cube where a view direction's face (or edge, corner) is. */
  async cubeClick(dir: [number, number, number]) {
    const cube = shell?.cube;
    const el = document.getElementById('viewcube')!;
    if (!cube) return null;
    const p = cube.screenPointOf(dir);
    const opts = { clientX: p.x, clientY: p.y, button: 0, buttons: 1, pointerId: 7, bubbles: true };
    el.dispatchEvent(new PointerEvent('pointermove', { ...opts, buttons: 0 }));
    el.dispatchEvent(new PointerEvent('pointerdown', opts));
    el.dispatchEvent(new PointerEvent('pointerup', { ...opts, buttons: 0 }));
    await (window as any).__test.settle();
    return { local: viewer.viewDirLocal(), fwd: viewer.viewDirection(), up: UP_VECTOR[viewer.upAxis],
             ortho: viewer.isOrtho, pose: viewer.pose() };
  },
  /**
   * Wheel notches over the viewport centre, as real wheel events. Per notch:
   * the surface depth under the cursor before, and how far the eye moved.
   */
  async wheel(notches: number, deltaY = -100, at: [number, number] = [0.5, 0.5]) {
    const r = canvas.getBoundingClientRect();
    const x = r.left + r.width * at[0], y = r.top + r.height * at[1];
    const steps: Array<{ depth: number | null; moved: number }> = [];
    for (let k = 0; k < notches; k++) {
      const d = viewer.surfaceDepthAt(x - r.left, y - r.top);
      const e0 = viewer.camera.getPosition().clone();
      canvas.dispatchEvent(new WheelEvent('wheel', { deltaY, clientX: x, clientY: y, bubbles: true, cancelable: true }));
      const e1 = viewer.camera.getPosition();
      steps.push({ depth: d, moved: e0.distance(e1) });
    }
    await new Promise(res => requestAnimationFrame(() => res(null)));
    return { steps, after: viewer.surfaceDepthAt(r.width * at[0], r.height * at[1]) };
  },
  /** The selection as a count and a cheap fingerprint, to compare before and after undo. */
  selPrint() {
    let h = 2166136261 >>> 0, n = 0;
    for (let i = 0; i < selMask.length; i++) {
      if (!selMask[i]) continue;
      n++;
      h = Math.imul(h ^ i, 16777619) >>> 0;
    }
    return { count: n, hash: h };
  },
  historyState() {
    return { index: history.index, canUndo: history.canUndo, canRedo: history.canRedo,
             entries: history.entries.map(e => `${e.kind}:${e.label}`) };
  },
  async historyUndo() { const ok = await history.undo(); return { ok, ...(window as any).__test.selPrint() }; },
  async historyRedo() { const ok = await history.redo(); return { ok, ...(window as any).__test.selPrint() }; },
  /** Press a key as the user would (the shell's shortcuts). */
  key(key: string, code: string, mods: { ctrl?: boolean; shift?: boolean; alt?: boolean } = {}) {
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key, code, ctrlKey: !!mods.ctrl,
                                                        shiftKey: !!mods.shift, altKey: !!mods.alt,
                                                        bubbles: true }));
    return { selected: selCount, tool };
  },
  stats() {
    return {
      total: doc.nTotal, alive: doc.nAlive, generated: doc.nGenerated,
      selected: selCount, status: $('status-msg').textContent,
      renderer: $('renderer-badge').textContent
    };
  },
  /**
   * Where the selection lies along the current view, as a share of the
   * scan's own depth there: 0 is the nearest visible splat, 1 the farthest
   * (1st and 99th percentile). `far` counts selected splats past `cut` of the
   * way through -- the back of the object. `y` are the selection's file-axis
   * y percentiles (5/50/95), for the sole.
   */
  /**
   * Everything an undo step has to put back, as one comparable string: the
   * selected splats (count and fingerprint), the painted volume's ops, the
   * view it was made from, and the document's splat counts.
   */
  selState() {
    const p = (window as any).__test.selPrint();
    const v = selVolume ? `${selVolume.ops.length}${selVolume.exact ? 'e' : 'n'}` : 'null';
    const view = selView ? selView.map(x => x.toFixed(3)).join(',') : '-';
    return `${p.count}/${p.hash}|vol ${v}|view ${view}|alive ${doc.nAlive}|gen ${doc.nGenerated}`;
  },
  /**
   * The scan's own frame on the ground plane of the up axis: its robust
   * centre, the long axis and the width axis (principal axes of the
   * footprint), and how far it reaches along each. For aiming a test
   * straight at a side.
   */
  scanAxes() {
    const c = doc.centres, vis = visibleMask();
    const up = UP_VECTOR[viewer.upAxis] as number[];
    const ui = up.findIndex(v => v !== 0);
    const [ai, bi] = [0, 1, 2].filter(k => k !== ui);
    const b = robustBounds(c, vis);
    if (!b) return null;
    let saa = 0, sab = 0, sbb = 0, n = 0;
    for (let i = 0; i < doc.nTotal; i += 5) {
      if (!vis[i]) continue;
      const da = c[i * 3 + ai] - b.centre[ai], db = c[i * 3 + bi] - b.centre[bi];
      saa += da * da; sab += da * db; sbb += db * db; n++;
    }
    saa /= n; sab /= n; sbb /= n;
    const ang = 0.5 * Math.atan2(2 * sab, saa - sbb);
    const long = [0, 0, 0], wide = [0, 0, 0];
    long[ai] = Math.cos(ang); long[bi] = Math.sin(ang);
    wide[ai] = -Math.sin(ang); wide[bi] = Math.cos(ang);
    const span = (ax: number[]) => {
      const t: number[] = [];
      for (let i = 0; i < doc.nTotal; i += 5) {
        if (vis[i]) t.push(c[i * 3] * ax[0] + c[i * 3 + 1] * ax[1] + c[i * 3 + 2] * ax[2]);
      }
      t.sort((p, q) => p - q);
      return t[Math.floor(t.length * 0.98)] - t[Math.floor(t.length * 0.02)];
    };
    return { centre: b.centre, long, wide, length: span(long), width: span(wide) };
  },
  /** Crop preview off, sliders back (the autotest's way to change what is visible). */
  resetCrop() { resetCrop(); return { outside: cropOut.length, active: cropActive }; },
  selectionDepth(cut = 0.3) {
    const v = viewer.viewDirection(), c = doc.centres, vis = visibleMask();
    const all: number[] = [];
    for (let i = 0; i < doc.nTotal; i += 7) {
      if (vis[i]) all.push(c[i * 3] * v[0] + c[i * 3 + 1] * v[1] + c[i * 3 + 2] * v[2]);
    }
    all.sort((a, b) => a - b);
    const lo = all[Math.floor(all.length * 0.01)] ?? 0, hi = all[Math.floor(all.length * 0.99)] ?? 1;
    const sel = selectionIndices();
    const rel: number[] = [], ys: number[] = [];
    let far = 0;
    for (let k = 0; k < sel.length; k++) {
      const i = sel[k];
      const t = (c[i * 3] * v[0] + c[i * 3 + 1] * v[1] + c[i * 3 + 2] * v[2] - lo) / Math.max(hi - lo, 1e-9);
      rel.push(t); ys.push(c[i * 3 + 1]);
      if (t > cut) far++;
    }
    rel.sort((a, b) => a - b); ys.sort((a, b) => a - b);
    const q = (a: number[], f: number) => a.length ? +a[Math.min(a.length - 1, Math.floor(f * a.length))].toFixed(3) : null;
    return { selected: sel.length, far, farShare: sel.length ? +(far / sel.length).toFixed(4) : 0,
             depth: [q(rel, 0.05), q(rel, 0.5), q(rel, 0.95), q(rel, 0.999)],
             y: [q(ys, 0.05), q(ys, 0.5), q(ys, 0.95)], view: v.map(x => +x.toFixed(3)) };
  }
};
