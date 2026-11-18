/**
 * One history for the whole document: selection steps and engine edits in the
 * order they happened, the way Photoshop's history panel works.
 *
 * The engine keeps its own undo stack for edits (core/splatcore/document.py);
 * the selection lives here in the app. This list interleaves the two:
 *
 *   - a selection step stores the selection before and after. Undoing it
 *     puts the old selection back; the engine is not asked.
 *   - an edit step stores the selection before and after the edit. Undoing
 *     it runs the engine's undo and then restores the selection the edit
 *     started from -- so Ctrl+Z after a repair brings back the splats *and*
 *     the stroke that selected them.
 *
 * Because the engine's stack is strictly linear and this list only ever adds
 * edits in the same order, the n-th edit here is always the engine's n-th
 * undo step.
 *
 * What a "selection" is stays open: it is made of parts, each with its own
 * snapshot, restore and equality. Today there is one part, the per-splat
 * mask; a painted volume can register a second (see `addPart`).
 */

export interface SelectionPart<S = unknown> {
  /** A copy of this part's state as it is now. Must not alias live data. */
  snapshot(): S;
  /** Put a snapshot back. Called with history notifications muted. */
  restore(s: S): void;
  /** Same selection? Without it, every notification counts as a change. */
  equal?(a: S, b: S): boolean;
  /** Rough size in bytes, for the memory cap. */
  bytes?(s: S): number;
}

type Snap = Record<string, unknown>;

export interface HistoryEntry {
  kind: 'base' | 'selection' | 'edit';
  label: string;
  before: Snap;
  after: Snap;
  time: number;
  /** Geometry revision after this step. Selection steps inherit it. */
  revision: number;
  /** A run of changes that folds into this step (`mergeNext`), and when it last did. */
  merge?: { key: string; at: number };
}

export interface HistoryHost {
  /** The engine's undo / redo. Resolve with the reply header, or null on failure. */
  engineUndo(): Promise<Record<string, any> | null>;
  engineRedo(): Promise<Record<string, any> | null>;
  /** Is an engine command running? */
  busy(): boolean;
  /** Undo or redo was asked for while a command runs, and did nothing: say so. */
  refused?(): void;
}

/**
 * Busy-overlay text to a history step name. The overlay text comes from
 * src/main.ts; the patterns match its wording.
 */
const BUSY_LABELS: Array<[RegExp, string]> = [
  [/^Deep/i, 'Deep Fill'],
  [/^Repair/i, 'Repair'],
  [/shadow/i, 'Remove Shadow'],
  [/^Lighten/i, 'Lighten'],
  [/^Solidif/i, 'Solidify'],
  [/^Delet/i, 'Delete'],
  [/^Apply/i, 'Move'],
  [/^Stamp/i, 'Clone Stamp'],
  [/^Crop/i, 'Crop'],
  [/^Remov\w* invalid/i, 'Remove Invalid Splats'],
  [/^Find\w* floaters/i, 'Select Floaters']
];

export function labelForBusy(text: string): string {
  for (const [re, l] of BUSY_LABELS) if (re.test(text)) return l;
  return text.replace(/\s*(…|\.\.\.)$/, '').trim() || 'Edit';
}

const MAX_ENTRIES = 200;
const MAX_BYTES = 256 * 1024 * 1024;

export class History {
  private parts = new Map<string, SelectionPart<any>>();
  private list: HistoryEntry[] = [];
  /** Index of the entry whose `after` is the current state. */
  private pos = -1;
  private nextRevision = 1;
  /** The selection as last committed to the list. */
  private current: Snap = {};
  private muted = 0;
  private pendingLabel: string | null = null;
  /** The next selection change may fold into the step on top (`mergeNext`). */
  private nextMerge: { key: string; ms: number } | null = null;
  private replayHeader: Record<string, any> | null = null;
  private working = false;
  private listeners: Array<() => void> = [];
  /** What the next selection step is called, if the caller knows better. */
  private nextLabel: string | null = null;
  /** The running engine command, as a step name. */
  private busyLabel: string | null = null;
  /** An edit recorded while its command still runs (see `noteEdit`, `flush`). */
  private openEdit: HistoryEntry | null = null;

  constructor(private host: HistoryHost) {}

  /* ------------------------------------------------------------ setup */

  /**
   * Register a part of the selection. The volume brush, for instance:
   *
   *   history.addPart('volume', {
   *     snapshot: () => volume.clone(),
   *     restore: (v) => { volume.copyFrom(v); redrawVolume(); },
   *     equal: (a, b) => a.equals(b),
   *   });
   *
   * and after each change to it: `history.noteSelection('Volume')`.
   */
  addPart<S>(name: string, part: SelectionPart<S>) {
    this.parts.set(name, part);
    if (this.list.length) this.current[name] = part.snapshot();
  }

  onChange(fn: () => void) { this.listeners.push(fn); }
  private changed() { for (const f of this.listeners) f(); }
  /** After a successful undo or redo, with the step it went over. */
  onStep(fn: (dir: 'undo' | 'redo', e: HistoryEntry) => void) { this.stepListeners.push(fn); }
  private stepListeners: Array<(dir: 'undo' | 'redo', e: HistoryEntry) => void> = [];
  private stepped(dir: 'undo' | 'redo', e: HistoryEntry) { for (const f of this.stepListeners) f(dir, e); }

  /** A new document: one base entry, nothing to undo. */
  reset(label: string) {
    this.pendingLabel = null;
    this.openEdit = null;
    this.current = this.snapshot();
    this.list = [{ kind: 'base', label, before: this.current, after: this.current,
                   time: Date.now(), revision: 0 }];
    this.pos = 0;
    this.changed();
  }

  /** No document any more: an empty list, and nothing recorded until `reset`. */
  clear() {
    this.pendingLabel = null;
    this.openEdit = null;
    this.current = {};
    this.list = [];
    this.pos = -1;
    this.changed();
  }

  /* ------------------------------------------------------ notifications */

  /** Run `fn` without recording what it does to the selection. */
  mute<T>(fn: () => T): T {
    this.muted++;
    try { return fn(); } finally { this.muted--; }
  }

  get isReplaying() { return this.muted > 0 || this.working; }

  /** Name the next selection step ("Select All"), when the label would be wrong. */
  labelNext(label: string) { this.nextLabel = label; }

  /**
   * The next selection change is one more of a run (`key`): it folds into the
   * step on top if that step is the same run and changed less than `ms` ago,
   * else it starts the run as a step of its own. Dragging the depth slider
   * re-runs the stroke on every frame, and that is one step, not sixty --
   * but its own step, after the stroke's.
   *
   * This used to be a time window for any change: the slider's first frame
   * folded into the stroke before it (Ctrl+Z then took the stroke away
   * instead of the new depth), and a lasso drawn within 0.8 s of letting go
   * of the slider folded into the slider's step.
   */
  mergeNext(key: string, ms = 800) { this.nextMerge = { key, ms }; }

  /**
   * The selection changed. Recorded as a step now -- or, while an engine
   * command runs, held back: if the command turns out to be an edit, the
   * change belongs to the edit (a repair clears the selection); if not
   * (picking floaters), it becomes a step when the command finishes.
   */
  noteSelection(label: string) {
    if (this.muted || this.working || !this.list.length) return;
    const l = this.nextLabel ?? label;
    const merge = this.nextMerge;
    this.nextLabel = null;
    this.nextMerge = null;
    if (this.host.busy()) { this.pendingLabel = this.busyLabel ?? l; return; }
    this.commitSelection(l, merge);
  }

  /**
   * An engine command starts, with the text the busy overlay shows. That
   * names the step it may become ("Repairing …" becomes "Repair").
   */
  noteBusy(text: string) {
    if (this.working) return;
    this.busyLabel = labelForBusy(text);
  }

  /**
   * The engine went idle: a held-back selection change becomes a step --
   * unless the command recorded its edit before it was done with the
   * selection (`noteEdit`). Then the change is the end of that edit: a
   * repair clears the selection after its edit is on the list.
   */
  flush() {
    const open = this.openEdit;
    this.openEdit = null;
    if (this.working) return;
    if (open && open === this.list[this.pos]) {
      this.pendingLabel = null;
      const now = this.snapshot();
      if (!this.same(now, this.current)) {
        open.after = now;
        this.current = now;
        this.changed();
      }
      return;
    }
    if (this.pendingLabel === null) return;
    const l = this.pendingLabel;
    this.pendingLabel = null;
    this.commitSelection(l);
  }

  private commitSelection(label: string, merge: { key: string; ms: number } | null = null) {
    const now = this.snapshot();
    if (this.same(now, this.current)) return;
    const top = this.list[this.pos];
    const t = performance.now();
    if (merge && top?.kind === 'selection' && this.pos === this.list.length - 1
        && top.merge?.key === merge.key && t - top.merge.at < merge.ms) {
      top.after = now;
      top.merge.at = t;
      this.current = now;
      this.changed();
      return;
    }
    this.push({ kind: 'selection', label, before: this.current, after: now, time: Date.now(),
                revision: this.revision,
                ...(merge ? { merge: { key: merge.key, at: t } } : {}) });
  }

  /**
   * An engine reply came back. If it says an edit was made, record it, with
   * the selection from before the command and the one it left behind.
   * Undo and redo replies (they carry `applied`) are the history's own.
   *
   * Called as soon as the reply is applied to the document, so that nothing
   * that fails afterwards (a rebuild, a status line) can leave the engine
   * with an undo step this list does not have. What the command still does
   * to the selection before the engine goes idle is folded in by `flush`.
   */
  noteEdit(header: Record<string, any>, label?: string) {
    if (this.working) { this.replayHeader = header; return; }
    if (!this.list.length || 'applied' in header || header.canUndo !== true) return;
    this.pendingLabel = null;
    const after = this.snapshot();
    const e: HistoryEntry = { kind: 'edit', label: label ?? this.busyLabel ?? 'Edit',
                              before: this.current, after, time: Date.now(),
                              revision: this.nextRevision++ };
    this.push(e);
    this.openEdit = this.host.busy() ? e : null;
  }

  private push(e: HistoryEntry) {
    // a new step drops whatever could have been redone
    this.list.length = this.pos + 1;
    this.list.push(e);
    this.pos = this.list.length - 1;
    this.current = e.after;
    this.trim();
    this.changed();
  }

  /**
   * Keep the list bounded. The oldest steps go first; an edit that falls off
   * stays in the engine but can no longer be reached from here, which is
   * where Photoshop draws the line too.
   */
  private trim() {
    let bytes = 0;
    for (const e of this.list) bytes += this.bytesOf(e.before) + this.bytesOf(e.after);
    while (this.list.length > 2 && (this.list.length > MAX_ENTRIES || bytes > MAX_BYTES)) {
      const gone = this.list.splice(1, 1)[0];
      bytes -= this.bytesOf(gone.before) + this.bytesOf(gone.after);
      // The base takes on both the selection and geometry left by the dropped step.
      this.list[0] = { ...this.list[0], before: gone.after, after: gone.after,
                       revision: gone.revision };
      this.pos--;
    }
  }

  /* ------------------------------------------------------------- moving */

  get canUndo() { return this.pos > 0; }
  get canRedo() { return this.pos < this.list.length - 1; }
  get entries(): readonly HistoryEntry[] { return this.list; }
  get index() { return this.pos; }
  /** Zero denotes the original file (or an empty history). */
  get revision() { return this.list[this.pos]?.revision ?? 0; }
  get undoLabel() { return this.canUndo ? this.list[this.pos].label : null; }
  get redoLabel() { return this.canRedo ? this.list[this.pos + 1].label : null; }

  /** One step back. Resolves false when nothing happened. */
  async undo(): Promise<boolean> {
    if (!this.canUndo) return false;
    if (this.working || this.host.busy()) { this.host.refused?.(); return false; }
    const e = this.list[this.pos];
    if (e.kind === 'edit') {
      const h = await this.replay(() => this.host.engineUndo());
      if (!h || h.applied !== true) { this.changed(); return false; }
    }
    this.pos--;
    this.restore(e.before);
    this.changed();
    this.stepped('undo', e);
    return true;
  }

  async redo(): Promise<boolean> {
    if (!this.canRedo) return false;
    if (this.working || this.host.busy()) { this.host.refused?.(); return false; }
    const e = this.list[this.pos + 1];
    if (e.kind === 'edit') {
      const h = await this.replay(() => this.host.engineRedo());
      if (!h || h.applied !== true) { this.changed(); return false; }
    }
    this.pos++;
    this.restore(e.after);
    this.changed();
    this.stepped('redo', e);
    return true;
  }

  /** Walk to an entry, one step at a time (the history panel's click). */
  async jumpTo(i: number) {
    while (this.pos > i) if (!(await this.undo())) break;
    while (this.pos < i) if (!(await this.redo())) break;
  }

  private async replay(fn: () => Promise<Record<string, any> | null>) {
    this.working = true;
    this.replayHeader = null;
    try {
      const r = await fn();
      return r ?? this.replayHeader;
    } finally {
      this.working = false;
      this.pendingLabel = null;
    }
  }

  /* -------------------------------------------------------- snapshots */

  private snapshot(): Snap {
    const s: Snap = {};
    for (const [k, p] of this.parts) s[k] = p.snapshot();
    return s;
  }

  private restore(s: Snap) {
    this.mute(() => {
      for (const [k, p] of this.parts) if (k in s) p.restore(s[k]);
    });
    this.current = s;
  }

  private same(a: Snap, b: Snap): boolean {
    for (const [k, p] of this.parts) {
      if (!(k in a) || !(k in b)) return false;
      if (p.equal ? !p.equal(a[k], b[k]) : a[k] !== b[k]) return false;
    }
    return true;
  }

  private bytesOf(s: Snap): number {
    let n = 0;
    for (const [k, p] of this.parts) if (p.bytes && k in s) n += p.bytes(s[k]);
    return n;
  }
}
