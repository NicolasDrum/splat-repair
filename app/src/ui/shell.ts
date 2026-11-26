/**
 * The app shell: menu bar, keyboard shortcuts, options bar, history panel,
 * help card and the small things around the viewport.
 *
 * Everything is one list of commands. The menus, the shortcut handler and the
 * help card are all built from it, so a shortcut shown in a menu is the one
 * that works, and the help card cannot fall behind.
 *
 * Commands act through the same buttons the panels show (clicking
 * #btn-repair, say) wherever one exists. The panel and the menu then cannot
 * disagree about what a command does or when it is allowed: a greyed-out
 * button is a greyed-out menu item.
 */
import type { Viewer, NavPreset } from '../engine/viewer';
import type { History } from './history';
import { ViewCube, viewName } from './viewcube';

export interface ShellApi {
  viewer: Viewer;
  history: History;
  hasDoc(): boolean;
  selCount(): number;
  tool(): string;
  setTool(t: string): void;
  selectAll(): void;
  deselect(): void;
  frameSelection(): void;
  openPath(p: string): void;
  brushStep(dir: 1 | -1): void;
  status(msg: string, kind?: 'ok' | 'err' | ''): void;
  /** Is an engine command running? Nothing new opens meanwhile. */
  busy?(): boolean;
  /** Would opening another file lose edits? The test the close question uses. */
  unexported?(): boolean;
  /** Live splats without a valid position (NaN or infinite), and their removal. */
  invalidCount?(): number;
  dropInvalid?(): void;
  /** Lasso and box take everything inside the outline, the back included. */
  through?(): boolean;
  setThrough?(on: boolean): void;
}

/** What the engine opens; the same list the open dialog filters for. */
const SPLAT_FILE = /\.(ply|splat|spz)$/i;

export type SelMode = 'replace' | 'add' | 'subtract';

interface Cmd {
  label: string | (() => string);
  keys?: string;              // as shown
  bind?: string[];            // what matches, see `matches`
  run?: () => void;
  enabled?: () => boolean;
  checked?: () => boolean;
  sub?: () => Item[];
  /** Not in menus, only a key (Escape, Backspace...). */
  hidden?: boolean;
  /** Tooltip (a recent file's full path). */
  title?: string;
}
type Item = Cmd | '-' | { note: string };

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const click = (id: string) => () => { const b = $<HTMLButtonElement>(id); if (b && !b.disabled) b.click(); };
const live = (id: string) => () => { const b = $<HTMLButtonElement>(id); return !!b && !b.disabled; };

const TOOL_NAME: Record<string, string> = {
  orbit: 'Orbit', brush: 'Brush', lasso: 'Lasso', box: 'Rectangle', clone: 'Clone Stamp', xform: 'Move'
};

const PRESET_KEY = 'splatRepair.navPreset';
const RECENT_KEY = 'splatRepair.recent';
const PANEL_KEY = 'splatRepair.panel.';
const BETA_KEY = 'splatRepair.beta';

/** How the mouse works, in words, per preset. Shown in the help card and the status bar. */
function navRows(p: NavPreset): Array<[string, string]> {
  const common: Array<[string, string]> = [
    ['Zoom to the pointer', 'Wheel'],
    ['Zoom in finer steps', 'Shift+Wheel'],
    ['Zoom through the surface', 'Ctrl+Wheel'],
    ['Center on a point', 'Middle-click · double-click in Orbit'],
    ['Fly', 'Hold right + W A S D, Q E'],
    ['Fly faster / speed', 'Shift · wheel while flying'],
    ['Like Maya and Houdini', 'Alt + left, middle, right'],
  ];
  if (p === 'supersplat') {
    return [
      ['Orbit around the focus point', 'Drag left in Orbit · Alt+left'],
      ['Pan', 'Drag right · Shift+left in Orbit'],
      ['Zoom by dragging', 'Drag middle · Ctrl+left in Orbit'],
      ...common
    ];
  }
  return [
    ['Orbit around the point under the pointer', 'Drag middle or right · Alt+left'],
    ['Pan', 'Shift+middle · Alt+middle'],
    ['Zoom by dragging', 'Ctrl+middle · Alt+right'],
    ['In Orbit', 'Left orbits, Shift pans, Ctrl zooms'],
    ...common
  ];
}

function navHint(p: NavPreset, tool: string): string {
  if (tool === 'orbit') {
    return p === 'supersplat'
      ? 'Left: orbit · right: pan · wheel: zoom · Home: default view · F1: all keys'
      : 'Left: orbit · Shift: pan · wheel: zoom · Home: default view · F1: all keys';
  }
  return p === 'supersplat'
    ? 'Alt+left: orbit · right: pan · wheel: zoom · Home: default view · F1: all keys'
    : 'Middle/right: orbit · Shift+middle: pan · wheel: zoom · Home: default view · F1: all keys';
}

export class Shell {
  selMode: SelMode = 'replace';
  /**
   * "Beta Tools" (Edit menu): the tools that are paused -- the
   * shadow brush and "Solidify" -- are shown. Off unless the user ticked
   * it; see `setBeta`.
   */
  beta = false;
  cube: ViewCube | null = null;
  private menus: Array<{ title: string; items: () => Item[] }> = [];
  private openIdx = -1;
  private hot = -1;
  private toastTimer = 0;
  private lastState = '';
  private historyKey = '';

  constructor(private api: ShellApi) {
    const v = api.viewer;
    try {
      const p = localStorage.getItem(PRESET_KEY) as NavPreset | null;
      if (p === 'blender' || p === 'supersplat') v.preset = p;
    } catch { /* not persistent */ }
    let beta = false;
    try { beta = localStorage.getItem(BETA_KEY) === '1'; } catch { /* not persistent: off */ }
    this.setBeta(beta, false);
    this.defineMenus();
    this.buildMenubar();
    this.wireKeys();
    this.wireOptions();
    this.wirePanels();
    this.wireNav();
    this.wireHelp();
    this.wireDrop();
    api.history.onChange(() => this.renderHistory());
    api.history.onStep((dir, e) => api.status(
      `${dir === 'undo' ? 'undone' : 'redone'}: ${e.label}`, 'ok'));
    this.renderHistory();
    v.onNotice = (m) => this.toast(m);
    this.onTool(api.tool());
  }

  /* ----------------------------------------------------------- commands */

  private defineMenus() {
    const a = this.api, v = a.viewer, h = a.history;
    const doc = () => a.hasDoc();
    const sel = () => a.selCount() > 0;
    const tool = (t: string, key: string, bind: string[]): Cmd => ({
      label: TOOL_NAME[t], keys: key, bind, run: () => a.setTool(t),
      checked: () => a.tool() === t
    });
    const view = (name: string, keys: string, bind: string[], dir: [number, number, number]): Cmd => ({
      label: name, keys, bind, enabled: doc, run: () => this.snap(dir)
    });
    const preset = (p: NavPreset, label: string): Cmd => ({
      label, checked: () => v.preset === p, run: () => this.setPreset(p)
    });
    const mode = (m: SelMode, label: string, keys: string): Cmd => ({
      label, keys, checked: () => this.selMode === m, run: () => this.setSelMode(m)
    });
    this.menus = [
      { title: 'File', items: () => [
        { label: 'Open…', keys: 'Ctrl+O', bind: ['ctrl+o'], run: click('btn-open') },
        { label: 'Open Recent', sub: () => this.recentItems(), enabled: () => this.recent().length > 0 },
        { note: 'Or drag the file into the window.' },
        '-',
        { label: 'Export…', keys: 'Ctrl+S', bind: ['ctrl+s', 'ctrl+shift+s'],
          run: click('btn-export'), enabled: live('btn-export') },
        '-',
        // through the main process, which asks about unexported edits first;
        // window.close() here closed the window past that question
        { label: 'Quit', keys: 'Ctrl+Q', bind: ['ctrl+q'], run: () => window.host.closeWindow() }
      ] },
      { title: 'Edit', items: () => [
        { label: () => h.undoLabel ? `Undo: ${h.undoLabel}` : 'Undo', keys: 'Ctrl+Z',
          bind: ['ctrl+z'], run: () => { void h.undo(); }, enabled: () => h.canUndo },
        { label: () => h.redoLabel ? `Redo: ${h.redoLabel}` : 'Redo', keys: 'Ctrl+Shift+Z',
          bind: ['ctrl+shift+z', 'ctrl+y'], run: () => { void h.redo(); }, enabled: () => h.canRedo },
        '-',
        { label: 'Delete Selection', keys: 'Del', bind: ['Delete', 'Backspace'],
          run: click('btn-delete'), enabled: live('btn-delete') },
        { label: 'Transform', keys: 'G', run: () => a.setTool('xform'), enabled: sel },
        '-',
        { label: 'Show History', run: () => this.openPanel('history-tile') },
        '-',
        // There is no settings page: what the app remembers is ticked in its
        // menus (the mouse preset under View), and so is this.
        { label: 'Beta Tools', checked: () => this.beta, run: () => {
          this.setBeta(!this.beta);
          this.toast(this.beta ? 'Beta Tools on' : 'Beta Tools off');
        } },
        { note: 'Remove Shadow and Solidify. Both are still in progress and hidden otherwise.' }
      ] },
      { title: 'Select', items: () => [
        { label: 'Select All', keys: 'Ctrl+A', bind: ['ctrl+a'], run: () => a.selectAll(), enabled: doc },
        { label: 'Deselect', keys: 'Ctrl+D', bind: ['ctrl+d'], run: () => a.deselect(), enabled: sel },
        { label: 'Invert Selection', keys: 'Ctrl+Shift+I', bind: ['ctrl+shift+i'],
          run: () => { h.labelNext('Invert Selection'); click('btn-invert')(); }, enabled: live('btn-invert') },
        '-',
        { label: 'Select Floaters', run: () => { this.openPanel('cleanup-tile'); click('btn-floaters')(); },
          enabled: live('btn-floaters') },
        '-',
        mode('replace', 'New Selection', ''),
        mode('add', 'Add', 'Shift'),
        mode('subtract', 'Subtract', 'Ctrl'),
        { note: 'Shift and Ctrl work while dragging, without switching the mode.' },
        '-',
        { label: 'Lasso and Rectangle Select Through', keys: 'Alt+Z', bind: ['alt+z'],
          checked: () => a.through?.() ?? false, run: () => a.setThrough?.(!(a.through?.() ?? false)) },
        { note: 'Off: only what you see. On: everything behind it too.' }
      ] },
      { title: 'View', items: () => [
        { label: 'Default View', keys: 'Home', bind: ['Home'], run: () => this.home(), enabled: doc },
        { label: 'Frame All', keys: '.', bind: ['Period', 'NumpadDecimal'], run: () => this.frameAll(), enabled: doc },
        { label: 'Frame Selection', keys: 'F', bind: ['f'], run: () => a.frameSelection(), enabled: doc },
        '-',
        view('Front', 'Num 1', ['Numpad1', 'Digit1'], [0, 0, 1]),
        view('Back', 'Ctrl+Num 1', ['ctrl+Numpad1', 'ctrl+Digit1'], [0, 0, -1]),
        view('Right', 'Num 3', ['Numpad3', 'Digit3'], [1, 0, 0]),
        view('Left', 'Ctrl+Num 3', ['ctrl+Numpad3', 'ctrl+Digit3'], [-1, 0, 0]),
        view('Top', 'Num 7', ['Numpad7', 'Digit7'], [0, 1, 0]),
        view('Bottom', 'Ctrl+Num 7', ['ctrl+Numpad7', 'ctrl+Digit7'], [0, -1, 0]),
        { label: 'Orthographic', keys: 'Num 5', bind: ['Numpad5', 'Digit5'], enabled: doc,
          checked: () => v.isOrtho, run: () => this.toggleOrtho() },
        '-',
        { label: 'Switch Up Axis', keys: 'U', bind: ['u'], run: click('btn-up'), enabled: live('btn-up') },
        '-',
        preset('blender', 'Mouse Like Blender'),
        preset('supersplat', 'Mouse Like SuperSplat'),
        { note: 'Fly: hold the right mouse button, then W A S D, Q and E. The wheel changes the speed.' },
        '-',
        { label: 'Full Screen', keys: 'F11', bind: ['F11'], run: () => this.fullscreen(),
          checked: () => !!document.fullscreenElement }
      ] },
      { title: 'Tools', items: () => [
        tool('orbit', 'V', ['v', 'h']),
        tool('brush', 'B', ['b']),
        tool('lasso', 'L', ['l']),
        tool('box', 'R', ['r', 'm']),
        tool('clone', 'S', ['s']),
        tool('xform', 'G', ['g']),
        '-',
        { label: 'Increase Brush Size', keys: '+', bind: ['key:+', 'NumpadAdd'], run: () => a.brushStep(1) },
        { label: 'Decrease Brush Size', keys: '−', bind: ['key:-', 'NumpadSubtract'], run: () => a.brushStep(-1) }
      ] },
      { title: 'Filter', items: () => [
        { label: 'Repair', keys: 'Shift+F5', bind: ['shift+F5'], run: click('btn-repair'),
          enabled: live('btn-repair') },
        { label: 'Try Another Fill', run: click('btn-reroll'), enabled: live('btn-reroll') },
        { label: 'Deep Fill', checked: () => $('depth-deep')?.classList.contains('active') ?? false,
          enabled: live('depth-deep'),
          run: () => ($('depth-deep').classList.contains('active') ? click('depth-quick') : click('depth-deep'))() },
        // The paused tools, only with "Beta Tools" on. Off, their entries
        // are not in the list at all, so neither a key nor the help card
        // (both built from it) can reach them.
        ...(this.beta ? [
          '-',
          // what the button says: with the shadow menu on "Lighten" it lifts,
          // and the entry went on calling that "Remove Shadow"
          { label: () => `${$('btn-unshadow')?.textContent || 'Remove Shadow'} (Beta)`,
            run: click('btn-unshadow'), enabled: live('btn-unshadow') },
          { label: 'Solidify (Beta)', run: click('btn-solidify'), enabled: live('btn-solidify') }
        ] as Item[] : []),
        '-',
        { label: 'Crop', run: () => { this.openPanel('cleanup-tile'); click('btn-crop')(); },
          enabled: live('btn-crop') },
        { label: 'Show Crop Box', run: () => this.openPanel('cleanup-tile') },
        // Splats whose position is NaN or infinite. Nobody sees them, so
        // the entry says how many there are, and is off when there are none.
        { label: () => {
            const n = a.invalidCount?.() ?? 0;
            return `Remove Invalid Splats (${n ? n.toLocaleString('en-US') : 'none found'})`;
          },
          run: () => a.dropInvalid?.(), enabled: () => (a.invalidCount?.() ?? 0) > 0 }
      ] },
      { title: 'Help', items: () => [
        { label: 'Keys and Mouse', keys: 'F1', bind: ['F1', 'key:?'], run: () => this.showHelp(true) },
        { label: 'About Splat Repair', run: () => a.status('Splat Repair 0.1 · Repair Gaussian splats, '
          + 'engine in Python, viewport in PlayCanvas', 'ok') }
      ] }
    ];
  }

  /**
   * Show or hide the paused tools: everything in the page marked
   * `data-beta` (the retouch panel), and with it their menu entries
   * (`defineMenus` reads the flag). `remember`: keep it for the next start.
   * Hidden, not removed: the code behind the controls stays wired, and a
   * step they made earlier can still be undone.
   */
  setBeta(on: boolean, remember = true) {
    this.beta = on;
    for (const el of document.querySelectorAll<HTMLElement>('[data-beta]')) el.hidden = !on;
    if (remember) {
      try { localStorage.setItem(BETA_KEY, on ? '1' : '0'); } catch { /* not persistent */ }
    }
    if (!$('help').hidden) this.showHelp(true);
  }

  /** Every command as the menus list it now, for tests: menu, label, keys shown, whether a key is bound. */
  commands(): Array<{ menu: string; label: string; keys: string; bound: boolean }> {
    const out: Array<{ menu: string; label: string; keys: string; bound: boolean }> = [];
    for (const m of this.menus) {
      for (const it of m.items()) {
        if (typeof it !== 'object' || !('label' in it)) continue;
        out.push({ menu: m.title, label: typeof it.label === 'function' ? it.label() : it.label,
                   keys: it.keys ?? '', bound: !!it.bind?.length });
      }
    }
    return out;
  }

  private allCommands(): Cmd[] {
    const out: Cmd[] = [];
    for (const m of this.menus) for (const it of m.items()) if (typeof it === 'object' && 'label' in it) out.push(it);
    return out;
  }

  /* --------------------------------------------------------------- menus */

  private buildMenubar() {
    const bar = $('menubar');
    bar.textContent = '';
    this.menus.forEach((m, i) => {
      const wrap = document.createElement('div');
      wrap.className = 'menu';
      const t = document.createElement('button');
      t.className = 'menu__title';
      t.textContent = m.title;
      t.setAttribute('role', 'menuitem');
      t.setAttribute('aria-haspopup', 'true');
      t.addEventListener('click', () => this.openMenu(this.openIdx === i ? -1 : i));
      t.addEventListener('pointerenter', () => { if (this.openIdx >= 0 && this.openIdx !== i) this.openMenu(i); });
      wrap.appendChild(t);
      bar.appendChild(wrap);
    });
    document.addEventListener('pointerdown', (e) => {
      if (this.openIdx >= 0 && !(e.target as HTMLElement).closest?.('.menubar')) this.openMenu(-1);
    }, true);
    window.addEventListener('blur', () => this.openMenu(-1));
  }

  private openMenu(i: number) {
    const bar = $('menubar');
    bar.querySelectorAll('.menu').forEach((el, k) => {
      el.classList.toggle('open', k === i);
      el.querySelector('.menu__title')?.setAttribute('aria-expanded', String(k === i));
      el.querySelector('.menu__list')?.remove();
    });
    this.openIdx = i;
    this.hot = -1;
    if (i < 0) return;
    const wrap = bar.querySelectorAll('.menu')[i] as HTMLElement;
    wrap.appendChild(this.renderList(this.menus[i].items(), false));
  }

  private renderList(items: Item[], sub: boolean): HTMLElement {
    const ul = document.createElement('div');
    ul.className = 'menu__list' + (sub ? ' menu__list--sub' : '');
    ul.setAttribute('role', 'menu');
    for (const it of items) {
      if (it === '-') {
        const s = document.createElement('div'); s.className = 'menu__sep'; s.setAttribute('role', 'separator');
        ul.appendChild(s); continue;
      }
      if ('note' in it) {
        const n = document.createElement('div'); n.className = 'menu__note'; n.textContent = it.note;
        ul.appendChild(n); continue;
      }
      if (it.hidden) continue;
      const b = document.createElement('button');
      b.className = 'menu__item' + (it.sub ? ' menu__item--sub' : '');
      b.setAttribute('role', it.checked ? 'menuitemcheckbox' : 'menuitem');
      const on = it.checked?.() ?? false;
      if (it.checked) b.setAttribute('aria-checked', String(on));
      const c = document.createElement('span'); c.className = 'menu__check'; c.textContent = on ? '✓' : '';
      const l = document.createElement('span'); l.textContent = typeof it.label === 'function' ? it.label() : it.label;
      const k = document.createElement('span'); k.className = 'menu__keys'; k.textContent = it.keys ?? '';
      b.append(c, l, k);
      b.disabled = it.enabled ? !it.enabled() : false;
      if (it.title) b.title = it.title;
      if (it.sub) {
        b.addEventListener('pointerenter', () => {
          ul.querySelectorAll('.menu__list--sub').forEach(x => x.remove());
          if (b.disabled) return;
          const s = this.renderList(it.sub!(), true);
          s.style.setProperty('top', `${b.offsetTop - 6}px`);
          ul.appendChild(s);
        });
      } else {
        b.addEventListener('pointerenter', () => ul.querySelectorAll('.menu__list--sub').forEach(x => x.remove()));
        b.addEventListener('click', () => { this.openMenu(-1); it.run?.(); });
      }
      ul.appendChild(b);
    }
    return ul;
  }

  /** Arrow keys in an open menu. Returns true if it took the key. */
  private menuKey(e: KeyboardEvent): boolean {
    if (this.openIdx < 0) return false;
    const list = $('menubar').querySelectorAll('.menu')[this.openIdx]?.querySelector('.menu__list');
    const items = list ? [...list.querySelectorAll<HTMLButtonElement>(':scope > .menu__item:not([disabled])')] : [];
    if (e.key === 'Escape') { this.openMenu(-1); return true; }
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      const n = this.menus.length;
      this.openMenu((this.openIdx + (e.key === 'ArrowRight' ? 1 : n - 1)) % n);
      return true;
    }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      if (!items.length) return true;
      this.hot = (this.hot + (e.key === 'ArrowDown' ? 1 : items.length - 1) + (this.hot < 0 && e.key === 'ArrowUp' ? 1 : 0)) % items.length;
      items[this.hot].focus();
      return true;
    }
    if (e.key === 'Enter' || e.key === ' ') {
      const f = document.activeElement as HTMLButtonElement | null;
      if (f && items.includes(f)) f.click();
      return true;
    }
    return false;
  }

  /* ---------------------------------------------------------------- keys */

  /** Does a key event match a binding like 'ctrl+shift+z', 'Numpad1' or 'key:+'? */
  private static matches(e: KeyboardEvent, bind: string): boolean {
    const parts = bind.split('+');
    let key = parts.pop()!;
    if (key === '' && bind.endsWith('+')) key = '+';       // 'key:+'
    const ctrl = parts.includes('ctrl'), shift = parts.includes('shift'), alt = parts.includes('alt');
    if (key.startsWith('key:')) {
      // characters whose key needs Shift on some layouts: ignore Shift
      return e.key === key.slice(4) && (e.ctrlKey || e.metaKey) === ctrl && e.altKey === alt;
    }
    if ((e.ctrlKey || e.metaKey) !== ctrl || e.shiftKey !== shift || e.altKey !== alt) return false;
    if (key.length === 1) return e.key.toLowerCase() === key;
    return e.code === key || e.key === key;
  }

  private wireKeys() {
    const typing = (t: EventTarget | null) => t instanceof HTMLElement
      && ((t.tagName === 'INPUT' && (t as HTMLInputElement).type !== 'range' && (t as HTMLInputElement).type !== 'checkbox')
          || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
    // Capture phase on the window: this runs before the older handler in
    // main.ts, and a key handled here stops there. That handler used to
    // send Ctrl+Z straight to the engine, past the selection history.
    window.addEventListener('keydown', (e) => {
      if (this.api.viewer.isNavigating) { e.stopPropagation(); return; }
      if (!$('help').hidden) {
        if (e.key === 'Escape' || e.key === 'F1') { this.showHelp(false); e.preventDefault(); }
        // the card is modal: Tab goes between its button and its list, never
        // to the page behind it
        if (e.key === 'Tab') {
          e.preventDefault();
          const stops = [$('help-close'), $('help-body')];
          const i = stops.indexOf(document.activeElement as HTMLElement);
          stops[(i + (e.shiftKey ? stops.length - 1 : 1)) % stops.length].focus();
        }
        e.stopPropagation();
        return;
      }
      if (this.menuKey(e)) { e.preventDefault(); e.stopPropagation(); return; }
      if (e.key === 'Alt') { e.preventDefault(); return; }     // no menu bar flash
      // A dropdown keeps the focus after a pick; it has no use for Ctrl+Z,
      // and letting that through sent it to the engine past the history.
      if (typing(e.target)
          && !((e.ctrlKey || e.metaKey) && (e.target as HTMLElement).tagName === 'SELECT')) return;
      if (e.key === 'Escape') {
        e.stopPropagation();
        if (this.api.selCount() > 0) this.api.deselect();
        return;
      }
      for (const c of this.allCommands()) {
        if (!c.bind?.some(b => Shell.matches(e, b))) continue;
        e.preventDefault();
        e.stopPropagation();
        // held keys repeat; one press is one step
        if (e.repeat && !(c.label === 'Increase Brush Size' || c.label === 'Decrease Brush Size')) return;
        if (c.enabled && !c.enabled()) return;
        c.run?.();
        return;
      }
      // Anything else with Ctrl stops here too, so the old handler never
      // sees a combination this list does not know.
      if (e.ctrlKey || e.metaKey) e.stopPropagation();
    }, true);
  }

  /* ------------------------------------------------------------ options */

  private wireOptions() {
    for (const b of document.querySelectorAll<HTMLButtonElement>('[data-selmode]')) {
      b.addEventListener('click', () => this.setSelMode(b.dataset.selmode as SelMode));
    }
  }

  setSelMode(m: SelMode) {
    this.selMode = m;
    for (const b of document.querySelectorAll<HTMLButtonElement>('[data-selmode]')) {
      b.classList.toggle('active', b.dataset.selmode === m);
    }
  }

  /** The active tool changed (main.ts calls this from setTool). */
  onTool(t: string) {
    $('opt-tool').textContent = TOOL_NAME[t] ?? t;
    // the stamp neither selects nor reaches into depth
    $('opt-selmode').hidden = t === 'clone';
    $('brush-depth-field').hidden = t === 'clone';
    $('nav-hint').textContent = navHint(this.api.viewer.preset, t);
  }

  /* ------------------------------------------------------------- panels */

  private wirePanels() {
    for (const d of document.querySelectorAll<HTMLDetailsElement>('details[data-panel]')) {
      const key = PANEL_KEY + d.dataset.panel;
      try {
        const v = localStorage.getItem(key);
        if (v !== null) d.open = v === '1';
      } catch { /* not persistent */ }
      d.addEventListener('toggle', () => {
        try { localStorage.setItem(key, d.open ? '1' : '0'); } catch { /* not persistent */ }
      });
    }
    $('btn-deselect').addEventListener('click', () => this.api.deselect());
    $('btn-frame-sel').addEventListener('click', () => this.api.frameSelection());
    $('btn-invert').addEventListener('click', () => this.api.history.labelNext('Invert Selection'), true);
  }

  private openPanel(id: string) {
    const d = $(id) as HTMLDetailsElement;
    if (d && 'open' in d) d.open = true;
    d?.scrollIntoView({ block: 'nearest' });
  }

  /* -------------------------------------------------------------- history */

  private renderHistory() {
    const h = this.api.history;
    const key = h.entries.length + ':' + h.index + ':' + h.entries.map(e => e.label).join('|');
    $<HTMLButtonElement>('btn-undo').disabled = !h.canUndo;
    $<HTMLButtonElement>('btn-redo').disabled = !h.canRedo;
    $('btn-undo').title = h.undoLabel ? `Undo: ${h.undoLabel} (Ctrl+Z)` : 'Undo (Ctrl+Z)';
    $('btn-redo').title = h.redoLabel ? `Redo: ${h.redoLabel} (Ctrl+Shift+Z)` : 'Redo (Ctrl+Shift+Z)';
    if (key === this.historyKey) return;
    this.historyKey = key;
    const ol = $('history-list');
    ol.textContent = '';
    if (!h.entries.length) {
      const li = document.createElement('li');
      li.className = 'hintline quiet';
      li.textContent = 'Nothing yet. Every selection and every change shows up here.';
      ol.appendChild(li);
      return;
    }
    h.entries.forEach((e, i) => {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.className = 'history__row' + (i === h.index ? ' current' : i > h.index ? ' future' : '');
      const l = document.createElement('span'); l.textContent = e.label;
      const k = document.createElement('span'); k.className = 'history__kind';
      k.textContent = e.kind === 'edit' ? 'Splats' : e.kind === 'selection' ? 'Selection' : 'File';
      b.append(l, k);
      b.title = i === h.index ? 'Current state' : i < h.index ? 'Go back to here' : 'Redo up to here';
      b.addEventListener('click', () => { void h.jumpTo(i); });
      li.appendChild(b);
      ol.appendChild(li);
    });
    // keep the current step in view inside the list only -- scrollIntoView
    // would scroll the whole panel and hide the repair buttons
    const cur = ol.querySelector<HTMLElement>('.current')?.parentElement;
    if (cur) {
      const top = cur.offsetTop - ol.offsetTop, bottom = top + cur.offsetHeight;
      if (top < ol.scrollTop) ol.scrollTop = top;
      else if (bottom > ol.scrollTop + ol.clientHeight) ol.scrollTop = bottom - ol.clientHeight;
    }
  }

  /* --------------------------------------------------------- navigation */

  private wireNav() {
    const v = this.api.viewer;
    $('btn-home').addEventListener('click', () => this.home());
    $('btn-frame-all').addEventListener('click', () => this.frameAll());
    $('btn-ortho').addEventListener('click', () => this.toggleOrtho());
    this.cube = new ViewCube($<HTMLCanvasElement>('viewcube'), v);
    this.cube.onSnap = (name, ortho) => this.toast(`${name}${ortho ? ' · orthographic' : ''}`);
    v.addFrameListener(() => this.tick());
  }

  home() {
    if (!this.api.hasDoc()) return;
    this.api.viewer.home();
    this.toast('Default View');
  }

  frameAll() {
    if (!this.api.hasDoc()) return;
    this.api.viewer.frameAll();
  }

  snap(dir: [number, number, number]) {
    if (!this.api.hasDoc()) return;
    this.api.viewer.snapTo(dir, true);
    this.toast(`${viewName(dir)} · orthographic`);
  }

  toggleOrtho() {
    if (!this.api.hasDoc()) return;
    const v = this.api.viewer;
    v.setOrtho(!v.isOrtho);
    this.toast(v.isOrtho ? 'Perspective' : 'Orthographic');
  }

  setPreset(p: NavPreset) {
    this.api.viewer.preset = p;
    try { localStorage.setItem(PRESET_KEY, p); } catch { /* not persistent */ }
    this.onTool(this.api.tool());
    this.toast(p === 'blender' ? 'Mouse like Blender' : 'Mouse like SuperSplat');
    if (!$('help').hidden) this.showHelp(true);
  }

  private fullscreen() {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void document.documentElement.requestFullscreen();
  }

  /** Cheap per-frame sync of the bits of UI that follow state changed elsewhere. */
  private tick() {
    const a = this.api, v = a.viewer;
    const has = a.hasDoc(), n = a.selCount();
    const s = `${has}|${n > 0}|${v.isOrtho}`;
    if (s === this.lastState) return;
    this.lastState = s;
    $<HTMLButtonElement>('btn-deselect').disabled = n === 0;
    $<HTMLButtonElement>('btn-frame-sel').disabled = n === 0;
    for (const id of ['btn-home', 'btn-frame-all', 'btn-ortho']) $<HTMLButtonElement>(id).disabled = !has;
    $('btn-ortho').classList.toggle('active', v.isOrtho);
    $('btn-ortho').textContent = v.isOrtho ? 'Ortho' : 'Persp';
    $('btn-ortho').title = v.isOrtho
      ? 'Orthographic. Click for perspective (Num 5)'
      : 'Perspective. Click for orthographic (Num 5)';
  }

  toast(msg: string) {
    const t = $('toast');
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => { t.hidden = true; }, 1600);
  }

  /* ------------------------------------------------------------- recent */

  private recent(): string[] {
    try { return JSON.parse(localStorage.getItem(RECENT_KEY) ?? '[]'); } catch { return []; }
  }

  /** A file opened: remember it for "Open Recent". */
  opened(path: string) {
    const list = [path, ...this.recent().filter(p => p !== path)].slice(0, 8);
    try { localStorage.setItem(RECENT_KEY, JSON.stringify(list)); } catch { /* not persistent */ }
  }

  /**
   * A file that is gone leaves "Open Recent": its entry used to stay and
   * fail again on every click. True if it was on the list.
   */
  forget(path: string): boolean {
    const list = this.recent();
    const next = list.filter(p => p !== path);
    if (next.length === list.length) return false;
    try { localStorage.setItem(RECENT_KEY, JSON.stringify(next)); } catch { /* not persistent */ }
    return true;
  }

  private recentItems(): Item[] {
    const list = this.recent();
    const items: Item[] = list.map(p => ({ label: p.split(/[\\/]/).pop() ?? p, keys: '', title: p,
                                           run: () => this.api.openPath(p) }));
    items.push('-', { label: 'Clear List', run: () => { try { localStorage.removeItem(RECENT_KEY); } catch { /* */ } } });
    return items;
  }

  /* ---------------------------------------------------------------- drop */

  /**
   * Drag a splat onto the window to open it. The overlay says what will
   * happen while the file is held over the window; the browser cannot tell
   * the name before the drop, so the check for the type comes then. Every
   * drop is taken here, whatever lands where: Electron's default is to
   * navigate to a dropped file, which would throw the document away.
   */
  private wireDrop() {
    let inside = 0;
    const hasFiles = (e: DragEvent) => !!e.dataTransfer && [...e.dataTransfer.types].includes('Files');
    window.addEventListener('dragenter', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      if (inside++ === 0) this.showDrop(true);
    });
    window.addEventListener('dragover', (e) => {
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = hasFiles(e) && !this.api.busy?.() ? 'copy' : 'none';
    });
    window.addEventListener('dragleave', (e) => {
      if (!hasFiles(e)) return;
      if (--inside <= 0) { inside = 0; this.showDrop(false); }
    });
    window.addEventListener('drop', (e) => {
      e.preventDefault();
      inside = 0;
      this.showDrop(false);
      const files = [...(e.dataTransfer?.files ?? [])];
      if (files.length) this.dropFiles(files);
    });
  }

  private showDrop(on: boolean) {
    const el = $('drop');
    if (!on) { el.hidden = true; return; }
    const busy = this.api.busy?.() ?? false;
    // The same test as the question that follows the drop (main.ts,
    // hasUnexportedEdits). "Any edit in the history" warned of a loss after
    // everything was exported, and after every edit had been undone.
    const unexported = this.api.unexported?.() ?? false;
    $('drop-title').textContent = busy ? 'Not right now' : 'Release to open';
    $('drop-line').textContent = busy
      ? 'Wait until the current task has finished.'
      : unexported
        ? 'PLY, SPZ or SPLAT. The open file will be closed. Anything not exported '
          + 'is lost. You will be asked first.'
        : 'PLY, SPZ or SPLAT.';
    el.classList.toggle('drop--busy', busy);
    el.hidden = false;
  }

  /** Where a dropped file is on disk ('' if not). Tests stand in for the OS here. */
  pathOf: (f: File) => string = (f) => window.host.pathForFile(f);

  /** Open the first splat among dropped files, or say why not. */
  dropFiles(files: File[]) {
    if (this.api.busy?.()) {
      this.api.status('not opened: a task is still running, drop the file again afterwards', 'err');
      return;
    }
    const f = files.find(x => SPLAT_FILE.test(x.name));
    if (!f) {
      this.api.status(`${files[0].name} cannot be opened. Only PLY, SPZ and SPLAT.`, 'err');
      return;
    }
    const p = this.pathOf(f);
    if (!p) { this.api.status(`${f.name} is not a file on disk`, 'err'); return; }
    if (files.length > 1) this.toast(`Only one file at a time: ${f.name}`);
    this.api.openPath(p);
  }

  /* --------------------------------------------------------------- help */

  private wireHelp() {
    $('help-close').addEventListener('click', () => this.showHelp(false));
    $('help').addEventListener('pointerdown', (e) => { if (e.target === $('help')) this.showHelp(false); });
  }

  showHelp(on: boolean) {
    const el = $('help');
    if (!on) { el.hidden = true; return; }
    const body = $('help-body');
    body.textContent = '';
    const section = (title: string, rows: Array<[string, string]>) => {
      const s = document.createElement('section');
      s.className = 'keys';
      const l = document.createElement('p'); l.className = 'label'; l.textContent = title;
      s.appendChild(l);
      for (const [what, keys] of rows) {
        const r = document.createElement('div'); r.className = 'keys__row';
        const a = document.createElement('span'); a.textContent = what;
        const k = document.createElement('kbd'); k.textContent = keys;
        r.append(a, k); s.appendChild(r);
      }
      body.appendChild(s);
    };
    const p = this.api.viewer.preset;
    section(p === 'blender' ? 'Mouse (like Blender)' : 'Mouse (like SuperSplat)', navRows(p));
    for (const m of this.menus) {
      const rows: Array<[string, string]> = [];
      for (const it of m.items()) {
        if (typeof it !== 'object' || !('label' in it) || !it.keys) continue;
        rows.push([typeof it.label === 'function' ? it.label().split(':')[0] : it.label, it.keys]);
      }
      if (m.title === 'File') rows.splice(1, 0, ['Open by dropping', 'Drag the file into the window']);
      if (m.title === 'Select') {
        const r = rows.find(x => x[0] === 'Deselect');
        if (r) r[1] += ' · Esc'; else rows.push(['Deselect', 'Esc']);
      }
      if (m.title === 'Tools') rows.push(['Clone Stamp: set source', 'Alt+click · Ctrl+click']);
      if (rows.length) section(m.title, rows);
    }
    el.hidden = false;
    $('help-close').focus();
  }
}
