/**
 * Electron main process: owns the window, the engine process, and the only
 * channel between them.
 *
 * Electron rather than Tauri on purpose. The renderer *is* the product here,
 * and it leans hard on WebGPU: pinning an exact Chromium buys deterministic
 * WGSL compilation and adapter limits, and means a webview update on the user's
 * machine can't change how splats look overnight.
 */
import { app, BrowserWindow, dialog, ipcMain, net, protocol, shell } from 'electron';
import path from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Sidecar } from './sidecar.js';
import { mirrorFolderOf, removeMirrorFolder } from './mirror.js';

// The renderer is served over http in dev and file:// in production, and
// neither origin may fetch an arbitrary file:// path. A privileged scheme gives
// the viewport one well-defined way to stream a splat off disk -- no 130 MB
// round trip through IPC, and nothing else on disk becomes reachable.
protocol.registerSchemesAsPrivileged([{
  scheme: 'app',
  privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true }
}]);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..', '..');
const CORE = path.join(ROOT, '..', 'core');
const DEV_URL = process.env.VITE_DEV_SERVER_URL;

let win: BrowserWindow | null = null;
const engine = new Sidecar(CORE);

/**
 * Send to the page, if there still is one. The engine's log and exit arrive
 * while the app is closing too, and `send` on a destroyed window throws.
 */
function toPage(channel: string, value: unknown) {
  if (!win || win.isDestroyed() || win.webContents.isDestroyed()) return;
  win.webContents.send(channel, value);
}

/* ------------------------------------------------- edits not yet exported */

/**
 * Does the page hold edits that were never exported? It is asked, because
 * only the page knows its history -- but never waited on for more than a
 * second: a page that hangs must not keep the window from closing.
 */
function hasUnexported(w: BrowserWindow): Promise<boolean> {
  return new Promise((resolve) => {
    const wc = w.webContents;
    if (w.isDestroyed() || wc.isDestroyed() || wc.isCrashed()) { resolve(false); return; }
    const done = (v: boolean) => {
      clearTimeout(timer);
      ipcMain.removeListener('doc:unexported', onReply);
      resolve(v);
    };
    const onReply = (e: Electron.IpcMainEvent, v: unknown) => { if (e.sender === wc) done(v === true); };
    const timer = setTimeout(() => done(false), 1000);
    ipcMain.on('doc:unexported', onReply);
    wc.send('doc:ask-unexported');
  });
}

/** "Discard" or "Cancel", as a native box. True when the edits may go. */
async function confirmDiscard(w: BrowserWindow, what: 'quit' | 'open'): Promise<boolean> {
  const r = await dialog.showMessageBox(w, {
    type: 'warning',
    title: 'Unexported Changes',
    message: 'You have changes that have not been exported.',
    detail: what === 'open'
      ? 'If you open another file now, they will be lost.'
      : 'If you close Splat Repair now, they will be lost.',
    buttons: ['Discard', 'Cancel'],
    defaultId: 1,
    cancelId: 1,
    noLink: true
  });
  return r.response === 0;
}

function createWindow() {
  win = new BrowserWindow({
    width: 1540,
    height: 980,
    minWidth: 900,
    minHeight: 600,
    backgroundColor: '#100c0a',
    show: false,
    autoHideMenuBar: true,
    title: 'Splat Repair',
    // Windows picks the size it needs out of the .ico; Linux wants a bitmap.
    icon: path.join(ROOT, 'assets',
                    process.platform === 'win32' ? 'icon.ico' : 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      // The preload needs nothing but contextBridge, ipcRenderer and webUtils,
      // which a sandboxed preload has.
      sandbox: true
    }
  });

  // The app draws its own menu bar (ui/shell.ts). Electron's default menu,
  // hidden behind autoHideMenuBar, popped up on every Alt press -- which is
  // the camera's modifier -- and its accelerators (Ctrl+R reloads the page,
  // losing the document) sat under the app's own shortcuts.
  win.removeMenu();
  win.webContents.on('before-input-event', (e, input) => {
    if (input.type === 'keyDown' && input.key === 'F12') {
      win?.webContents.toggleDevTools();
      e.preventDefault();
    }
  });
  win.once('ready-to-show', () => win?.show());
  const w = win;
  // Closing (Ctrl+Q, Alt+F4, the X, the app quitting) threw unexported edits
  // away without a word. Ask first -- except in the autotest, which must
  // never wait for a click.
  let mayClose = false, asking = false;
  w.on('close', (e) => {
    if (mayClose || process.env.SPLAT_AUTOTEST) return;
    e.preventDefault();
    if (asking) return;
    asking = true;
    void (async () => {
      try {
        mayClose = !(await hasUnexported(w)) || await confirmDiscard(w, 'quit');
      } catch {
        mayClose = true;            // a failed question never keeps the window open
      } finally {
        asking = false;
      }
      if (mayClose && !w.isDestroyed()) w.close();
    })();
  });
  w.on('closed', () => { if (win === w) win = null; });
  // The page's process can die (out of memory on a large scan, a GPU driver
  // reset). The window stayed, blank and dead. Say so and start the page again.
  w.webContents.on('render-process-gone', (_e, details) => {
    if (details.reason === 'clean-exit' || w.isDestroyed()) return;
    process.stderr.write(`[page] render process gone: ${details.reason}\n`);
    if (process.env.SPLAT_AUTOTEST) {
      process.stdout.write(`AUTOTEST FAILED: view crashed (${details.reason})\n`);
      stopEngine();
      app.exit(1);
      return;
    }
    void dialog.showMessageBox(w, {
      type: 'error',
      title: 'Splat Repair',
      message: 'The view crashed.',
      detail: 'The window will reload. Open your file again afterwards. '
        + 'Anything that was not exported is lost.',
      buttons: ['Reload'],
      noLink: true
    }).then(() => { if (!w.isDestroyed()) w.webContents.reload(); });
  });
  // A file dropped where the page does not take it would replace the app
  // with the file. The page takes every drop (ui/shell.ts); this is the
  // backstop, and nothing else in the app navigates.
  win.webContents.on('will-navigate', (e) => e.preventDefault());
  if (process.env.SPLAT_AUTOTEST) {
    win.webContents.once('did-finish-load', () => {
      autotest(win!, process.env.SPLAT_AUTOTEST!,
               process.env.SPLAT_AUTOTEST_OUT ?? path.join(ROOT, 'shots'));
    });
  }
  // (the message is on the event; the arguments after it are deprecated)
  win.webContents.on('console-message', (e) => {
    if (process.env.SPLAT_AUTOTEST) process.stderr.write(`[page] ${e.message}\n`);
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    // Only web links leave the app. openExternal hands whatever it is given
    // to the system, and for file: or a registered scheme that means running
    // the program behind it.
    let web = false;
    try { web = new URL(url).protocol === 'https:'; } catch { /* not a URL */ }
    if (web) void shell.openExternal(url);
    return { action: 'deny' };
  });

  if (DEV_URL) win.loadURL(DEV_URL);
  else win.loadURL('app://local/index.html');
}

/**
 * Files the viewport is allowed to stream: only ones the user actually opened.
 * Filled by the open dialog and by a successful engine `open` (below), never
 * by the page naming a path.
 */
const allowedFiles = new Set<string>();

const BUNDLE = path.join(ROOT, 'dist', 'renderer');

app.whenReady().then(() => {
  // Everything the page needs comes from one origin: the bundle under
  // app://local/, and opened splats under app://local/_file/. Same origin means
  // no CORS, and a file:// page could not fetch a custom scheme at all.
  protocol.handle('app', async (request) => {
    const url = new URL(request.url);
    const pathname = decodeURIComponent(url.pathname);

    if (pathname.startsWith('/_file/')) {
      const target = path.normalize(pathname.slice('/_file/'.length));
      if (!allowedFiles.has(target)) return new Response('not permitted', { status: 403 });
      return net.fetch(pathToFileURL(target).toString());
    }

    const rel = pathname === '/' ? 'index.html' : pathname.replace(/^\//, '');
    const file = path.normalize(path.join(BUNDLE, rel));
    const inside = path.relative(BUNDLE, file);
    if (inside.startsWith('..') || path.isAbsolute(inside)) return new Response('forbidden', { status: 403 });
    return net.fetch(pathToFileURL(file).toString());
  });
  createWindow();
  engine.on('log', (s: string) => {
    process.stderr.write(`[engine] ${s}`);
    toPage('engine:log', s);
  });
  engine.on('exit', (code: number) => toPage('engine:exit', code));
  engine.on('event', (e: Record<string, any>) => toPage('engine:event', e));
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

/**
 * The folder the engine keeps the viewport's PLY copy of a .splat or .spz in
 * (electron/mirror.ts), from the `viewPath` of the last successful `open`
 * that named one. An engine has one such folder for as long as it runs, so a
 * PLY opened afterwards (its `viewPath` is the file itself) does not make
 * this forget it.
 */
let mirrorFolder: string | null = null;

/**
 * End the engine, and remove its mirror folder: killed, it cannot do that
 * itself any more (sidecar.ts, stop).
 */
function stopEngine() {
  engine.stop();
  if (mirrorFolder) {
    removeMirrorFolder(mirrorFolder);
    mirrorFolder = null;
  }
}

// The engine holds the document in memory and has no business outliving the
// window; a stranded multi-hundred-MB python process is a real support cost.
app.on('window-all-closed', () => {
  stopEngine();
  if (process.platform !== 'darwin') app.quit();
});
app.on('before-quit', () => stopEngine());

/** Where the last save dialog pointed; the one path a `save` may write to. */
let savePath: string | null = null;

/**
 * The engine commands the page may send: the ones the app uses (every
 * `engine('…')` in src/main.ts). Any name used to be passed on, so a page
 * gone wrong could reach whatever the engine has or grows beside them.
 */
const PAGE_COMMANDS = new Set([
  'ping', 'capabilities', 'open', 'save',
  'repair', 'deep_repair', 'refine', 'unshadow', 'solidify',
  'delete', 'drop_nonfinite', 'transform', 'clone',
  'object_select', 'floaters', 'undo', 'redo'
]);
/** In the autotest also the engine's own counts, for a check to read. */
const AUTOTEST_COMMANDS = new Set(['stats']);

/**
 * Engine calls. Payload crosses as a transferable, not as JSON numbers.
 *
 * The page is trusted to edit the document, not to pick paths: the commands
 * that touch the disk are checked here, so a page gone wrong cannot have the
 * engine write or read where the user never pointed. (The autotest drives the
 * engine without dialogs and is left as it was.)
 */
ipcMain.handle('engine:call', async (_e, cmd: string, args: Record<string, any>,
                                     payload: ArrayBuffer | null) => {
  try {
    args = args ?? {};
    const auto = !!process.env.SPLAT_AUTOTEST;
    if (typeof cmd !== 'string'
        || !(PAGE_COMMANDS.has(cmd) || (auto && AUTOTEST_COMMANDS.has(cmd)))) {
      throw new Error(`The app does not send the command "${String(cmd).slice(0, 40)}" to the engine.`);
    }
    if (cmd === 'save' && !auto) {
      const target = typeof args.path === 'string' ? path.normalize(args.path) : '';
      if (!target || target !== savePath) {
        throw new Error('Export only writes to the location you chose in the dialog.');
      }
      savePath = null;            // one dialog, one file
    }
    if (cmd === 'deep_repair' && 'reference' in args && !auto) {
      // a picture file for the engine to read in place of its own fill:
      // a test hook, not something the app sends
      const { reference: _dropped, ...rest } = args;
      args = rest;
    }
    if (cmd === 'open' && typeof args.path === 'string' && path.isAbsolute(args.path)
        && !existsSync(args.path)) {
      // said here, in one sentence and with a kind the page can act on (it
      // takes the entry out of "Open Recent"), whatever the engine says
      return { header: { ok: false, kind: 'missing',
                         error: `${path.basename(args.path)} is no longer there.` },
               payload: null };
    }
    const reply = await engine.call(cmd, args,
      payload ? new Uint8Array(payload) : undefined);
    if (cmd === 'open' && reply.header.ok) {
      // The file the user opened, and the PLY the engine wrote of it for the
      // viewport, become readable through app://local/_file/. Decided here,
      // from the engine's own answer: the page used to name the paths itself.
      for (const p of [args.path, reply.header.viewPath]) {
        if (typeof p === 'string' && p) allowedFiles.add(path.normalize(p));
      }
      // The engine's mirror folder, to remove on quit. The engine itself
      // takes the previous copy out of it with every further open.
      mirrorFolder = mirrorFolderOf(reply.header.viewPath) ?? mirrorFolder;
    }
    const out = reply.payload.byteLength
      ? reply.payload.buffer.slice(reply.payload.byteOffset,
                                   reply.payload.byteOffset + reply.payload.byteLength)
      : null;
    return { header: reply.header, payload: out };
  } catch (err: any) {
    return { header: { ok: false, error: String(err?.message ?? err) }, payload: null };
  }
});

ipcMain.handle('dialog:openSplat', async () => {
  const r = await dialog.showOpenDialog(win!, {
    title: 'Open Splat',
    properties: ['openFile'],
    filters: [
      { name: 'Gaussian Splats', extensions: ['ply', 'splat', 'spz'] },
      { name: 'PLY', extensions: ['ply'] },
      { name: 'Niantic spz', extensions: ['spz'] },
      { name: 'splat', extensions: ['splat'] }
    ]
  });
  if (r.canceled) return null;
  allowedFiles.add(path.normalize(r.filePaths[0]));
  return r.filePaths[0];
});

/**
 * Before the page replaces a document that has unexported edits. (Closing the
 * window asks the same from here, see createWindow.) The autotest never waits
 * for a click.
 */
ipcMain.handle('dialog:confirmDiscard', async () => {
  if (process.env.SPLAT_AUTOTEST || !win || win.isDestroyed()) return true;
  return confirmDiscard(win, 'open');
});

/**
 * "Quit" in the page's menu (Ctrl+Q). Closed from here, so it goes through
 * the window's `close` event like the X and Alt+F4 do: `window.close()` in
 * the page does not, and took unexported edits with it unasked.
 */
ipcMain.on('window:close', (e) => {
  BrowserWindow.fromWebContents(e.sender)?.close();
});

ipcMain.handle('dialog:saveSplat', async (_e, suggested: string) => {
  const r = await dialog.showSaveDialog(win!, {
    title: 'Export Splat',
    defaultPath: suggested,
    // The extension decides the format on the engine side, so the filter the
    // user picks is the format they get.
    filters: [
      { name: 'PLY (lossless)', extensions: ['ply'] },
      { name: 'Niantic spz (small, all SH bands)', extensions: ['spz'] },
      { name: 'splat (for web viewers, no SH)', extensions: ['splat'] }
    ]
  });
  savePath = r.canceled || !r.filePath ? null : path.normalize(r.filePath);
  return r.canceled ? null : r.filePath;
});

/* ------------------------------------------------------------- automation */

/**
 * Real shadows baked into the test scans, where the shadow-brush step paints:
 * the band under the hut's balcony, the dark sole beside the shoe's stand.
 * World units; `dir` is the painting view, `radius` the stroke's.
 */
const SHADOW_SITES = [
  { file: 'hut', key: 'hut-balc90', target: [1.274, 0.372, 0.965], dir: [-0.9802, 0.135, 0.145], dist: 1.5997, radius: 0.03639,
    points: [[1.4983, 0.1476, 0.498], [1.4488, 0.1472, 0.568], [1.3961, 0.1468, 0.6449], [1.3593, 0.1465, 0.7293], [1.3246, 0.1462, 0.8195], [1.3016, 0.1459, 0.9141], [1.2469, 0.1455, 1.0189], [1.2535, 0.1454, 1.1174]] },
  { file: 'shoe', key: 'shoe-sole', target: [1.737, -4.592, 0.525], dir: [-0.01, -1.0, 0.0], dist: 18.0000, radius: 0.69899,
    points: [[6.8863, -0.7699, -0.0327], [6.54, -0.8258, -0.7696], [5.989, -0.8197, -1.3204], [5.5987, -0.7276, -1.6738]] },
  { file: 'brush', key: 'brush-sole', target: [1.737, -4.592, 0.525], dir: [-0.01, -1.0, 0.0], dist: 18.0000, radius: 0.69899,
    points: [[6.871, -0.7263, -0.031], [6.5684, -0.9134, -0.7775], [6.0919, -1.1794, -1.367], [5.6133, -0.7837, -1.6826]] },
];

/**
 * Headless-ish end-to-end run, so the app is covered by a test and not only the
 * engine beneath it. Enabled by env var, never reachable in a normal launch.
 */
async function autotest(w: BrowserWindow, ply: string, outDir: string) {
  const fs = await import('node:fs/promises');
  await fs.mkdir(outDir, { recursive: true });
  const log: string[] = [];
  const say = (s: string) => { log.push(s); process.stdout.write(s + '\n'); };
  const run = (js: string) => w.webContents.executeJavaScript(js, true);
  const shot = async (name: string) => {
    await new Promise(r => setTimeout(r, 450));
    const img = await w.webContents.capturePage();
    await fs.writeFile(path.join(outDir, name + '.png'), img.toPNG());
  };
  /** The viewport alone, so panel text does not count as a change. */
  const canvasShot = async (name: string): Promise<Uint8Array> => {
    await new Promise(r => setTimeout(r, 450));
    const b = await run(`(() => { const r = document.getElementById('canvas').getBoundingClientRect();
      return { x: Math.round(r.x), y: Math.round(r.y), width: Math.round(r.width), height: Math.round(r.height) }; })()`);
    const img = await w.webContents.capturePage(b);
    await fs.writeFile(path.join(outDir, name + '.png'), img.toPNG());
    return img.toBitmap();        // (getBitmap is its deprecated alias)
  };
  const compare = (a: Uint8Array, b: Uint8Array) => {
    if (a.length !== b.length) return 'different size';
    let diff = 0, sum = 0;
    for (let i = 0; i < a.length; i += 4) {
      const d = Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1])
              + Math.abs(a[i + 2] - b[i + 2]);
      sum += d;
      if (d > 24) diff++;
    }
    const px = a.length / 4;
    return `${(100 * diff / px).toFixed(2)}% of pixels differ, `
         + `mean difference ${(sum / px).toFixed(2)}/765`;
  };

  /** The shell (src/ui/): Home and ".", the view cube, the zoom, one history. */
  const shellChecks = async () => {
      // ---- the shell: navigation, view cube, one history (src/ui/) ----
      await run("window.__test.setDepth('quick')");
      await run('window.__test.clearSelection()');
      {
        // Home and "." from somewhere else: the bulk of the scan centred and in view
        for (const k of ['Home', 'Period']) {
          const h = await run(`window.__test.home('${k}')`);
          say(`${k === 'Home' ? 'Home' : 'Period'}:   ${JSON.stringify(h)}`);
          if (!h?.centred || !h?.fits) throw new Error(`${k}: scene not centred and fully in view`);
        }
        await shot('18-home');
        // the view cube: a face looks straight along it, orthographic; a corner
        // looks from in between, in perspective
        // (clicked from the standard view, where top and front both face the eye;
        // the corner from the front view, a little inside so it is on the face)
        const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        await run("window.__test.home('Home')");
        const top = await run('window.__test.cubeClick([0,1,0])');
        say(`cube top:   ${JSON.stringify(top)}`);
        if (!top || dot(top.fwd, top.up) > -0.999 || !top.ortho) throw new Error('cube "Top" does not look from above');
        await shot('19-cube-top');
        await run("window.__test.home('Home')");
        const front = await run('window.__test.cubeClick([0,0,1])');
        say(`cube front: ${JSON.stringify(front?.local)}`);
        if (!front || front.local[2] < 0.999 || !front.ortho) throw new Error('cube "Front" is wrong');
        const corner = await run('window.__test.cubeClick([0.85,0.85,1])');
        say(`cube corner: ${JSON.stringify(corner?.local)}`);
        if (!corner || dot(corner.local, [0.577, 0.577, 0.577]) < 0.995 || corner.ortho) {
          throw new Error('cube corner is wrong');
        }
        await shot('19b-cube-corner');
        // the wheel: each notch moves the eye the same share of the distance to
        // the surface under the cursor -- never past it, and never stalling
        await run("window.__test.home('Home')");
        const zin = await run('window.__test.wheel(30, -100)');
        const zout = await run('window.__test.wheel(8, 100)');
        const share = (z: any) => z.steps.filter((s: any) => s.depth).map((s: any) => s.moved / s.depth);
        const si = share(zin), so = share(zout);
        // Very close in, a sparse scan has too few splats in the pick window
        // and the zoom carries on towards the last surface it found: the last
        // depth that was measured counts.
        const known = zin.steps.filter((s: any) => s.depth);
        const d0 = known[0]?.depth, d1 = known[known.length - 1]?.depth;
        if (known.length < 15 || zin.steps.some((s: any) => !(s.moved > 0))) {
          throw new Error('zoom: surface not found, or a notch without movement: ' + JSON.stringify(zin.steps));
        }
        const fmt = (a: number[]) => a.length ? `${Math.min(...a).toFixed(3)}..${Math.max(...a).toFixed(3)}` : '–';
        say(`zoom:     in ${zin.steps.length} notches, share per notch ${fmt(si)}, `
            + `depth ${d0?.toFixed(3)} → ${d1?.toFixed(4)}; out share ${fmt(so)}`);
        const overshoot = zin.steps.some((s: any) => s.depth && s.moved >= s.depth);
        // a notch is 1 − 1/1.2 ≈ 0.165 of the depth; one where the surface
        // under the cursor changed between measuring and zooming may differ
        const med = si.slice().sort((a: number, b: number) => a - b)[si.length >> 1];
        if (!si.length || overshoot || Math.abs(med - 0.165) > 0.03 || Math.min(...si) < 0.08) {
          throw new Error('zoom not proportional to the distance: ' + JSON.stringify(si));
        }
        if (!(d1 > 0) || d1 > d0 * 0.2) throw new Error(`zoom stalls or passes through: ${d0} → ${d1}`);
        await shot('20-zoomed');
      }
      {
        // History: selection changes are steps, and undoing an edit brings
        // back the splats and the selection the edit was made from.
        await run("window.__test.home('Home')");
        await run('window.__test.clearSelection()');
        const p = (s: any) => `${s.count}/${s.hash}`;
        const s0 = await run('window.__test.selPrint()');
        await run("window.__test.box(0.42,0.42,0.58,0.58)");
        const s1 = await run('window.__test.selPrint()');
        await run("window.__test.box(0.30,0.30,0.70,0.70,'add')");
        const s2 = await run('window.__test.selPrint()');
        if (!s1.count || s2.count <= s1.count) throw new Error('box selected nothing: ' + JSON.stringify([s1, s2]));
        const u1 = await run('window.__test.historyUndo()');
        const u2 = await run('window.__test.historyUndo()');
        const r1 = await run('window.__test.historyRedo()');
        const r2 = await run('window.__test.historyRedo()');
        say(`selection: ${p(s0)} → ${p(s1)} → ${p(s2)}; undo ${p(u1)}, ${p(u2)}; redo ${p(r1)}, ${p(r2)}`);
        if (p(u1) !== p(s1) || p(u2) !== p(s0) || p(r1) !== p(s1) || p(r2) !== p(s2)) {
          throw new Error('undo does not restore the selection');
        }
        const alive0 = (await run('window.__test.stats()')).alive;
        const del = await run('window.__test.deleteSelection()');
        const afterDel = await run('window.__test.selPrint()');
        const ue = await run('window.__test.historyUndo()');
        const alive1 = (await run('window.__test.stats()')).alive;
        say(`delete:   ${alive0} → ${del.alive} splats, selection afterwards ${afterDel.count}; `
            + `undone: ${alive1} splats, selection ${p(ue)} (before ${p(s2)})`);
        if (alive1 !== alive0 || p(ue) !== p(s2)) throw new Error('undo after delete: splats or selection not back');
        const re = await run('window.__test.historyRedo()');
        const alive2 = (await run('window.__test.stats()')).alive;
        if (alive2 !== del.alive || re.count !== 0) throw new Error('redo after delete is wrong');
        await run('window.__test.historyUndo()');
        // the keys, through the menu shell: Ctrl+D drops it, Ctrl+Z brings it back
        await run("window.__test.key('d', 'KeyD', { ctrl: true })");
        const kd = await run('window.__test.selPrint()');
        await run("window.__test.key('z', 'KeyZ', { ctrl: true })");
        await new Promise(r => setTimeout(r, 300));
        const kz = await run('window.__test.selPrint()');
        say(`keys:     Ctrl+D → ${kd.count}, Ctrl+Z → ${p(kz)}`);
        if (kd.count !== 0 || p(kz) !== p(s2)) throw new Error('Ctrl+D / Ctrl+Z on the selection is wrong');
        say(`history:  ${JSON.stringify(await run('window.__test.historyState()'))}`);
        await shot('21-history');
        await run('window.__test.clearSelection()');
      }
      {
        // the new chrome, for looking at: a menu open, the brush's options, the help card
        await run("document.querySelectorAll('.menu__title')[3].click()");
        await shot('22-menu-ansicht');
        await run("document.querySelectorAll('.menu__title')[3].click()");
        await run("window.__test.key('b', 'KeyB')");
        await shot('23-brush-options');
        await run("window.__test.key('F1', 'F1')");
        await shot('24-help');
        await run("window.__test.key('Escape', 'Escape')");
        await run("window.__test.key('v', 'KeyV')");
      }

  };

  /**
   * The selection and its history (src/ui/history.ts, src/main.ts).
   *
   * 1. Lasso and box on the shoe take only the visible surface: from below,
   *    the sole and not the insole or the upper above it; from the side, that
   *    side and not the other. "Through" takes the back too.
   * 2. One run of selection steps and edits, its state written down after
   *    every step; then all the way back and all the way forward again, and
   *    every state on the way must be the one written down. The first undo
   *    is a real Ctrl+Z while a dropdown has the focus.
   */
  const selectionChecks = async (parts = 'all') => {
    await run("window.__test.upAxis('-y')");
    await run('window.__test.clearSelection()');
    const rc: number[] = await run(`(() => { const r = document.getElementById('canvas').getBoundingClientRect();
      return [r.left, r.top, r.width, r.height]; })()`);
    // an ellipse in canvas pixels, turned by `deg`, as viewport fractions
    const ell = (cx: number, cy: number, a: number, b: number, deg = 0) => {
      const t = deg * Math.PI / 180;
      return JSON.stringify(Array.from({ length: 40 }, (_, k) => {
        const u = k / 40 * 2 * Math.PI, x = a * Math.cos(u), y = b * Math.sin(u);
        return [+(cx + (x * Math.cos(t) - y * Math.sin(t)) / rc[2]).toFixed(4),
                +(cy + (x * Math.sin(t) + y * Math.cos(t)) / rc[3]).toFixed(4)];
      }));
    };
    if (parts !== 'history') {
      // ---- 1. only what is visible ----
      // the sole from straight below, the outline on the sole round the stand hole
      const SOLE = ell(0.487, 0.52, 300, 80, -28);
      // the sides: straight along the shoe's width axis, an outline along the
      // middle of the near wall
      const ax = await run('window.__test.scanAxes()');
      const dist = ax.length * 1.5;
      const aimAt = (d: number[]) => `window.__test.aim(${JSON.stringify(ax.centre)}, ${JSON.stringify(d)}, ${dist})`;
      const wpp = (await run(aimAt(ax.wide))).worldPerPixel;
      const SIDE = ell(0.5, 0.5, 0.32 * ax.length / wpp, 0.07 * ax.length / wpp, 0);
      const neg = ax.wide.map((v: number) => -v);
      say(`shoe:     length ${ax.length.toFixed(2)}, width ${ax.width.toFixed(2)}, side view from ${dist.toFixed(1)}`);
      const cases: Array<[string, string, string, string]> = [
        ['Sohle', 'window.__test.frame(0, -92, 1.9)', SOLE, 'window.__test.frame(0, 88, 1.9)'],
        ['Seite', aimAt(ax.wide), SIDE, aimAt(neg)],
        ['andere Seite', aimAt(neg), SIDE, aimAt(ax.wide)]
      ];
      // `name` goes into the output file names and stays as it is; `shown` is
      // what the log says
      const SHOWN: Record<string, string> = { Sohle: 'sole', Seite: 'side', 'andere Seite': 'other side' };
      for (const [name, cam, poly, back] of cases) {
        const shown = SHOWN[name] ?? name;
        const res: Record<string, any> = {};
        for (const reach of ['surface', 'through']) {
          await run(cam);
          await run('window.__test.settle()');
          const l = await run(`window.__test.lasso(${poly}, 'replace', '${reach}')`);
          const d = await run('window.__test.selectionDepth(0.5)');
          const k = await run('window.__test.volumeCheck()');
          res[reach] = { ...d, ms: l.ms, grid: l.surface, volume: k };
          // for the engine's reference (core/tests/test_volume.py, SPLAT_VOLUME_DUMP)
          await fs.writeFile(path.join(outDir, `volume-lasso-${reach}-${name.replace(/ /g, '-')}.json`),
                             JSON.stringify(await run('window.__test.volumeDump()')), 'utf-8');
          await shot(`s-${name}-${reach}`);
          await run(back);
          await run('window.__test.settle()');
          await shot(`s-${name}-${reach}-gegenseite`);
          if (k.ok === false) throw new Error(`volume (${shown}, ${reach}) selects differently from the selection`);
        }
        const s = res.surface, t = res.through;
        say(`lasso ${shown}: surface ${s.selected} splats in ${s.ms} ms (grid ${JSON.stringify(s.grid)}), `
            + `behind half depth ${s.far} (${(100 * s.farShare).toFixed(2)} %), depth ${JSON.stringify(s.depth)}, y ${JSON.stringify(s.y)}; `
            + `through ${t.selected}, behind half depth ${t.far} (${(100 * t.farShare).toFixed(1)} %)`);
        if (!s.selected) throw new Error(`lasso ${shown}: nothing selected`);
        if (s.farShare > 0.005) throw new Error(`lasso ${shown} takes the back side too: ${s.far} of ${s.selected}`);
        if (!(t.farShare > 0.3)) throw new Error(`lasso ${shown} through does not take the back side`);
        // from below the view runs up the shoe: the insole lies some 2 units
        // (0.15 of the height) above the bottom of the sole
        if (name === 'Sohle' && !(s.depth[3] < 0.15)) {
          throw new Error(`lasso sole reaches ${s.depth[3]} of the height, up to the insole`);
        }
      }
      // the box does the same
      await run('window.__test.frame(0, -92, 1.9)');
      await run('window.__test.settle()');
      const b = await run("window.__test.box(0.45,0.45,0.53,0.6)");
      const bd = await run('window.__test.selectionDepth(0.5)');
      say(`rectangle sole: ${b.selected} splats, behind half depth ${bd.far}`);
      if (!b.selected || bd.farShare > 0.005) throw new Error('rectangle takes the back side too');
      // Alt+Z switches the reach, and the buttons say so
      const reachNow = () => run(`document.querySelector('[data-reach].active')?.dataset.reach`);
      await run("window.__test.key('z', 'KeyZ', { alt: true })");
      const r1 = await reachNow();
      await run("window.__test.key('z', 'KeyZ', { alt: true })");
      const r2 = await reachNow();
      say(`Alt+Z:    ${r1} → ${r2}`);
      if (r1 !== 'through' || r2 !== 'surface') throw new Error('Alt+Z does not switch "Through"');
      await run('window.__test.clearSelection()');
    }
    if (parts !== 'lasso') {
      // ---- 2. every step back and forth ----
      await run("window.__test.setFill('patch')");      // no background refine in the middle
      await run('window.__test.frame(0, -92, 1.9)');
      await run('window.__test.settle()');
      const STROKE5 = '[[0.84,-0.63,-2.81],[2.27,-0.63,-1.06],[3.69,-0.63,0.69],[5.12,-0.63,2.44],[6.54,-0.63,4.19]]';
      const states: string[] = [await run('window.__test.selState()')];
      const names: string[] = ['start'];
      const h0 = (await run('window.__test.historyState()')).entries.length;
      // SPLAT_SEL_PAUSE=ms between steps, as a hand would leave them
      const pause = Number(process.env.SPLAT_SEL_PAUSE ?? 0);
      const step = async (name: string, js: string) => {
        if (pause) await new Promise(r => setTimeout(r, pause));
        await run(js);
        states.push(await run('window.__test.selState()'));
        names.push(name);
        if (states[states.length - 1] === states[states.length - 2]) {
          throw new Error(`step "${name}" changed nothing: ${states[states.length - 1]}`);
        }
      };
      await step('brush', `window.__test.paintWorld(${STROKE5}, 58, 1.5)`);
      // the slider, twice in one drag: one step of its own, after the stroke's
      await run('window.__test.slideDepth(0.6)');
      await step('depth', 'window.__test.slideDepth(0.9)');
      await run("(() => { const e = document.getElementById('brush-depth'); e.value = '1.5'; })()");
      await step('lasso add', `window.__test.lasso(${ell(0.6, 0.45, 120, 60)}, 'add')`);
      await step('rectangle subtract', "window.__test.box(0.55,0.40,0.62,0.50,'subtract')");
      await step('select all', "window.__test.key('a', 'KeyA', { ctrl: true })");
      await step('lasso', `window.__test.lasso(${ell(0.45, 0.55, 140, 70)})`);
      await step('invert', "window.__test.key('I', 'KeyI', { ctrl: true, shift: true })");
      await step('deselect', "window.__test.key('d', 'KeyD', { ctrl: true })");
      // made while the crop preview hides part of the shoe, then the preview off
      await run('window.__test.crop([0, 0, 0.5], [1, 1, 1])');
      await step('lasso with crop', `window.__test.lasso(${ell(0.487, 0.52, 300, 80, -28)})`);
      await run('window.__test.resetCrop()');
      await step('delete', 'window.__test.deleteSelection()');
      await step('lasso through', `window.__test.lasso(${ell(0.42, 0.6, 90, 60)}, 'replace', 'through')`);
      await step('lighten', "window.__test.unshadow('lift', 0.4)");
      await step('lasso 2', `window.__test.lasso(${ell(0.55, 0.45, 110, 60)})`);
      await step('repair', 'window.__test.repair()');
      const hs = await run('window.__test.historyState()');
      say(`history:  ${hs.entries.length - h0} new steps for ${names.length - 1}: ${JSON.stringify(hs.entries.slice(h0))}`);
      // back: the first one a real Ctrl+Z with the fill dropdown focused
      const n = names.length - 1;
      const bad: string[] = [];
      if (hs.entries.length - h0 !== n) bad.push(`history: ${hs.entries.length - h0} steps for ${n} changes`);
      await run(`(() => { const s = document.getElementById('deep-fill');
        for (let e = s.parentElement; e; e = e.parentElement) { if (e.tagName === 'DETAILS') e.setAttribute('open', ''); e.hidden = false; }
        s.focus(); return document.activeElement?.id; })()`);
      w.show(); w.focus();
      const focused = await run('document.activeElement?.id');
      // SPLAT_SEL_KEY=0: the first undo through the history too, not the key
      if (process.env.SPLAT_SEL_KEY === '0') await run('window.__test.historyUndo()');
      else {
        w.webContents.sendInputEvent({ type: 'keyDown', keyCode: 'Z', modifiers: ['control'] } as any);
        w.webContents.sendInputEvent({ type: 'keyUp', keyCode: 'Z', modifiers: ['control'] } as any);
      }
      for (let i = 0; i < 100 && (await run('window.__test.historyState()')).index !== hs.index - 1; i++) {
        await new Promise(r => setTimeout(r, 100));
      }
      let now = await run('window.__test.selState()');
      say(`Ctrl+Z in the dropdown (${focused}): ${now === states[n - 1] ? 'selection back' : 'WRONG ' + now}`);
      if (now !== states[n - 1]) bad.push(`Ctrl+Z in the dropdown: ${now} instead of ${states[n - 1]}`);
      await run('document.activeElement?.blur?.()');
      for (let k = n - 1; k >= 1; k--) {
        const u = await run('window.__test.historyUndo()');
        now = await run('window.__test.selState()');
        if (!u.ok || now !== states[k - 1]) bad.push(`undo over "${names[k]}": ${now} instead of ${states[k - 1]}`);
      }
      for (let k = 1; k <= n; k++) {
        const r = await run('window.__test.historyRedo()');
        now = await run('window.__test.selState()');
        if (!r.ok || now !== states[k]) bad.push(`redo over "${names[k]}": ${now} instead of ${states[k]}`);
      }
      say(`undo/redo: ${n} steps back and forth, ${bad.length} wrong`);
      for (const b of bad) say('  ' + b);
      await shot('s-history');
      if (bad.length) throw new Error('undo or redo does not restore the state: ' + bad[0]);
      await run('window.__test.undo()');                 // the repair off again
      await run('window.__test.clearSelection()');
      await run("window.__test.setFill('lama')");
    }
  };


  try {
    allowedFiles.add(path.normalize(ply));
    await new Promise(r => setTimeout(r, 1200));
    say('renderer: ' + JSON.stringify(await run('window.__test.stats()')));

    // Opened the way a user drags it in: a drop on the window (simulated --
    // the page gets the path from webUtils, here from the test). A file of
    // the wrong kind first: it must say so and open nothing.
    {
      const bad = await run(`window.__test.dropFiles(${JSON.stringify([path.join(path.dirname(ply), 'notes.txt')])})`);
      say(`drop txt: ${JSON.stringify({ overlay: bad.overlay, gone: bad.gone, splats: bad.splats, status: bad.status })}`);
      if (!bad.overlay || !bad.gone || bad.splats || !/err/.test(bad.statusKind)) {
        throw new Error('dropping a file of the wrong kind: ' + JSON.stringify(bad));
      }
    }
    const t0 = Date.now();
    const dropped = await run(`window.__test.dropFiles(${JSON.stringify([ply])})`);
    const opened = { splats: dropped.splats, alive: dropped.alive, path: dropped.path };
    say(`open:     dropped ${JSON.stringify(opened)} in ${((Date.now() - t0) / 1000).toFixed(1)}s, `
        + `hint "${dropped.line}", open recent ${JSON.stringify(dropped.recent)}`);
    if (!opened?.splats) throw new Error('no splats loaded: ' + JSON.stringify(await run('window.__test.stats()')));
    if (!dropped.overlay || !dropped.gone || path.normalize(dropped.path ?? '') !== path.normalize(ply)
        || dropped.recent !== dropped.path) {
      throw new Error('dropping does not open correctly: ' + JSON.stringify(dropped));
    }
    // Scans are y-down (3DGS/COLMAP), so the file opens with -Y up. Pin it:
    // a choice the user saved for this file must not change the test, and
    // every orbit angle below is in this frame.
    say(`up:       ${JSON.stringify(await run('window.__test.upAxis()'))}`);
    // The shadow brush and "Solidify" are paused: they show only with
    // "Beta Tools" ticked (Edit menu). Off, nothing of them may be
    // on the page, in a menu, on a key or in the help card. Then on for the
    // rest of the run, which uses both. Through the test hook, which stores
    // nothing: this run shares the user's profile, and their choice stays.
    {
      const off = await run('window.__test.beta(false)');
      say(`beta off: ${JSON.stringify(off)}`);
      if (off.on || off.panel || off.shown.length || off.menu.length || off.keys.length || off.help.length
          || !off.setting) {
        throw new Error('Beta Tools off, but Remove Shadow or Solidify are still there: ' + JSON.stringify(off));
      }
      await shot('00-beta-aus');
      const on = await run('window.__test.beta(true)');
      say(`beta on:  ${JSON.stringify(on)}`);
      if (!on.on || !on.panel || on.menu.length !== 2 || !on.menu.every((m: string) => /\(Beta\)$/.test(m))
          || !/Beta/.test(on.heading)) {
        throw new Error('Beta Tools on, but Remove Shadow or Solidify are missing or not marked as beta: '
                        + JSON.stringify(on));
      }
      await shot('00-beta-an');
    }
    // Run on a synthetic plane generated by test_server.make_ply.
    if (process.env.SPLAT_PROBE_GPU_BRUSH) {
      const check = async (label: string, x: number, radius: number,
                           mode: 'replace' | 'add' | 'subtract') => {
        const preview = await run(`window.__test.selectGpu([[${x},0,0]], ${radius}, 1.5, '${mode}')`);
        if (!preview.ok || !preview.capsules) throw new Error(`GPU brush unavailable: ${JSON.stringify(preview)}`);
        const gpu = await run('window.__test.gpuBrushMask()');
        await run('window.__test.selectCpu()');
        const cpu = await run('window.__test.selPrint()');
        say(`brush ${label}: GPU ${JSON.stringify(gpu)}, CPU ${JSON.stringify(cpu)}`);
        if (gpu.count !== cpu.count || gpu.hash !== cpu.hash) throw new Error(`brush parity failed: ${label}`);
        return gpu;
      };
      const base = await check('base', 0, 65, 'replace');
      if (!base.base || base.generated) throw new Error('expected a base-only selection');
      await run('window.__test.transform([0,0,0,1], [0.05,0,0])');
      const mixed = await check('mixed', 0, 100, 'replace');
      if (!mixed.base || !mixed.generated) throw new Error('expected both base and generated selections');
      await run('window.__test.clearSelection()');
      await run('window.__test.invert()');
      await run('window.__test.transform([0,0,0,1], [0.05,0,0])');
      const generated = await check('generated replace', 0, 60, 'replace');
      if (generated.base || !generated.generated) throw new Error('expected a generated-only selection');
      const more = await check('generated add', 0.8, 60, 'add');
      if (more.count <= generated.count) throw new Error('add should grow the selection');
      const less = await check('generated subtract', 0, 60, 'subtract');
      if (less.count >= more.count) throw new Error('subtract should shrink the selection');
      await run('window.__test.undo()');
      const undone = await check('mixed after undo', 0, 100, 'replace');
      if (!undone.base || !undone.generated) throw new Error('undo should restore the mixed selection case');
      await run('window.__test.clearSelection()');
      const failed = await run("window.__test.selectGpu([[0,0,0]], 100, 1.5, 'replace', true)");
      if (failed.ok || !failed.capsules) throw new Error('expected a failed generated pass');
      const fallback = await run('window.__test.gpuBrushMask()');
      const cpu = await run('window.__test.selPrint()');
      say(`brush fallback: streams ${JSON.stringify(fallback)}, CPU ${JSON.stringify(cpu)}`);
      if (!fallback.base || !fallback.generated || fallback.count !== cpu.count
          || fallback.hash !== cpu.hash || fallback.hash !== undone.hash) {
        throw new Error('CPU fallback must update both preview textures');
      }
      say('AUTOTEST OK (GPU brush)');
      app.quit();
      return;
    }
    // SPLAT_PROBE_VOLUME="[[x,y],...];r;yaw,pitch,dist": paint a stroke at
    // viewport fractions in the file's own up frame, and show the volume's
    // outline from the front and from two sides.
    if (process.env.SPLAT_PROBE_VOLUME) {
      const [pts, rad, cam] = process.env.SPLAT_PROBE_VOLUME.split(';');
      if (cam) await run(`window.__test.frame(${cam})`);
      await new Promise(r => setTimeout(r, 500));
      await shot('v0-before');
      say(`probe:    ${JSON.stringify(await run(`window.__test.paintStroke(${pts}, ${rad || 40}, 1.5)`))}`);
      say(`volume:   ${JSON.stringify(await run('window.__test.volume()'))}`);
      say(`check:    ${JSON.stringify(await run('window.__test.volumeCheck()'))}`);
      await shot('v1-front');
      await run('window.__test.outline(false)');
      await shot('v1-front-tint-only');
      await run('window.__test.outline(true)');
      const [yaw, pitch, dist] = (cam || '30,22,2.6').split(',').map(Number);
      await run(`window.__test.frame(${yaw + 50}, ${pitch}, ${dist})`);
      await shot('v2-side');
      await run(`window.__test.frame(${yaw - 70}, ${pitch + 25}, ${dist})`);
      await shot('v3-other-side');
      app.quit();
      return;
    }
    // SPLAT_PROBE_DEEPSTEP=n: the shadow step's lift and undo, then the deep
    // step's stroke, n times over -- the stroke that twice selected nothing
    if (process.env.SPLAT_PROBE_DEEPSTEP) {
      await run("window.__test.upAxis('-y')");
      const STROKE = `[[0.84,-0.63,-2.81],[2.27,-0.63,-1.06],[3.69,-0.63,0.69],[5.12,-0.63,2.44],[6.54,-0.63,4.19],[-0.4,-0.63,-1.8],[1.02,-0.63,-0.05],[2.45,-0.63,1.7],[3.88,-0.63,3.45],[5.3,-0.63,5.2],[-1.64,-0.63,-0.79],[-0.22,-0.63,0.96],[1.21,-0.63,2.71],[2.63,-0.63,4.46],[4.06,-0.63,6.21]]`;
      const site = SHADOW_SITES.find(s => path.basename(ply).toLowerCase().includes(s.file))!;
      const target = [0, 1, 2].map(k => site.points.reduce((a, p) => a + p[k], 0) / site.points.length);
      for (let it = 0; it < Number(process.env.SPLAT_PROBE_DEEPSTEP); it++) {
        await run('window.__test.frame(0, -92, 1.9)');
        const p0 = await run(`window.__test.paintWorld(${STROKE}, 58, 1.5)`);
        await run('window.__test.clearSelection()');
        const a = await run(`window.__test.aim(${JSON.stringify(target)}, ${JSON.stringify(site.dir)}, ${site.dist})`);
        const ps = await run(`window.__test.paintWorld(${JSON.stringify(site.points)}, ${site.radius / a.worldPerPixel}, 1.5)`);
        const au = await run("window.__test.unshadow('auto')");
        const li = au.info ? null : await run('window.__test.shadowInstead()');
        const un = await run('window.__test.undo()');
        await run("window.__test.unshadow('auto', 1)");
        await run('window.__test.clearSelection()');
        const wait = it % 3 === 0 ? 0 : it % 3 === 1 ? 400 : 1500;
        await run('window.__test.frame(0, -92, 1.9)');
        if (wait) await new Promise(r => setTimeout(r, wait));
        const p1 = await run(`window.__test.paintWorld(${STROKE}, 58, 1.5)`);
        const dg = await run('window.__test.paintDiag()');
        say(`deepstep ${it} (wait ${wait} ms): before ${p0.selected}, shadow ${ps.selected} `
            + `${au.info ? 'acted' : 'lightened ' + li?.info?.lifted}, undo ${JSON.stringify(un)}, after ${p1.selected}`);
        if (!p1.selected) say(`  diag: ${JSON.stringify(dg)}`);
        await run('window.__test.clearSelection()');
      }
      app.quit();
      return;
    }
    // SPLAT_PROBE_UI=1: every menu, panel, tool and overlay as a screenshot,
    // at the window's first size and maximised, for looking at the handling
    if (process.env.SPLAT_PROBE_UI) {
      await run("window.__test.upAxis('-y')");
      const key = (keyCode: string, modifiers: string[] = []) => {
        w.webContents.sendInputEvent({ type: 'keyDown', keyCode, modifiers } as any);
        w.webContents.sendInputEvent({ type: 'keyUp', keyCode, modifiers } as any);
      };
      const pass = async (tag: string) => {
        await run("window.__test.home('Home')");
        await shot(`ui-${tag}-00-start`);
        const n = await run("document.querySelectorAll('.menu__title').length");
        for (let i = 0; i < n; i++) {
          await run(`document.querySelectorAll('.menu__title')[${i}].click()`);
          if (i === 0) {
            await run(`(() => { const b = [...document.querySelectorAll('.menu__item--sub')][0];
              b?.dispatchEvent(new PointerEvent('pointerenter')); })()`);
          }
          await shot(`ui-${tag}-1${i}-menu`);
          await run(`document.querySelectorAll('.menu__title')[${i}].click()`);
        }
        for (const [t, k] of [['brush', 'b'], ['lasso', 'l'], ['box', 'r'], ['clone', 's']] as const) {
          await run(`window.__test.key('${k}', 'Key${k.toUpperCase()}')`);
          await shot(`ui-${tag}-2-tool-${t}`);
        }
        await run("window.__test.box(0.42,0.42,0.58,0.58)");
        await run("window.__test.key('g', 'KeyG')");
        await shot(`ui-${tag}-2-tool-xform`);
        await run("window.__test.key('v', 'KeyV')");
        await run(`(() => { for (const d of document.querySelectorAll('#panel details')) d.open = true;
          document.getElementById('panel').scrollTop = 0; })()`);
        await shot(`ui-${tag}-3-panel-top`);
        await run("(() => { const p = document.getElementById('panel'); p.scrollTop = p.scrollHeight / 2; })()");
        await shot(`ui-${tag}-3-panel-mid`);
        await run("(() => { const p = document.getElementById('panel'); p.scrollTop = p.scrollHeight; })()");
        await shot(`ui-${tag}-3-panel-bottom`);
        await run(`(() => { for (const d of document.querySelectorAll('#panel details.more, #cleanup-tile, #export-tile')) d.open = false;
          document.getElementById('panel').scrollTop = 0; })()`);
        await run("window.__test.key('F1', 'F1')");
        await shot(`ui-${tag}-4-help`);
        await run("window.__test.key('Escape', 'Escape')");
        await run(`(() => { const dt = new DataTransfer(); dt.items.add(new File([new Uint8Array(1)], 'x.ply'));
          document.getElementById('viewport').dispatchEvent(new DragEvent('dragenter', { dataTransfer: dt, bubbles: true })); })()`);
        await shot(`ui-${tag}-5-drop`);
        await run(`(() => { const dt = new DataTransfer(); dt.items.add(new File([new Uint8Array(1)], 'x.ply'));
          document.getElementById('viewport').dispatchEvent(new DragEvent('dragleave', { dataTransfer: dt, bubbles: true })); })()`);
        // keyboard focus, from the top of the page
        w.focus();
        await run('document.activeElement?.blur?.()');
        for (let k = 1; k <= 14; k++) {
          key('Tab');
          await new Promise(r => setTimeout(r, 60));
          if (k === 2 || k === 9 || k === 14) await shot(`ui-${tag}-6-focus-${k}`);
        }
        say(`ui ${tag}: ${JSON.stringify(await run(`(() => { const r = document.getElementById('canvas').getBoundingClientRect();
          return { w: innerWidth, h: innerHeight, canvas: [r.width, r.height] }; })()`))}`);
        await run('window.__test.clearSelection()');
      };
      await pass('default');
      w.maximize();
      await new Promise(r => setTimeout(r, 900));
      await pass('max');
      w.unmaximize();
      w.setSize(900, 600);
      await new Promise(r => setTimeout(r, 900));
      await pass('min');
      say('AUTOTEST OK (screenshots only)');
      app.quit();
      return;
    }
    // SPLAT_TEST_SELECTION=1: only the selection checks (lasso depth, history)
    if (process.env.SPLAT_TEST_SELECTION) {
      await selectionChecks(process.env.SPLAT_TEST_SELECTION);
      say('AUTOTEST OK (selection only)');
      app.quit();
      return;
    }
    // SPLAT_TEST_SHELL=1: only the shell checks, for working on the UI
    if (process.env.SPLAT_TEST_SHELL) {
      await run("window.__test.upAxis('-y')");
      if (process.env.SPLAT_PROBE_AIM) {
        const S = '[[0.84,-0.63,-2.81],[2.27,-0.63,-1.06],[3.69,-0.63,0.69],[5.12,-0.63,2.44],[6.54,-0.63,4.19],[-0.4,-0.63,-1.8],[1.02,-0.63,-0.05],[2.45,-0.63,1.7],[3.88,-0.63,3.45],[5.3,-0.63,5.2],[-1.64,-0.63,-0.79],[-0.22,-0.63,0.96],[1.21,-0.63,2.71],[2.63,-0.63,4.46],[4.06,-0.63,6.21]]';
        await run('window.__test.frame(0, -92, 1.9)');
        await new Promise(r => setTimeout(r, 400));
        say(`aim a: ${JSON.stringify(await run(`window.__test.paintWorld(${S}, 58, 1.5)`))} ${JSON.stringify(await run('window.__test.nav()'))}`);
        await run('window.__test.aim([0.5,0.5,0.5],[0,1,0.2],1.5)');
        await new Promise(r => setTimeout(r, 400));
        await run('window.__test.frame(0, -92, 1.9)');
        await new Promise(r => setTimeout(r, 400));
        say(`aim b: ${JSON.stringify(await run(`window.__test.paintWorld(${S}, 58, 1.5)`))} ${JSON.stringify(await run('window.__test.nav()'))}`);
      }
      await shellChecks();
      say('AUTOTEST OK (shell only)');
      return;
    }
    await run("window.__test.upAxis('-y')");
    // SPLAT_SIZE_K=0 runs the old point depth test, for before/after.
    if (process.env.SPLAT_SIZE_K) {
      say(`sizeK:    ${process.env.SPLAT_SIZE_K} (was ${JSON.stringify(await run(
        `window.__test.setSizeReach(${Number(process.env.SPLAT_SIZE_K)})`))})`);
    }
    await shot('01-loaded');
    say(`cropRange: ${JSON.stringify(await run('window.__test.cropRange()'))}`);
    say(`surface:  ${JSON.stringify(await run('window.__test.surfaceCoverage()'))}`);

    // Look at the sole, where the stand artifact is: from straight below.
    // (In the old +Y frame this was frame(0, 88) -- the same camera.)
    await run('window.__test.frame(0, -92, 1.9)');
    await shot('02-sole');

    // SPLAT_PROBE_REACH: delete what the stand stroke takes, at several
    // depths and with/without the size-aware test, and show what is left.
    if (process.env.SPLAT_PROBE_REACH) {
      const S = `[[0.84,-0.63,-2.81],[2.27,-0.63,-1.06],[3.69,-0.63,0.69],[5.12,-0.63,2.44],[6.54,-0.63,4.19],[-0.4,-0.63,-1.8],[1.02,-0.63,-0.05],[2.45,-0.63,1.7],[3.88,-0.63,3.45],[5.3,-0.63,5.2],[-1.64,-0.63,-0.79],[-0.22,-0.63,0.96],[1.21,-0.63,2.71],[2.63,-0.63,4.46],[4.06,-0.63,6.21]]`;
      const cfg = JSON.parse(process.env.SPLAT_PROBE_REACH.startsWith('[')
        ? process.env.SPLAT_PROBE_REACH : '[[0,1.5,1.5],[2,1.5,1.5],[0,1.5,3]]');
      for (const [k, cap, depth, what] of cfg) {
        await run(`window.__test.setSizeReach(${k}, ${cap})`);
        await run('window.__test.frame(0, -92, 1.9)');
        await new Promise(r => setTimeout(r, 300));
        const p = what === 'box'
          ? (await run(`window.__test.box(0.35,0.40,0.60,0.62)`), await run('window.__test.stats()'))
          : await run(`window.__test.paintWorld(${S}, 58, ${depth})`);
        const d = await run('window.__test.deleteSelection()');
        say(`probe k=${k} cap=${cap} depth=${depth}: selected ${p.selected}, ${JSON.stringify(d)}`);
        await shot(`p-k${k}-c${cap}-d${depth}${what ?? ''}`);
        await run('window.__test.frame(-38, -128, 2.1)');
        await shot(`p-k${k}-c${cap}-d${depth}${what ?? ''}-side`);
        await run('window.__test.undo()');
      }
      app.quit();
      return;
    }

    // SPLAT_PROBE_SHADOW=<sites.json>: paint over real shadows the way a
    // user would and record what the shadow brush sends, answers and shows.
    // The file maps a scan's name (hut, shoe, brush ...) to sites:
    // {key, target, dir, dist, points, radius} in world units.
    if (process.env.SPLAT_PROBE_SHADOW) {
      const all = JSON.parse(await fs.readFile(process.env.SPLAT_PROBE_SHADOW, 'utf-8'));
      const base = path.basename(ply).toLowerCase();
      const key = Object.keys(all).find(k => base.includes(k));
      const standCheck = async (when: string) => {
        await run('window.__test.frame(0, -92, 1.9)');
        await new Promise(r => setTimeout(r, 400));
        const S = '[[0.84,-0.63,-2.81],[2.27,-0.63,-1.06],[3.69,-0.63,0.69],[5.12,-0.63,2.44],[6.54,-0.63,4.19]]';
        const p = await run(`window.__test.paintWorld(${S}, 58, 1.5)`);
        say(`stand stroke ${when}: ${p.selected} selected, ${p.points.length} points in view`);
        await run('window.__test.clearSelection()');
      };
      await standCheck('before');
      for (const s of key ? all[key] : []) {
        const a = await run(`window.__test.aim(${JSON.stringify(s.target)}, ${JSON.stringify(s.dir)}, ${s.dist})`);
        await new Promise(r => setTimeout(r, 1200));
        const rpx = s.radius / a.worldPerPixel;
        const before = await canvasShot(`sh-${s.key}-0-before`);
        for (const mode of ['auto', 'lift'] as const) {
          const p = await run(`window.__test.paintWorld(${JSON.stringify(s.points)}, ${rpx}, 1.5)`);
          await shot(`sh-${s.key}-1-painted-${mode}`);
          {
            // does the selection hold after the stroke? (a selection that
            // empties itself would leave the button dead)
            const seen: number[] = [];
            for (const t of [0, 300, 700, 1500]) {
              await new Promise(r => setTimeout(r, t));
              seen.push((await run('window.__test.stats()')).selected);
            }
            say(`  selection after the stroke: ${seen.join(' → ')}`);
          }
          const r = await run(`window.__test.unshadow('${mode}')`);
          await run('window.__test.clearSelection()');
          const after = await canvasShot(`sh-${s.key}-2-${mode}`);
          // the same view once more after a camera nudge: a viewer that does
          // not redraw edited splats shows the change only now
          await run(`window.__test.aim(${JSON.stringify(s.target)}, ${JSON.stringify(s.dir)}, ${s.dist * 1.0001})`);
          const nudged = await canvasShot(`sh-${s.key}-3-${mode}-nudged`);
          await run(`window.__test.aim(${JSON.stringify(s.target)}, ${JSON.stringify(s.dir)}, ${s.dist})`);
          say(`shadow ${s.key} ${mode}: aim ${a.err.toFixed(3)}, r ${rpx.toFixed(1)} px, `
              + `selected ${p.selected}, picture ${compare(before, after)}; after nudge ${compare(before, nudged)}`);
          say(`  note:   ${r.note}`);
          say(`  status: ${r.status}`);
          say(`  info:   ${JSON.stringify(r.info)}`);
          if (r.info) await run('window.__test.undo()');
        }
      }
      await standCheck('after');
      // lift, Ctrl+Z, then straight on with the next stroke, as a user would
      for (const s of key ? all[key] : []) {
        const a = await run(`window.__test.aim(${JSON.stringify(s.target)}, ${JSON.stringify(s.dir)}, ${s.dist})`);
        await new Promise(r => setTimeout(r, 800));
        const rpx = s.radius / a.worldPerPixel;
        const p1 = await run(`window.__test.paintWorld(${JSON.stringify(s.points)}, ${rpx}, 1.5)`);
        await run(`window.__test.unshadow('auto')`);
        const r1 = await run(`window.__test.shadowInstead()`);
        await run('window.__test.clearSelection()');
        say(`  sequence: selected ${p1.selected}, lifted ${r1.info?.lifted}, selection afterwards ${r1.selected}`);
        await run('window.__test.undo()');
        say(`  after Ctrl+Z: ${JSON.stringify(await run('window.__test.stats()'))}`);
        await run("window.__test.unshadow('auto', 1)");
        await run('window.__test.clearSelection()');
        await run('window.__test.frame(0, -92, 1.9)');
        await new Promise(r => setTimeout(r, 400));
        const S2 = '[[0.84,-0.63,-2.81],[2.27,-0.63,-1.06],[3.69,-0.63,0.69],[5.12,-0.63,2.44],[6.54,-0.63,4.19]]';
        const p2 = await run(`window.__test.paintWorld(${S2}, 58, 1.5)`);
        say(`  stand stroke right after: ${p2.selected}`);
        await new Promise(r => setTimeout(r, 500));
        say(`  0.5 s later: ${(await run('window.__test.stats()')).selected}`);
        await run('window.__test.clearSelection()');
        await run("window.__test.unshadow('auto', 1)");
        await standCheck('after sequence');
      }
      app.quit();
      return;
    }

    if (process.env.SPLAT_PROBE_STATE) {
      const pr = await run('window.__test.probeState()');
      say(`probeState: ${JSON.stringify(pr)}`);
      await shot('02b-state');
      await run('window.__test.forceTint()');
      await shot('02a-tintall');
      app.quit();
      return;
    }

    // A stroke over the stand artifact in the arch, aimed in world space so it
    // stays on target whatever the framing. Covering the haze around the dark
    // core matters as much as the core: leave it and the repair is clean
    // geometry under a grey smear.
    // Points measured from the stroke that was validated against the engine
    // directly, so the app and the engine are being asked to do the same job.
    const STROKE = `[[0.84,-0.63,-2.81],[2.27,-0.63,-1.06],[3.69,-0.63,0.69],[5.12,-0.63,2.44],[6.54,-0.63,4.19],[-0.4,-0.63,-1.8],[1.02,-0.63,-0.05],[2.45,-0.63,1.7],[3.88,-0.63,3.45],[5.3,-0.63,5.2],[-1.64,-0.63,-0.79],[-0.22,-0.63,0.96],[1.21,-0.63,2.71],[2.63,-0.63,4.46],[4.06,-0.63,6.21]]`;
    const paint = await run(`window.__test.paintWorld(${STROKE}, 58, 1.5)`);
    say(`painted: ${JSON.stringify(paint)}`);
    if (!paint.selected) throw new Error('brush selected nothing');
    say(`selection: ${JSON.stringify(await run('window.__test.selectionStats()'))}`);
    say(`timing: ${JSON.stringify(await run(`window.__test.timeSelect(${STROKE}, 58, 12)`))}`);
    // Show each path's answer on its own, then compare the pictures: two
    // implementations of one rule drift silently otherwise.
    // (the volume outline off: the pictures compare the tint, nothing else)
    await run('window.__test.outline(false)');
    const g = await run(`window.__test.selectGpu(${STROKE}, 58, 1.5)`);
    say(`gpuSelect: ${JSON.stringify(g)}`);
    // the viewport only: the history panel gains a line between the shots
    const imgGpu = await canvasShot('03b-gpu');
    const c = await run('window.__test.selectCpu()');
    say(`cpuSelect: ${JSON.stringify(c)}`);
    const imgCpu = await canvasShot('03c-cpu');
    say(`agreement: ${compare(imgGpu, imgCpu)}`);
    await run('window.__test.outline(true)');
    // Known good, measured from the stroke validated directly against the
    // engine: centroid [2.53,-0.96,1.93], x 5-95% -0.63..5.60, z -2.21..5.45.
    await shot('03-selected');

    // The painted volume. Worked out again from scratch it must select
    // exactly what the selection holds; the dump lets the engine's reference
    // (core/tests/test_volume.py, SPLAT_VOLUME_DUMP) check the same.
    const dumpVolume = async (name: string) => {
      const d = await run('window.__test.volumeDump()');
      await fs.writeFile(path.join(outDir, `volume-${name}.json`), JSON.stringify(d), 'utf-8');
    };
    const checkVolume = async (what: string) => {
      const v = await run('window.__test.volume()');
      const k = await run('window.__test.volumeCheck()');
      say(`volume ${what}: ${JSON.stringify(v)} · check ${JSON.stringify(k)}`);
      if (k.ok === false) throw new Error(`volume (${what}) selects differently from the selection`);
      return v;
    };
    await checkVolume('brush');
    await dumpVolume('brush');
    {
      const s = await run('window.__test.snapshotRoundTrip()');
      say(`snapshot: ${JSON.stringify(s)}`);
      if (!s.same) throw new Error('selection differs after snapshot and restore');
      // un-paint the first row of the stroke, paint its middle again: the
      // volume keeps the order, newest wins
      const S = JSON.parse(STROKE) as number[][];
      const sub = await run(`window.__test.paintWorldMode(${JSON.stringify(S.slice(0, 3))}, 40, 1.5, 'subtract')`);
      await checkVolume('minus');
      await shot('03v-subtract');
      const again = await run(`window.__test.paintWorldMode(${JSON.stringify([S[1]])}, 30, 1.5, 'add')`);
      await checkVolume('again');
      say(`volume order: ${paint.selected} → minus ${sub.selected} → again ${again.selected}`);
      if (!(sub.selected < paint.selected) || !(again.selected > sub.selected)) {
        throw new Error('subtracting and painting again on the volume: ' + JSON.stringify([paint.selected, sub.selected, again.selected]));
      }
      const s2 = await run('window.__test.snapshotRoundTrip()');
      say(`snapshot: ${JSON.stringify(s2)}`);
      if (!s2.same) throw new Error('snapshot with subtract differs');
      await dumpVolume('order');
      await run(`window.__test.paintWorldMode(${STROKE}, 58, 1.5, 'replace')`);
      await shot('03v-volume');
      await run('window.__test.frame(-38, -128, 2.1)');
      await shot('03v-volume-side');
      await run('window.__test.frame(0, -92, 1.9)');
      await new Promise(r => setTimeout(r, 300));
    }

    // The outline while the brush is still down, painted with real mouse
    // input through the window as a hand would; and what the scan hides of
    // it drawn faint, seen from the far side.
    {
      await run('window.__test.clearSelection()');
      await run("window.__test.key('b', 'KeyB')");
      await run('window.__test.frame(0, -92, 1.9)');
      await run('window.__test.settle()');
      const pts: number[][] = await run(`(() => {
        const r = document.getElementById('canvas').getBoundingClientRect();
        return ${STROKE}.slice(0, 5).map(p => { const s = window.__test.toScreen(p[0], p[1], p[2]);
          return [r.left + s.fx * r.width, r.top + s.fy * r.height]; }); })()`);
      await run("(() => { const e = document.getElementById('brush-size'); e.value = '116'; e.dispatchEvent(new Event('input')); })()");
      const pos = (k: number) => {
        const t = k / 40 * (pts.length - 1), i = Math.min(pts.length - 2, Math.floor(t)), f = t - i;
        return { x: Math.round(pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f),
                 y: Math.round(pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f) };
      };
      const live0 = (await run('window.__test.outlineState()')).liveBuilds;
      // The live outline and the brush's own preview come with rendered
      // frames. A test window behind another one (two runs side by side)
      // got two frames in the whole stroke: in front first, and drawing.
      w.show(); w.moveTop(); w.focus();
      await run('window.__test.frameTimes(true)');
      for (let k = 0; k < 30; k++) {
        await new Promise(r => setTimeout(r, 100));
        const f = await run('window.__test.frameTimes(null)');
        if (f.frames >= 5) break;
        if (k === 29) throw new Error('the window does not draw, so the live outline cannot be checked');
      }
      await run('window.__test.frameTimes(true)');
      w.webContents.sendInputEvent({ type: 'mouseMove', ...pos(0) });
      w.webContents.sendInputEvent({ type: 'mouseDown', ...pos(0), button: 'left', clickCount: 1 });
      let mid: any = null;
      for (let k = 1; k <= 40; k++) {
        w.webContents.sendInputEvent({ type: 'mouseMove', ...pos(k), button: 'left', modifiers: ['leftButtonDown'] } as any);
        await new Promise(r => setTimeout(r, 18));
        if (k === 28) {
          await new Promise(r => setTimeout(r, 450));      // the throttle's turn, and a frame
          mid = await run('window.__test.outlineState()');
          await shot('03w-live-outline');
        }
      }
      w.webContents.sendInputEvent({ type: 'mouseUp', ...pos(40), button: 'left', clickCount: 1 });
      // and off the viewport again: Chromium replays the last mouse position
      // as a move after layout changes, and the stamp's hover marker later
      // followed this one
      const wh = await run('innerHeight');
      w.webContents.sendInputEvent({ type: 'mouseMove', x: 4, y: wh - 4 });
      await new Promise(r => setTimeout(r, 300));
      const ft = await run('window.__test.frameTimes(false)');
      const end = await run('window.__test.outlineState(true)');
      say(`live:     mid-stroke ${JSON.stringify(mid)}; afterwards ${JSON.stringify(end)}; frames ${JSON.stringify(ft)}`);
      if (!mid?.painting || !mid.live || !(mid.segments > 0) || mid.liveBuilds - live0 < 2) {
        throw new Error('outline does not appear during the stroke: ' + JSON.stringify(mid));
      }
      if (end.painting || end.live || !(end.selected > 0) || !(end.segments > 0)) {
        throw new Error('outline after the stroke is wrong: ' + JSON.stringify(end));
      }
      if (!ft || !(ft.median <= 40)) throw new Error('painting stutters with the outline: ' + JSON.stringify(ft));
      // from where it was painted most of the outline is in view; from the
      // other side the scan stands in front of part of it (the shoe of
      // nearly all, the hut's sand and the castle's ground of a third: the
      // shoe's stroke runs off their edge)
      const front = end.hidden / Math.max(1, end.segments);
      await run('window.__test.frame(0, 88, 1.9)');
      const above = await run('window.__test.outlineState(true)');
      await shot('03w-hidden-above');
      await run('window.__test.frame(-38, -128, 2.1)');
      const side = await run('window.__test.outlineState(true)');
      await shot('03w-hidden-side');
      const back = above.hidden / Math.max(1, above.segments);
      say(`hidden:   from the front ${(100 * front).toFixed(0)} %, from above ${(100 * back).toFixed(0)} %, `
          + `from the side ${(100 * side.hidden / Math.max(1, side.segments)).toFixed(0)} % of the edge`);
      if (front > 0.25 || back < Math.max(0.25, front + 0.2)) throw new Error('hidden part of the outline detected wrongly: '
                                                      + JSON.stringify({ end, above }));
      await run("(() => { const e = document.getElementById('brush-size'); e.value = '60'; e.dispatchEvent(new Event('input')); })()");
      await run('window.__test.clearSelection()');
      await run('window.__test.frame(0, -92, 1.9)');
      await new Promise(r => setTimeout(r, 300));
    }

    // What the size-aware depth test would add over the point test (it is
    // off, SIZE_K = 0, so this asks for k = 3, the value it was tuned at):
    // large splats behind the surface that still cover it. It may only add.
    // Shown on their own, with everything else hidden, from the front and
    // from the side.
    const reach = await run(`window.__test.reachDiff(${STROKE}, 58, 1.5, 3)`);
    say(`reach:    ${JSON.stringify(reach)}`);
    if (reach.after < reach.before) throw new Error('size-aware reach removed splats');
    await run('window.__test.isolate(true)');
    await shot('03a-reach-added');
    await run('window.__test.frame(-38, -128, 2.1)');
    await shot('03a-reach-added-side');
    await run('window.__test.isolate(false)');
    await run('window.__test.frame(0, -92, 1.9)');
    await new Promise(r => setTimeout(r, 300));
    const p0 = await run(`window.__test.paintWorld(${STROKE}, 58, 1.5)`);

    // The view a selection was made from is the one the brushes send, not
    // the camera when the button is pressed; and the depth slider after a
    // stroke redoes that stroke, exactly, from that view.
    const v0 = await run('window.__test.views()');
    await run('window.__test.frame(-38, -128, 2.1)');
    await new Promise(r => setTimeout(r, 300));
    const v1 = await run('window.__test.views()');
    const same = (a: number[], b: number[]) => a.every((x, i) => Math.abs(x - b[i]) < 1e-6);
    if (!v1.selView || !same(v1.selView, v0.selView) || same(v1.now, v0.selView)) {
      throw new Error('the view direction of the selection follows the camera: ' + JSON.stringify([v0, v1]));
    }
    const dShallow = await run('window.__test.slideDepth(0.5)');
    const dBack = await run('window.__test.slideDepth(1.5)');
    say(`views:    painted ${JSON.stringify(v0.selView.map((x: number) => +x.toFixed(3)))}, `
        + `now ${JSON.stringify(v1.now.map((x: number) => +x.toFixed(3)))}`);
    say(`depth:    1.5 → ${p0.selected}, 0.5 → ${dShallow.selected}, back to 1.5 → ${dBack.selected}`);
    if (dShallow.selected > p0.selected || dBack.selected !== p0.selected) {
      throw new Error('depth slider after the stroke: ' + JSON.stringify([p0.selected, dShallow, dBack]));
    }
    await run('window.__test.frame(0, -92, 1.9)');
    await new Promise(r => setTimeout(r, 300));

    // transform: move the selection and put it back
    await run("window.__test.box(0.35,0.40,0.60,0.62)");
    say(`gizmo:    ${JSON.stringify(await run('window.__test.gizmoState()'))}`);
    const xf = await run('window.__test.transform([0.2588,0,0,0.9659],[2,0,0],1.1)');
    say(`xform:    ${JSON.stringify(xf)}`);
    await shot('16-transform');
    say(`undo:     ${JSON.stringify(await run('window.__test.undo()'))}`);
    await run('window.__test.clearSelection()');

    // Clone stamp, as a user would: Alt + click a source on the surface,
    // then paint a short stroke beside it. Placed on screen, not in world
    // units, so it runs on any file: the shoe's coordinates used to be sent
    // to every object, and on the hut the stamp found no surface and the
    // test still passed. Checked: the new splats land under the stroke, they
    // carry the source's colour rather than the target's (tone match off),
    // the crosshair sits on the source, and one undo takes the stroke back.
    {
      let plan: { fx: number; fy: number; pts: number[][] } | null = null;
      for (const [fx, fy] of [[0.40, 0.50], [0.40, 0.45], [0.40, 0.55], [0.34, 0.50],
                              [0.44, 0.40], [0.44, 0.60], [0.30, 0.45], [0.30, 0.55]]) {
        const pts: number[][] = [];
        for (let k = 0; k < 3; k++) pts.push(await run(`window.__test.pick(${fx + 0.025 * k}, ${fy})`));
        const src = await run(`window.__test.pick(${fx + 0.14}, ${fy})`);
        if (pts.every(Boolean) && src) { plan = { fx, fy, pts }; break; }
      }
      if (!plan) throw new Error('clone stamp: no surface found for source and target');
      const g0 = (await run('window.__test.stats()')).generated;
      const src = await run(`window.__test.setCloneSourceAtScreen(${plan.fx + 0.14}, ${plan.fy})`);
      const cs = await run('window.__test.cloneState()');
      const vp = await run('window.__test.viewportSize()');
      say(`cloneSrc: ${JSON.stringify(src.source?.map((v: number) => +v.toFixed(3)))}, crosshair ${JSON.stringify(cs.marker)}`);
      if (!src.source || !cs.marker
          || Math.abs(cs.marker[0] - (plan.fx + 0.14) * vp.width) > 4
          || Math.abs(cs.marker[1] - plan.fy * vp.height) > 4) {
        throw new Error('clone stamp: source or crosshair is wrong: ' + JSON.stringify(cs));
      }
      // Before painting the brush shows the source; while painting, the dabs.
      const hv = await run(`window.__test.hoverClone(${plan.fx}, ${plan.fy})`);
      if (!hv.preview?.peek) throw new Error('clone stamp: no preview of the source in the brush');
      await shot('15a-clone-before');
      const st = await run(`window.__test.stampStroke(${JSON.stringify(plan.pts)}, 28, 0)`);
      const i = st.info ?? {};
      const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
      say(`stamp:    ${st.dabs} dabs, +${st.generated - g0}, landed ${i.landed}, `
          + `colour new ${JSON.stringify(i.newRgb)} source ${JSON.stringify(i.srcRgb)} `
          + `before ${JSON.stringify(i.oldRgb)} · preview ${st.preview?.dabs} dabs · ${st.status}`);
      if (!(st.generated > g0)) throw new Error('clone stamp set nothing: ' + st.status);
      if (!(st.preview?.dabs >= 1 && st.preview.coverage > 0)) {
        throw new Error('clone stamp: dabs not shown while painting: ' + JSON.stringify(st.preview));
      }
      if (!(i.landed >= 0.9)) throw new Error('clone stamp does not land under the stroke: ' + i.landed);
      if (i.newRgb && i.srcRgb && i.oldRgb && dist(i.srcRgb, i.oldRgb) > 0.05
          && dist(i.newRgb, i.srcRgb) >= dist(i.newRgb, i.oldRgb)) {
        throw new Error('clone stamp does not copy from the source: ' + JSON.stringify(i));
      }
      await shot('15-clone');
      const u = await run('window.__test.undo()');
      say(`undo:     ${JSON.stringify(u)}`);
      if (u.generated !== g0) throw new Error('clone stamp: undo left splats behind');
      await run("(() => { document.getElementById('clone-match').value = '0.6'; })()");
    }

    // cleanup: debris detection and the crop box
    say(`floaters: ${JSON.stringify(await run('window.__test.floaters(0.3)'))}`);
    await shot('12-floaters');
    say(`invert:   ${JSON.stringify(await run('window.__test.invert()'))}`);
    say(`crop:     ${JSON.stringify(await run('window.__test.crop([0.15,0,0.15],[0.85,1,0.85])'))}`);
    await shot('13-crop-preview');
    // What the crop preview hides cannot be selected: the brush, lasso and
    // box all see the same visible set the GPU preview shows.
    {
      await run('window.__test.clearSelection()');
      await run(`window.__test.paintWorld(${STROKE}, 58, 1.5)`);
      const cb = await run('window.__test.cropCheck()');
      await run("window.__test.box(0.05,0.05,0.95,0.95,'replace','through')");
      const cx = await run('window.__test.cropCheck()');
      say(`crop+sel: brush ${JSON.stringify(cb)}, box ${JSON.stringify(cx)}`);
      if (cb.selectedHidden || cx.selectedHidden) throw new Error('selected what the crop hides');
      await run('window.__test.clearSelection()');
    }
    const cr = await run('window.__test.commitCrop()');
    say(`cropped:  ${JSON.stringify(cr)}`);
    await shot('14-cropped');
    say(`undo:     ${JSON.stringify(await run('window.__test.undo()'))}`);
    // The preview follows the document after an undo, too.
    {
      const before = await run('window.__test.crop([0.15,0,0.15],[0.85,1,0.85])');
      await run(`window.__test.paintWorld(${STROKE}, 58, 1.5)`);
      await run('window.__test.deleteSelection()');
      await run('window.__test.undo()');
      const after = await run('window.__test.cropCheck()');
      say(`crop/undo: before ${before.outside}, after delete and undo ${after.outside}`);
      if (after.outside !== before.outside) throw new Error('crop out of date after undo');
      await run('window.__test.crop([0,0,0],[1,1,1])');
    }

    // the outline tools, including the modifiers and the depth switch
    await run('window.__test.clearSelection()');
    const L = '[[0.30,0.35],[0.55,0.30],[0.62,0.55],[0.50,0.70],[0.30,0.62]]';
    say(`lasso:    ${JSON.stringify(await run(`window.__test.lasso(${L})`))}`);
    await shot('09-lasso');
    await checkVolume('lasso');
    await dumpVolume('lasso');
    {
      // what closing the holes inside an outline changes: it adds what sits
      // in a hole, and drops the far wall seen through one
      await run('window.__test.holeFill(false)');
      const l0 = await run(`window.__test.lasso(${L})`);
      const b0 = await run("window.__test.box(0.35,0.40,0.60,0.62)");
      const p0 = await run(`window.__test.paintWorldMode(${STROKE}, 58, 1.5, 'replace')`);
      await run('window.__test.holeFill(true)');
      const p1 = await run(`window.__test.paintWorldMode(${STROKE}, 58, 1.5, 'replace')`);
      say(`holes:    brush ${p0.selected} → ${p1.selected}`);
      if (!p1.selected) throw new Error('brush with hole search selected nothing');
      const l1 = await run(`window.__test.lasso(${L})`);
      const b1 = await run("window.__test.box(0.35,0.40,0.60,0.62)");
      say(`holes:    lasso ${l0.selected} → ${l1.selected}, box ${b0.selected} → ${b1.selected}`);
      if (!l1.selected || !b1.selected) throw new Error('closing holes left nothing');
      await run(`window.__test.lasso(${L})`);
    }
    say(`lasso+:   ${JSON.stringify(await run(`window.__test.lasso(${L}, 'add')`))}`);
    say(`box:      ${JSON.stringify(await run("window.__test.box(0.35,0.40,0.60,0.62)"))}`);
    await shot('10-box');
    {
      // The size-aware reach (off by default) against the point test: it
      // may only add, for the box and the lasso alike.
      const b0 = await run("window.__test.box(0.35,0.40,0.60,0.62)");
      const l0 = await run(`window.__test.lasso(${L})`);
      const old = await run('window.__test.setSizeReach(3)');
      const b3 = await run("window.__test.box(0.35,0.40,0.60,0.62)");
      const l3 = await run(`window.__test.lasso(${L})`);
      await run(`window.__test.setSizeReach(${old.k}, ${old.cap})`);
      say(`box k=${old.k}/3:  ${b0.selected} / ${b3.selected} · lasso ${l0.selected} / ${l3.selected}`);
      if (b3.selected < b0.selected || l3.selected < l0.selected) {
        throw new Error('size-aware reach removed splats for box or lasso');
      }
      await run("window.__test.box(0.35,0.40,0.60,0.62)");
    }
    say(`box add:  ${JSON.stringify(await run("window.__test.box(0.62,0.40,0.75,0.62,'add')"))}`);
    say(`box sub:  ${JSON.stringify(await run("window.__test.box(0.62,0.40,0.75,0.62,'subtract')"))}`);
    await checkVolume('box minus');
    say(`through:  ${JSON.stringify(await run("window.__test.box(0.35,0.40,0.60,0.62,'replace','through')"))}`);
    await shot('11-through');
    await checkVolume('through');
    await dumpVolume('through');
    await run('window.__test.clearSelection()');
    await run("window.__test.setDepth('quick')");
    await run(`window.__test.paintWorld(${STROKE}, 58, 1.5)`);

    const rep = await run('window.__test.repair()');
    say(`repair: ${JSON.stringify(rep)}`);
    if (!rep?.generated) throw new Error('repair produced no splats: ' + rep?.status);
    await shot('04-repaired');

    await run('window.__test.frame(-38, -128, 2.1)');
    await shot('05-angle');

    const und = await run('window.__test.undo()');
    say(`undo: ${JSON.stringify(und)}`);
    await shot('06-undone');

    // the same stroke filled again, differently
    await run(`window.__test.paintWorld(${STROKE}, 58, 1.5)`);
    await run('window.__test.repair()');
    const rr = await run('window.__test.reroll()');
    say(`reroll: ${JSON.stringify(rr)}`);
    if (!rr?.generated) throw new Error('reroll produced nothing: ' + rr?.status);
    await run('window.__test.undo()');

    // the two retouching brushes: each must either act or say why not,
    // and come back off the undo stack cleanly
    // over the stand hole they must refuse -- there is no surface to edit --
    // and on intact sole beside it they must act
    const SOLE = '[[-6.6,-0.63,1.4],[-5.4,-0.63,1.4],[-6.6,-0.63,2.6],[-5.4,-0.63,2.6],[-6,-0.63,2]]';
    for (const [brush, where] of [['unshadow', STROKE], ['solidify', STROKE],
                                  ['unshadow', SOLE], ['solidify', SOLE]]) {
      await run(`window.__test.paintWorld(${where}, 58, 1.5)`);
      const r = await run(`window.__test.${brush}()`);
      say(`${brush}: ${JSON.stringify(r)}`);
      if (!r?.status) throw new Error(brush + ' gave no status');
      if (r.generated) {
        const u = await run('window.__test.undo()');
        if (u.generated !== 0) throw new Error(brush + ' did not undo cleanly');
      }
      await run('window.__test.clearSelection()');
    }

    // The shadow brush on a real shadow baked into the scan, as a user paints
    // it. "Find shadow" must say what it did or why not -- in the tile,
    // not only in the status bar -- and "Lighten" (directly, or through
    // "Lighten instead") must visibly lighten the stroke and nothing
    // far from it; Ctrl+Z must bring the picture back. Owner's report: "the
    // shadow remove tool just doesn't do anything at all".
    {
      const base = path.basename(ply).toLowerCase();
      const found = SHADOW_SITES.find(s => base.includes(s.file));
      const site = found && {
        ...found,
        target: [0, 1, 2].map(k => found.points.reduce((a, p) => a + p[k], 0) / found.points.length),
      };
      if (site) {
        await run(`window.__test.aim(${JSON.stringify(site.target)}, ${JSON.stringify(site.dir)}, ${site.dist})`);
      } else {
        await run('window.__test.frame(0, -92, 1.9)');
      }
      await new Promise(r => setTimeout(r, 1200));
      const rect = await run(`(() => { const r = document.getElementById('canvas').getBoundingClientRect();
        return { x: Math.round(r.x), y: Math.round(r.y), width: Math.round(r.width), height: Math.round(r.height) }; })()`);
      const changed = (a: Uint8Array, b: Uint8Array) => {
        let n = 0;
        for (let i = 0; i < Math.min(a.length, b.length); i += 4) {
          if (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]) > 24) n++;
        }
        return 100 * n / (a.length / 4);
      };
      // the viewport as it settles: grabbed until two frames in a row agree
      // (a camera still gliding made 60 % of the picture "change")
      const grab = async (name: string) => {
        let last: Uint8Array | null = null;
        let img = null as any;
        for (let k = 0; k < 12; k++) {
          await new Promise(r => setTimeout(r, 500));
          img = await w.webContents.capturePage(rect);
          const bm: Uint8Array = img.toBitmap();
          if (last && changed(last, bm) < 0.01) break;
          last = bm;
        }
        await fs.writeFile(path.join(outDir, name + '.png'), img.toPNG());
        return img.toBitmap() as Uint8Array;
      };
      const paint = async () => {
        const p = site
          ? await run(`window.__test.aim(${JSON.stringify(site.target)}, ${JSON.stringify(site.dir)}, ${site.dist})`)
          : null;
        const rpx = site ? site.radius / p.worldPerPixel : 58;
        return run(`window.__test.paintWorld(${site ? JSON.stringify(site.points) : SOLE}, ${rpx}, 1.5)`);
      };
      // the picture to compare against, from exactly the camera the stroke
      // is painted from (a camera still settling made 60 % of it "change")
      await paint();
      await run('window.__test.clearSelection()');
      const before = await grab('18-shadow-before');
      const painted = await paint();
      await shot('18-shadow-painted');
      const auto = await run("window.__test.unshadow('auto')");
      await shot('18-shadow-auto-tile');
      say(`shadow auto (${site?.key ?? 'test stroke'}, ${painted.selected} selected): `
          + `${auto.info ? 'acted' : 'refused'} · ${auto.note}`);
      if (!auto.note) throw new Error('find shadow: no feedback in the tile');
      let lift;
      if (!auto.info) {
        if (!auto.offersLift && !/hole/i.test(auto.note)) {
          throw new Error('find shadow refuses but does not offer to lighten');
        }
        lift = await run('window.__test.shadowInstead()');
      } else {
        await run('window.__test.undo()');
        await paint();
        lift = await run("window.__test.unshadow('lift')");
      }
      await run('window.__test.clearSelection()');
      const after = await grab('18-shadow-lifted');
      const pct = changed(before, after);
      say(`shadow lift: ${JSON.stringify(lift.info)} · ${pct.toFixed(2)} % of pixels brighter`);
      if (lift.info?.mode !== 'lift') throw new Error('lighten did nothing: ' + lift.status);
      if (pct < 0.3) throw new Error(`lighten barely visible: ${pct.toFixed(2)} % of pixels`);
      await run('window.__test.undo()');
      await run("window.__test.unshadow('auto', 1)");   // menu back to its default
      await run('window.__test.clearSelection()');
      const undone = await grab('18-shadow-undone');
      const back = changed(before, undone);
      say(`shadow undo: ${back.toFixed(2)} % of pixels differ from before`);
      if (back > 0.2) throw new Error(`lighten not undone cleanly: ${back.toFixed(2)} %`);
      await run('window.__test.frame(0, -92, 1.9)');
    }

    // and again, the deep way: render the patch, let an inpainter fill it,
    // fit the new Gaussians to that
    const caps = await run('window.__test.caps()');
    say(`caps: ${JSON.stringify(caps)}`);
    // LaMa only. A test must never start Photoshop -- it costs credits and
    // opens a window -- so where LaMa cannot run the deep step is skipped
    // rather than left to whatever the menu falls back to.
    const fill = caps?.deep && caps?.viewFill ? await run("window.__test.setFill('lama')") : null;
    if (fill === 'lama') {
      // This stroke twice selected nothing, right after the shadow step. The
      // stroke is aimed in world space through viewProjection(), and the
      // surface depth under it is turned back into a point through the live
      // camera. viewProjection() used to hand out the matrices the camera
      // component caches when a frame renders; a window that renders no frame
      // between the camera move and the stroke (behind others, GPU busy with
      // the engine) aimed it through the shadow step's close-up instead and
      // put every capsule in mid-air. It is built from the camera as it
      // stands now (engine/viewer.ts). So: camera move and stroke in the same
      // turn, no frame in between -- the case that failed -- and it must
      // select exactly what the same stroke selects after a frame.
      const dp = await run(`(() => { window.__test.frame(0, -92, 1.9);
        return window.__test.paintWorld(${STROKE}, 58, 1.5); })()`);
      const dDiag = await run('window.__test.paintDiag()');
      await run('window.__test.clearSelection()');
      await new Promise(r => setTimeout(r, 400));
      const dr = await run(`window.__test.paintWorld(${STROKE}, 58, 1.5)`);
      say(`deep painted: ${dp.selected} with no frame in between, ${dr.selected} after a frame`);
      if (!dp.selected || dp.selected !== dr.selected) {
        say(`  diag:    ${JSON.stringify(dDiag)}`);
        say(`  stroke:  ${JSON.stringify(dp)} / ${JSON.stringify(dr)}`);
        say(`  camera:  ${JSON.stringify(await run('window.__test.nav()'))}`);
        throw new Error('a stroke right after the camera move selects differently: ' + JSON.stringify([dp.selected, dr.selected]));
      }
      await run("window.__test.setDepth('deep')");
      const t1 = Date.now();
      const deepRes = await run('window.__test.repair()');
      say(`deep: ${JSON.stringify(deepRes)} in ${((Date.now() - t1) / 1000).toFixed(0)}s`);
      if (!deepRes?.generated) throw new Error('deep repair produced nothing');
      await shot('07-deep');
      await run('window.__test.frame(-38, -128, 2.1)');
      await shot('08-deep-angle');
    } else {
      say('deep: skipped (LaMa not available: '
          + (caps?.viewFillDetail ?? caps?.inpainterDetail ?? '?') + ')');
    }

    await shellChecks();
    await selectionChecks();

    // The canvas used to be pinned to whatever size it had on the first
    // frame, so going fullscreen grew the viewport around a render that
    // stayed put. Prove the backing store follows the container.
    const measure = () => run(`(() => {
      const c = document.getElementById('canvas');
      const b = (c.parentElement ?? c).getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio, 2);
      return { box: [Math.round(b.width), Math.round(b.height)],
               buffer: [c.width, c.height], dpr: window.devicePixelRatio,
               fits: Math.abs(c.width - Math.round(b.width * dpr)) <= 2
                  && Math.abs(c.height - Math.round(b.height * dpr)) <= 2 };
    })()`);
    // A window behind others is not rendered, and resize observers only run
    // when it is: wait for the frame (forcing one) rather than a fixed time.
    const settled = async () => {
      let m = await measure();
      for (let i = 0; i < 30 && !m.fits; i++) {
        if (i === 5) { w.show(); w.moveTop(); w.focus(); }   // behind other windows nothing renders
        w.webContents.invalidate();
        await new Promise(r => setTimeout(r, 100));
        m = await measure();
      }
      return m;
    };
    const before = await measure();
    w.setSize(1180, 760);
    await new Promise(r => setTimeout(r, 500));
    const small = await settled();
    w.maximize();
    await new Promise(r => setTimeout(r, 700));
    const big = await settled();
    await shot('17-maximised');
    await run("document.querySelectorAll('.menu__title')[1].click()");
    await shot('17b-maximised-menu');
    await run("document.querySelectorAll('.menu__title')[1].click()");
    say(`resize:   ${JSON.stringify(before)}`);
    say(`  small:  ${JSON.stringify(small)}`);
    say(`  large:  ${JSON.stringify(big)}`);
    if (!small.fits || !big.fits) throw new Error('canvas does not follow the viewport');

    say('AUTOTEST OK');
  } catch (e: any) {
    say('AUTOTEST FAILED: ' + (e?.message ?? e));
    process.exitCode = 1;
  } finally {
    await fs.writeFile(path.join(outDir, 'log.txt'), log.join('\n'), 'utf-8');
    // quit() ends the process with code 0 whatever process.exitCode says: a
    // failed run read as a pass to whatever started it. exit() skips the
    // quit events, so the engine is ended here.
    if (process.exitCode) {
      stopEngine();
      app.exit(1);
    } else {
      app.quit();
    }
  }
}
