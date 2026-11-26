import assert from 'node:assert/strict';
import { copyFileSync, mkdtempSync, mkdirSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { app, dialog, ipcMain } from 'electron';

// Keep the normal confirmation handlers enabled and isolate settings and exports.
assert.ok(process.argv[2], 'Usage: electron tests/unexported-edits.mjs <plane.ply>');
assert.ok(!process.env.SPLAT_AUTOTEST, 'Unset SPLAT_AUTOTEST to test confirmations');
const fixture = path.resolve(process.argv[2]);
const temp = mkdtempSync(path.join(tmpdir(), 'splat-history-'));
const profile = path.join(temp, 'profile');
mkdirSync(profile);
app.setPath('userData', profile);
const secondFile = path.join(temp, 'next.ply');
copyFileSync(fixture, secondFile);
console.log('Results and Electron profile:', temp);

let discard = false;
let output;
let saveDialogs = 0;
const confirmations = [];
dialog.showMessageBox = async (_window, options) => {
  confirmations.push(options);
  return { response: discard ? 0 : 1 };
};
dialog.showSaveDialog = async () => {
  saveDialogs++;
  return output ? { canceled: false, filePath: output } : { canceled: true };
};

async function until(read, accept, label) {
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    const value = await read();
    if (accept(value)) return value;
    await delay(25);
  }
  throw new Error('Timed out waiting for ' + label);
}

async function check(win) {
  const wc = win.webContents;
  const run = (code) => wc.executeJavaScript(code);
  const invoke = (method, ...args) => run(`window.__test[${JSON.stringify(method)}](...${JSON.stringify(args)})`);
  const stats = () => invoke('stats');
  const query = async () => {
    let result;
    const receive = (event, value) => {
      if (event.sender === wc) result = value;
    };
    ipcMain.on('doc:unexported', receive);
    try {
      wc.send('doc:ask-unexported');
      return await until(() => result, (x) => typeof x === 'boolean', 'renderer unexported reply');
    } finally {
      ipcMain.removeListener('doc:unexported', receive);
    }
  };
  const expectDirty = async (expected, context) => assert.equal(await query(), expected, context);
  const paint = async (x) => assert.ok(await invoke('paint', x, 0.5, 55, 1.5), 'fixture must be selectable');
  const remove = async (x) => {
    const { alive } = await stats();
    await paint(x);
    const next = (await invoke('deleteSelection')).alive;
    assert.ok(next < alive, 'delete must change the geometry');
    return next;
  };
  const exportTo = async (destination, expectedStatus) => {
    output = destination;
    const expectedDialogs = saveDialogs + 1;
    await run('document.querySelector("#btn-export").click()');
    await until(() => saveDialogs, (count) => count === expectedDialogs, 'save dialog');
    if (expectedStatus) {
      await until(stats, ({ status }) => expectedStatus.test(status), 'export result ' + expectedStatus);
    }
  };
  const undoEdit = async (expectedAlive) => {
    // Selection steps share the history, so undo the paint before reaching older geometry.
    for (let i = 0; i < 8; i++) {
      assert.ok((await invoke('historyUndo')).ok, 'expected an undo step');
      if ((await stats()).alive === expectedAlive) return;
    }
    assert.fail('undo did not reach the expected splat count');
  };

  await until(stats, ({ renderer }) => /WebGPU|WebGL/.test(renderer), 'graphics startup');
  await expectDirty(false, 'startup');
  const original = (await invoke('open', fixture)).alive;
  assert.ok(original > 0);
  await paint(0.5);
  await expectDirty(false, 'selection alone');
  const edited = (await invoke('deleteSelection')).alive;
  assert.ok(edited < original);
  await expectDirty(true, 'delete');
  const entries = await run(`(() => {
    for (let i = 0; i < 200; i++) window.__test.invert();
    return window.__test.historyState().entries;
  })()`);
  assert.equal(entries.length, 200);
  assert.ok(!entries.some((entry) => entry.startsWith('edit:')));
  assert.equal((await stats()).alive, edited);
  const dirtyAfterTrim = await query();
  const afterOpen = await invoke('open', secondFile);
  console.log({ original, edited, entries: entries.length, dirtyAfterTrim,
    openPrompts: confirmations.length, aliveAfterOpen: afterOpen.alive });
  assert.equal(confirmations.length, 1, 'opening after history trim must ask before discarding');
  assert.equal(dirtyAfterTrim, true, 'evicting an edit must preserve dirty state');
  assert.equal(afterOpen.path, fixture);
  assert.equal(afterOpen.alive, edited);

  await run('window.host.closeWindow()');
  const closePrompt = await until(() => confirmations[1], Boolean, 'close confirmation');
  assert.match(closePrompt.detail, /close Splat Repair/);
  assert.equal((await stats()).alive, edited);
  await expectDirty(true, 'canceling open and close preserves the edits');

  await exportTo(undefined);
  await expectDirty(true, 'canceling export');
  await exportTo(path.join(temp, 'missing', 'failure.ply'), /cannot|no such|not found|does not exist|folder/i);
  await expectDirty(true, 'failed export');
  const saved = path.join(temp, 'saved.ply');
  await exportTo(saved, /^exported as ply/);
  assert.match(readFileSync(saved).subarray(0, 1024).toString(), new RegExp(`element vertex ${edited}\\r?\\n`));
  await expectDirty(false, 'exporting after trim');
  await invoke('invert');
  await invoke('invert');
  await expectDirty(false, 'selections after export');

  const later = await remove(0.45);
  await expectDirty(true, 'editing after export');
  await undoEdit(edited);
  await expectDirty(false, 'undo to the exported revision');
  assert.ok((await invoke('historyRedo')).ok);
  assert.equal((await stats()).alive, later);
  await expectDirty(true, 'redo away from the exported revision');
  await exportTo(path.join(temp, 'later.ply'), /^exported as ply: /);
  await expectDirty(false, 'exporting the later edit');
  await undoEdit(edited);
  await expectDirty(true, 'undo away from the later export');
  await remove(0.55);
  assert.equal((await invoke('historyState')).canRedo, false);
  await expectDirty(true, 'a different edit at the same engine depth');

  discard = true;
  assert.equal((await invoke('open', secondFile)).alive, original);
  assert.equal((await invoke('historyState')).entries.length, 1);
  await expectDirty(false, 'opening a new document');
  await remove(0.5);
  await undoEdit(original);
  await expectDirty(false, 'undo to the original file');
  assert.ok((await invoke('historyRedo')).ok);
  await expectDirty(true, 'redo away from the original file');

  const reloaded = await invoke('open', saved);
  assert.equal(reloaded.path, saved);
  assert.equal(reloaded.alive, edited, 'the exported PLY must reopen with the saved geometry');
  await expectDirty(false, 'opening the exported PLY');
}

let exitCode = 1;
app.on('will-quit', () => app.exit(exitCode));
const watchdog = setTimeout(() => {
  console.error('Unexported edits test exceeded 90 seconds');
  app.exit(1);
}, 90000);
app.once('browser-window-created', (_event, win) => {
  win.webContents.once('did-finish-load', async () => {
    try {
      await check(win);
      console.log('PASS unexported edits: confirmations, exports, undo/redo and branching');
      exitCode = 0;
    } catch (error) {
      console.error(error);
    } finally {
      clearTimeout(watchdog);
      win.destroy();
      app.quit();
    }
  });
});
await import('../dist/electron/main.mjs');
