import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { transform } from 'esbuild';

// Use the app's existing TS compiler tooling on every supported Node version.
const source = await readFile(new URL('../src/ui/history.ts', import.meta.url), 'utf8');
const { code } = await transform(source, { loader: 'ts', format: 'esm' });
const { History } = await import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));

test('the byte cap keeps the revision left by evicted edits and selections', async () => {
  let selection = 0;
  let engineUndoCalls = 0;
  const history = new History({
    busy: () => false,
    engineUndo: async () => {
      engineUndoCalls++;
      return { applied: true };
    },
    engineRedo: async () => ({ applied: true })
  });
  history.addPart('selection', {
    snapshot: () => selection,
    restore: (value) => { selection = value; },
    equal: (a, b) => a === b,
    // Charge the history for large snapshots without allocating 256 MiB.
    bytes: () => 50 * 1024 * 1024
  });
  history.reset('Open');
  assert.equal(history.revision, 0);

  history.noteEdit({ canUndo: true }, 'Delete');
  const firstEdit = history.revision;
  assert.ok(firstEdit > 0);
  for (const value of [1, 2]) {
    selection = value;
    history.noteSelection('Select');
  }
  assert.deepEqual(history.entries.map(({ kind }) => kind), ['base', 'selection']);
  assert.equal(history.revision, firstEdit);
  assert.ok(await history.undo());
  assert.equal(selection, 1);
  assert.equal(history.revision, firstEdit);
  assert.equal(engineUndoCalls, 0);
  assert.equal(history.canUndo, false);

  history.noteEdit({ canUndo: true }, 'Delete again');
  const secondEdit = history.revision;
  assert.notEqual(secondEdit, firstEdit);
  assert.equal(history.canRedo, false);
  assert.ok(await history.undo());
  assert.equal(history.revision, firstEdit);
  assert.equal(engineUndoCalls, 1);
  assert.ok(await history.redo());
  assert.equal(history.revision, secondEdit);

  history.clear();
  assert.equal(history.revision, 0);
  history.reset('Open another file');
  assert.equal(history.revision, 0);
});
