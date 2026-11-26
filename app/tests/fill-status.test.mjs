import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { transform } from 'esbuild';

const source = await readFile(new URL('../src/ui/fill-status.ts', import.meta.url), 'utf8');
const { code } = await transform(source, { loader: 'ts', format: 'esm' });
const { formatSideError } = await import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));

test('the side-view error is displayed as the percentage of full brightness', () => {
  assert.equal(formatSideError({ offAxisGeo: 0.12, offAxisDeep: 0.09 }),
               ' · from the side 12.0 → 9.0%');
  assert.equal(formatSideError({ offAxisGeo: 0.02, offAxisDeep: 0 }),
               ' · from the side 2.0 → 0.0%');
});

test('the side-view summary is omitted when either metric is missing', () => {
  assert.equal(formatSideError({ offAxisGeo: null, offAxisDeep: 0.09 }), '');
  assert.equal(formatSideError({ offAxisGeo: 0.12 }), '');
});

test('the multi-chart summary identifies the first repaired face', () => {
  assert.equal(formatSideError({ offAxisGeo: 0.12, offAxisDeep: 0.09, deep_charts: [{}, {}] }),
               ' · from the side (first repaired face) 12.0 → 9.0%');
});
