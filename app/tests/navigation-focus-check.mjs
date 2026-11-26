import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const source=readFileSync('src/engine/viewer.ts','utf8');
const begin=source.indexOf('  private beginGesture(');
const end=source.indexOf('\n  /** Centre the view',begin);
const method=source.slice(begin,end);
const body=method.slice(method.indexOf('{')+1,method.lastIndexOf('}'));
const fn=new Function('g','x','y',body);
for (const point of [[20,30],[800,400]]) {
  const ctx={preset:'supersplat',drag:{pivot:'stale'},target:{x:1,y:2,z:3},distance:7,
    surfaceDepthAt(){throw new Error('Orbit must keep the existing focus');}};
  fn.call(ctx,'orbit',...point);
  assert.equal(ctx.drag.pivot,null);
  assert.deepEqual(ctx.target,{x:1,y:2,z:3});
  assert.equal(ctx.distance,7);
}
let samples=0;
const ctx={preset:'blender',drag:{pivot:'stale'},surfaceDepthAt(){samples++;return null;}};
fn.call(ctx,'orbit',20,30);
assert.equal(samples,1);
assert.equal(ctx.drag.pivot,null);
console.log('PASS stable SuperSplat focus across cursor positions; Blender surface picking preserved');
