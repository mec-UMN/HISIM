import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const plain = (v) => JSON.parse(JSON.stringify(v));
function load() {
  const sandbox = {};
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(new URL('./tilefmt.js', import.meta.url), 'utf8'), sandbox);
  return sandbox.HISIM.tilefmt;
}

test('layerList accepts arrays and printed python lists', () => {
  const t = load();
  assert.deepEqual(plain(t.layerList("['L84', 'L85', 'L9']")), ['L84', 'L85', 'L9']);
  assert.deepEqual(plain(t.layerList(['L1', 'L2'])), ['L1', 'L2']);
  assert.deepEqual(plain(t.layerList('[]')), []);
  assert.deepEqual(plain(t.layerList(undefined)), []);
});

test('typeLabel drops the layer list that hw_type carries', () => {
  const t = load();
  assert.equal(t.typeLabel({ hw_type: 'L123 L120 L117 CPU', hw_class: 'CPU' }), 'CPU');
  assert.equal(t.typeLabel({ hw_type: 'L162 Out Mem', hw_class: 'Mem' }), 'Out Mem');
  assert.equal(t.typeLabel({ hw_type: 'Mem Tile', hw_class: 'Mem' }), 'Mem Tile');
  assert.equal(t.typeLabel({ hw_type: 'L5', hw_class: 'SA' }), 'SA');
  assert.equal(t.typeLabel({}), '—');
});

test('layerRanges sorts numerically and merges runs', () => {
  const t = load();
  assert.equal(t.layerRanges(['L8', 'L1', 'L2', 'L3', 'L6', 'L9', 'L2']), 'L1–L3, L6, L8–L9');
  assert.equal(t.layerRanges([]), '');
});

test('layerSummary gives a count, a short preview and the full text', () => {
  const t = load();
  const long = Array.from({ length: 40 }, (_, i) => `L${i * 3}`);
  const s = plain(t.layerSummary(long, 24));
  assert.equal(s.count, 40);
  assert.ok(s.preview.length <= 25 && s.preview.endsWith('…'));
  assert.ok(s.full.includes('L117'));
  assert.deepEqual(plain(t.layerSummary(['L4', 'L5'], 24)), { count: 2, preview: 'L4–L5', full: 'L4–L5' });
  assert.deepEqual(plain(t.layerSummary([], 24)), { count: 0, preview: '', full: '' });
});
