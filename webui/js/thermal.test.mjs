import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const plain = (v) => JSON.parse(JSON.stringify(v));

function load() {
  const sandbox = {};
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(new URL('./thermal.js', import.meta.url), 'utf8'), sandbox);
  return sandbox.HISIM.thermal;
}

test('formatTemp shows kelvin and celsius, dashes for missing', () => {
  const t = load();
  assert.equal(t.formatTemp(350.16), '350.2 K (77.0 °C)');
  assert.equal(t.formatTemp(null), '—');
  assert.equal(t.formatTemp(undefined), '—');
  assert.equal(t.formatTemp(NaN), '—');
});

test('statusLabel', () => {
  const t = load();
  assert.equal(t.statusLabel('none'), '—');
  assert.equal(t.statusLabel(undefined), '—');
  assert.equal(t.statusLabel('running'), 'running…');
});

test('mapRange and gridToCsv', () => {
  const t = load();
  assert.deepEqual(plain(t.mapRange([[300, 310], [305, 320]])), { min: 300, max: 320 });
  const csv = t.gridToCsv([[300, 310]], [0, 1, 2], [0, 2]);
  assert.equal(csv, 'y_mm\\x_mm,0.5,1.5\n1,300,310\n');
});

test('heatRgb clamps and returns integer channels', () => {
  const t = load();
  assert.deepEqual(plain(t.heatRgb(-1)), plain(t.heatRgb(0)));
  assert.deepEqual(plain(t.heatRgb(2)), plain(t.heatRgb(1)));
  assert.ok(t.heatRgb(0.5).every((c) => Number.isInteger(c) && c >= 0 && c <= 255));
});

test('gridsToCsv writes every die in one long-format table', () => {
  const t = load();
  const csv = t.gridsToCsv([{ die: 0, grid: [[300, 301]] }, { die: 1, grid: [[310, 311]] }], [0, 1, 2], [0, 2]);
  assert.equal(csv, 'die,x_mm,y_mm,temperature_k\n0,0.5,1,300\n0,1.5,1,301\n1,0.5,1,310\n1,1.5,1,311\n');
});

test('montageLayout tiles all dies into one image', () => {
  const t = load();
  const one = plain(t.montageLayout(1, 80, 70));
  assert.equal(one.cols, 1);
  assert.equal(one.positions.length, 1);
  const four = plain(t.montageLayout(4, 80, 70));
  assert.equal(four.cols, 2);
  assert.equal(four.rows, 2);
  assert.equal(four.positions.length, 4);
  assert.ok(four.width > one.width && four.height > one.height);
  const xs = new Set(four.positions.map((p) => p.x));
  assert.equal(xs.size, 2);
  assert.ok(four.positions.every((p) => p.x + four.cellW <= four.width && p.y + four.cellH <= four.height));
});
