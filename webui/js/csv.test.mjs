import assert from 'node:assert/strict';

// csv.js is a classic browser script: it assigns `window.HISIM.csv = ...`
// and then references the bare global `HISIM` (as real browsers alias
// globals onto `window`). Mirror that by making `window` literally be
// the Node global object, exactly like a real browser's `window === globalThis`.
globalThis.window = globalThis;
await import('./csv.js');
const { tilesToCSV } = globalThis.HISIM.csv;

const columns = [['id', 'Tile'], ['note', 'Note'], ['area_mm2', 'Area (mm2)']];
const rows = [
  { id: 'C0_T0_0', note: 'plain', area_mm2: 0.125 },
  { id: 'C0_T0_1', note: 'has, a comma', area_mm2: 0.5 },
  { id: 'C0_T0_2', note: 'has "quotes"', area_mm2: 0 },
];

const csv = tilesToCSV(columns, rows);
const lines = csv.split('\r\n');

assert.equal(lines[0], 'Tile,Note,Area (mm2)');
assert.equal(lines[1], 'C0_T0_0,plain,0.125');
assert.equal(lines[2], 'C0_T0_1,"has, a comma",0.5');
assert.equal(lines[3], 'C0_T0_2,"has ""quotes""",0');
assert.equal(lines.length, 4);

console.log('csv.js tests passed');
