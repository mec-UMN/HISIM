import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const plain = (v) => JSON.parse(JSON.stringify(v));
function load(file, name) {
  const sandbox = {};
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(new URL(file, import.meta.url), 'utf8'), sandbox);
  return sandbox.HISIM[name];
}

test('identical values get a padded linear axis with one tick, not a collapsed log axis', () => {
  const { axisFor } = load('./plotscale.js', 'plotscale');
  const a = plain(axisFor([7.77, 7.77], '(s)'));
  assert.equal(a.type, 'linear');
  assert.equal(a.constant, true);
  assert.ok(a.range[0] < 7.77 && a.range[1] > 7.77);
  assert.deepEqual(a.tickvals, [7.77]);
});

test('values within one order of magnitude stay linear and padded; wide spreads go log', () => {
  const { axisFor } = load('./plotscale.js', 'plotscale');
  const close = plain(axisFor([10, 12, 15], '(s)'));
  assert.equal(close.type, 'linear');
  assert.ok(close.range[0] < 10 && close.range[1] > 15);
  assert.equal(close.constant, false);
  const wide = plain(axisFor([0.001, 0.5, 40], '(s)'));
  assert.equal(wide.type, 'log');
  assert.equal(wide.range, undefined);                     // let Plotly size a log axis
});

test('non-positive values can never be put on a log axis', () => {
  const { axisFor } = load('./plotscale.js', 'plotscale');
  assert.equal(plain(axisFor([0, 5, 5000], '(s)')).type, 'linear');
});

test('tick format follows the unit in the label', () => {
  const { axisFor } = load('./plotscale.js', 'plotscale');
  const s = plain(axisFor([1, 2], 'End-to-end latency (s)'));
  assert.equal(s.ticksuffix, 's');
  assert.equal(s.tickformat, '.3~s');
  const usd = plain(axisFor([1, 2], 'Cost per part ($)'));
  assert.equal(usd.tickprefix, '$');
  const area = plain(axisFor([100, 200], 'Total chip area (mm²)'));
  assert.equal(area.ticksuffix, ' mm²');
  assert.equal(plain(axisFor([], '(s)')).type, 'linear');
});

test('bank count is flagged as ineffective while banks are auto-sized', () => {
  const { sweepWarning } = load('./sweepinfo.js', 'sweepinfo');
  assert.match(sweepWarning('n_bank', { set_suff_banks: true }), /auto-sized|identical/i);
  assert.equal(sweepWarning('n_bank', { set_suff_banks: false }), '');
  assert.equal(sweepWarning('n_sa', { set_suff_banks: true }), '');
  assert.equal(sweepWarning(undefined, {}), '');
});

test('clock sweeps are not flagged: each run gets its own network clock in its private engine copy', () => {
  const { sweepWarning } = load('./sweepinfo.js', 'sweepinfo');
  assert.equal(sweepWarning('clock_hz', {}), '');
});
