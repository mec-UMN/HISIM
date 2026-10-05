import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const plain = (v) => JSON.parse(JSON.stringify(v));
function load() {
  const sandbox = {};
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(new URL('./pow2.js', import.meta.url), 'utf8'), sandbox);
  return sandbox.HISIM.pow2;
}

test('snap rounds to the nearest power of two inside the knob range', () => {
  const p = load();
  assert.equal(p.snap(470, 1, 512), 512);
  assert.equal(p.snap(3, 1, 512), 4);
  assert.equal(p.snap(5, 1, 512), 4);
  assert.equal(p.snap(0, 1, 512), 1);
  assert.equal(p.snap(1000, 1, 512), 512);
  assert.equal(p.snap(20, 1, 64), 16);
});

test('the slider works in exponent steps', () => {
  const p = load();
  assert.deepEqual(plain(p.exponentRange(1, 512)), { min: 0, max: 9 });
  assert.equal(p.toExponent(64), 6);
  assert.equal(p.fromExponent(4), 16);
  assert.deepEqual(plain(p.values(1, 64)), [1, 2, 4, 8, 16, 32, 64]);
});

const controls = fs.readFileSync(new URL('./controls.js', import.meta.url), 'utf8');
const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
test('power-of-two knobs drive their slider and number box through the helper', () => {
  assert.match(controls, /field\.scale === 'pow2'/);
  assert.match(controls, /HISIM\.pow2\.snap\(/);
  assert.match(controls, /HISIM\.pow2\.fromExponent\(/);
  assert.match(html, /<script src="js\/pow2\.js\?v=/);
  assert.ok(html.indexOf('js/pow2.js') < html.indexOf('js/controls.js'));
});
