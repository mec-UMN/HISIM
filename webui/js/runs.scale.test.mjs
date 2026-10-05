import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const runs = fs.readFileSync(new URL('./runs.js', import.meta.url), 'utf8');
const controls = fs.readFileSync(new URL('./controls.js', import.meta.url), 'utf8');
const css = fs.readFileSync(new URL('../css/app.css', import.meta.url), 'utf8');

test('sweep response and design trade-off calibrate their axes with the shared helper', () => {
  assert.match(runs, /HISIM\.plotscale\.axisFor\(/);
  assert.doesNotMatch(runs, /const useLog = /);               // no more blanket log axes
  assert.match(runs, /identical/);                              // a flat sweep is called out
  for (const file of ['plotscale.js', 'sweepinfo.js']) {
    assert.match(html, new RegExp(`<script src="js/${file.replace('.', '\\.')}\\?v=`));
  }
  assert.ok(html.indexOf('js/plotscale.js') < html.indexOf('js/runs.js'));
  assert.ok(html.indexOf('js/sweepinfo.js') < html.indexOf('js/controls.js'));
});

test('the sweep box warns when the chosen knob cannot change the result', () => {
  assert.match(html, /id="sweep-warning"/);
  assert.match(controls, /HISIM\.sweepinfo\.sweepWarning\(/);
  assert.match(css, /\.sweep-warning\s*\{/);
});

test('assets carry a fresh cache-busting version', () => {
  assert.match(html, /app\.css\?v=thermal-1/);
  assert.doesNotMatch(html, /\?v=thermal-[34]/);
});
