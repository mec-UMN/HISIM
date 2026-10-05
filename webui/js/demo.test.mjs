import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const app = fs.readFileSync(new URL('./app.js', import.meta.url), 'utf8');
const api = fs.readFileSync(new URL('./api.js', import.meta.url), 'utf8');

test('the Demo button has no star and opens a menu of pre-computed demo views', () => {
  assert.match(html, /<button class="btn ghost" id="btn-demo"[^>]*>Demo<\/button>/);
  assert.doesNotMatch(html, /★/);
  for (const view of ['showcase', 'architecture', 'thermal', 'sweeps']) {
    assert.match(html, new RegExp(`data-demo="${view}"`));
  }
  assert.match(html, /id="demo-menu"[^>]*hidden/);
});

test('the demo loads the bundled results instead of launching a simulation', () => {
  assert.match(api, /loadDemo:\s*\(\) => request\('\/demo\/load', \{ method: 'POST' \}\)/);
  assert.match(app, /HISIM\.api\.loadDemo\(\)/);
  assert.doesNotMatch(app, /DEMO_PATCH/);
  assert.match(app, /function openDemo\(/);
});

test('opening a demo run also puts its configuration in the knob panel', () => {
  assert.match(app, /HISIM\.controls\.set\(/);
  assert.match(app, /demoConfig/);
});
