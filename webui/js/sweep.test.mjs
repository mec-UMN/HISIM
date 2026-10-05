import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const app = fs.readFileSync(new URL('./app.js', import.meta.url), 'utf8');

test('a sweep launch sends the whole sweep request (labels, model comparison) to the API', () => {
  assert.match(app, /const \{ knob, knob_label, values, compare_models, models \} = HISIM\.controls\.sweepRequest\(\);/);
  assert.match(app, /startSweep\(\{ config, knob, knob_label, values, compare_models, models \}\)/);
  assert.match(app, /sweep-mode-label/);
});
