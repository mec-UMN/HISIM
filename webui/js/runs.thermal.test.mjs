import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const source = fs.readFileSync(new URL('./runs.js', import.meta.url), 'utf8');

test('Runs & Compare and sweeps carry no thermal controls (thermal is a single-run feature)', () => {
  assert.doesNotMatch(source, /thermal|peak_temperature_k/i);
  assert.doesNotMatch(html, /btn-thermal-selected|btn-select-pareto/);
});
