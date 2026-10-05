import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const source = fs.readFileSync(new URL('./runs.js', import.meta.url), 'utf8');
const controls = fs.readFileSync(new URL('./controls.js', import.meta.url), 'utf8');
const css = fs.readFileSync(new URL('../css/app.css', import.meta.url), 'utf8');

test('Runs & Compare scopes a sweep launch + model filter and plots per-model response', () => {
  assert.match(html, /id="compare-group-filter"/);
  assert.match(html, /id="compare-model-filter"/);
  assert.match(html, /id="sweep-metric"/);
  assert.match(html, /id="plot-sweep"/);
  assert.match(html, /id="sweep-values-label"/);
  assert.match(source, /function drawSweep\(\)/);
  assert.match(source, /sweep_knob_label/);
  assert.match(source, /sweep_value/);
  assert.match(source, /function visibleRecords\(\)/);
  assert.match(controls, /sweep-values-label/);
});

test('the compare view keeps the empty state explicit', () => {
  assert.match(source, /No completed runs yet/);
});

test('comparing multiple models: launch+model filter drives per-model chart traces', () => {
  assert.match(source, /function modelsForGroup\(/);
  assert.match(source, /function renderCompareModelChecklist\(/);
  assert.match(source, /activeModels/);
  // sweep chart draws one line per model rather than a single flat series
  assert.match(source, /charts\.modelColor\(model\)/);
  // Pareto colors each point by its model too, not just selected/unselected
  assert.match(source, /charts\.modelColor\(p\.run\.sweep_model/);
});

test('multi-model sweep comparisons are explicitly enabled and counted', () => {
  assert.match(html, /id="sweep-compare-models"/);
  assert.match(html, /id="sweep-models-field"/);
  assert.match(html, /id="sweep-models"/);
  assert.match(controls, /function renderSweepModelChips\(/);
  assert.match(controls, /compare_models/);
  assert.match(controls, /\$\{values\.length\} values × \$\{modelCount\} models = \$\{totalRuns\} runs/);
});

test('sweep model choices are an always-visible button group', () => {
  assert.match(html, /class="model-chips sweep-model-buttons" id="sweep-models"/);
  assert.match(css, /#sweep-models\.sweep-model-buttons\s*\{[\s\S]*?display:\s*flex;[\s\S]*?flex-wrap:\s*wrap;[\s\S]*?max-height:\s*none;[\s\S]*?overflow:\s*visible;/);
});

test('comparison and bulk deletion use one explicit sweep launch scope', () => {
  assert.match(html, /id="compare-group-filter"/);
  assert.match(html, /id="btn-delete-selected"/);
  assert.match(html, /id="select-visible-runs"/);
  assert.match(source, /let activeGroup = null/);
  assert.match(source, /run\.group === activeGroup/);
  assert.match(source, /data-model-filter/);
  assert.match(source, /function deleteSelectedRuns\(/);
  // Checkboxes are an explicit user action, never silently preselected for deletion.
  assert.doesNotMatch(source, /completed\(\)\.slice\(0, 3\)/);
});

test('model filters use an explicit checklist that permits an empty chart scope', () => {
  assert.match(source, /function renderCompareModelChecklist\(/);
  assert.match(source, /data-model-filter/);
  assert.match(source, /data-models-all/);
  assert.match(source, /data-models-none/);
  assert.match(source, /Select at least one model/);
  assert.doesNotMatch(source, /aria-pressed/);
});
