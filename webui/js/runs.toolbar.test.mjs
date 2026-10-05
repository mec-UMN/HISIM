import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const source = fs.readFileSync(new URL('./runs.js', import.meta.url), 'utf8');
const css = fs.readFileSync(new URL('../css/app.css', import.meta.url), 'utf8');
const section = html.slice(html.indexOf('id="view-runs"'), html.indexOf('id="table-runs"'));

test('the runs toolbar is split into a title row with actions and a separate filter row', () => {
  const head = section.slice(section.indexOf('card-head'), section.indexOf('runs-filters'));
  const filters = section.slice(section.indexOf('runs-filters'));
  // title row: title, live summary, and the two actions
  for (const id of ['runs-summary', 'btn-delete-selected', 'btn-refresh-runs']) assert.match(head, new RegExp(`id="${id}"`));
  // filter row: launch, models, search, select visible
  for (const id of ['compare-group-filter', 'compare-model-filter', 'run-search', 'select-visible-runs']) {
    assert.match(filters, new RegExp(`id="${id}"`));
  }
  assert.match(filters, /class="filter-label">Models</);
  assert.doesNotMatch(head, /compare-group-filter|run-search/);
});

test('the summary reports shown / total runs and the selection', () => {
  assert.match(source, /el\('runs-summary'\)/);
  assert.match(source, /of \$\{total\} run/);
  assert.match(source, /selected/);
});

test('model chips are plain pills (no Selected/Hidden text) laid out inline with All / None', () => {
  assert.doesNotMatch(source, /model-check-state|'Selected' : 'Hidden'/);
  assert.match(css, /#compare-model-filter\s*\{[^}]*display:\s*flex/);
  assert.match(css, /\.runs-filters\s*\{[^}]*display:\s*flex[^}]*flex-wrap:\s*wrap/);
  assert.match(css, /\.runs-filters \.search\s*\{[^}]*flex:\s*1/);
});

test('the run table is sized to its rows instead of a fixed 380px box', () => {
  assert.match(html, /<div class="table-wrap auto resizable"><table class="data" id="table-runs">/);
  assert.match(css, /\.table-wrap\.auto\s*\{[^}]*height:\s*auto/);
});

test('the scorecard compares every visible run when none are ticked (selection only narrows it)', () => {
  assert.match(source, /const scope = chosenOrAll\(\)/);
  assert.match(source, /function chosenOrAll\(\)/);
  // ticking stays an explicit action: nothing is silently added to the selection used for deletion
  assert.doesNotMatch(source, /selected\.add\(run\.id\)\);\s*\/\/ default/);
  assert.match(html, /id="radar-scope"/);
});
