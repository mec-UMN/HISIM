import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const source = fs.readFileSync(new URL('./architecture.js', import.meta.url), 'utf8');
const three = fs.readFileSync(new URL('./architecture3d.js', import.meta.url), 'utf8');
const app = fs.readFileSync(new URL('./app.js', import.meta.url), 'utf8');

test('Package explorer has a Package | Thermal view switch instead of Heat by', () => {
  assert.doesNotMatch(html, /Heat by|id="arch-metric"|id="arch-thermal"[^-]/);
  // the colouring choices live under Package only
  assert.match(html, /Colour tiles by/);
  assert.match(html, /id="arch-color-field"/);
  for (const option of ['hw_class', 'latency_s', 'energy_j', 'area_mm2', 'router_area_mm2']) {
    assert.match(html, new RegExp(`<option value="${option}"`));
  }
  assert.match(source, /el\('arch-color-by'\)\.value/);
  assert.match(source, /setMetric\(el\('arch-color-by'\)\.value\)/);
  assert.match(source, /el\('arch-color-field'\)\.hidden = /);   // hidden in the Thermal view
  assert.match(html, /data-seg="arch-view"/);
  assert.match(html, /data-value="package"/);
  assert.match(html, /id="arch-view-thermal"[^>]*data-value="thermal"|data-value="thermal"[^>]*id="arch-view-thermal"/);
  for (const id of ['arch-thermal-png', 'arch-thermal-csv', 'arch-thermal-legend', 'arch-links']) {
    assert.match(html, new RegExp(`id="${id}"`));
  }
  assert.doesNotMatch(html, /arch-thermal-die/);            // no per-die selection
  assert.match(source, /function setArchView\(/);
  assert.match(source, /function tierForDie\(/);
  assert.match(source, /HISIM\.api\.thermalMaps/);
  assert.match(source, /HISIM\.thermal\.montageLayout/);   // one PNG with all layers
  assert.match(source, /HISIM\.thermal\.gridsToCsv/);       // one CSV with all dies
  assert.doesNotMatch(source, /runThermal/);
  assert.match(three, /function setThermal\(/);
  assert.match(three, /thermalSpec\.planes/);
  assert.match(source, /sort\(\(a, b\) => b - a\);\n    return tiers\[die\]/);
});

test('the dashboard button runs the thermal pass and the page follows it to completion', () => {
  assert.match(app, /run_id: runId/);
  assert.match(app, /thermal_status: record\?\.thermal_status/);
  assert.match(app, /btn-run-thermal/);
  assert.match(app, /HISIM\.api\.runThermal\(/);
  assert.match(app, /function watchThermal\(/);
  assert.doesNotMatch(app, /thermal-enable/);
  assert.match(fs.readFileSync(new URL('./api.js', import.meta.url), 'utf8'), /runThermal:/);
});

test('the hidden attribute wins over component display rules (so view-specific controls really hide)', () => {
  const css = fs.readFileSync(new URL('../css/app.css', import.meta.url), 'utf8');
  assert.match(css, /\[hidden\]\s*\{\s*display:\s*none\s*!important;?\s*\}/);
});

test('the tile inventory shows a type badge, a layer summary and keeps numbers on screen', () => {
  assert.match(html, /<script src="js\/tilefmt\.js/);
  assert.match(html, /id="tile-count"/);
  assert.match(source, /HISIM\.tilefmt\.typeLabel/);
  assert.match(source, /HISIM\.tilefmt\.layerSummary/);
  // numbers come before nothing wide: layer summary is truncated, full list in the tooltip
  assert.match(source, /class="layer-cell" title=/);
  assert.match(source, /class="num"/);
  const css = fs.readFileSync(new URL('../css/app.css', import.meta.url), 'utf8');
  assert.match(css, /table\.data td\.num\s*\{[^}]*text-align:\s*right/);
  assert.match(css, /\.layer-cell\s*\{[^}]*text-overflow:\s*ellipsis/);
  // the CSV export keeps the full raw columns
  assert.match(source, /CSV_COLUMNS/);
});

test('the Package view has a colour key that follows "Colour tiles by", and the thermal key stays in Thermal view', () => {
  assert.match(html, /id="arch-package-legend"/);
  assert.match(source, /function renderPackageLegend\(/);
  assert.match(source, /packageLegend\(/);
  assert.match(source, /arch-package-legend/);
  // refreshed on every change that can alter the colouring or the view
  assert.match(source, /setMetric\(el\('arch-color-by'\)\.value\);[\s\S]{0,40}renderPackageLegend\(\)/);
  assert.match(source, /host\.hidden = archView === 'thermal'/);   // only shown in the Package view
  assert.match(three, /import \{[^}]*HEAT_STOPS[^}]*\} from '\.\/architecture3d-layout\.js'/);
});


test('Thermal view is always selectable and explains itself when there is no thermal data', () => {
  assert.doesNotMatch(html, /id="arch-view-thermal"[^>]*disabled/);
  assert.match(html, /<div class="arch-stage">\s*<div class="plot xtall resizable" id="plot-arch3d"><\/div>\s*<div class="arch-empty" id="arch-thermal-empty" hidden>/);
  assert.match(html, /id="arch-thermal-goto"/);
  assert.doesNotMatch(source, /thermalButton\.disabled = /);
  assert.match(source, /el\('arch-thermal-empty'\)\.hidden = /);
  assert.match(source, /payload\?\.thermal_error/);           // the real reason (e.g. 3.5D not supported)
  // the coarsening / status note belongs to the Thermal view only
  assert.match(source, /note\.hidden = archView !== 'thermal'/);
});

test('the tile inventory footer is a compact bar with a Show all action', () => {
  assert.match(html, /<div class="table-foot" id="tile-foot">\s*<span id="tile-count"><\/span>\s*<button class="chip" id="tile-show-all"/);
  assert.match(source, /showAllTiles/);
  const css = fs.readFileSync(new URL('../css/app.css', import.meta.url), 'utf8');
  assert.match(css, /\.table-foot\s*\{[^}]*display:\s*flex/);
});
