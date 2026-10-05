import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const source = fs.readFileSync(new URL('./mapping.js', import.meta.url), 'utf8');
const appSource = fs.readFileSync(new URL('./app.js', import.meta.url), 'utf8');
const viewsSource = fs.readFileSync(new URL('./views.js', import.meta.url), 'utf8');
const apiSource = fs.readFileSync(new URL('./api.js', import.meta.url), 'utf8');

test('input file panel exposes all six HISIM CSV kinds', () => {
  for (const kind of ['chip_map', 'sys_map', 'layer_map', 'sa_spec', 'mem_spec', 'network_spec']) {
    assert.match(source, new RegExp(`['"]${kind}['"]`));
  }
  assert.match(html, /HISIM input files/);
  assert.match(source, /uploadMappingFile/);
  assert.match(source, /mappingFileDownloadUrl/);
  assert.match(source, /Download template/);
  assert.doesNotMatch(source, /info\.exists \? '' : ' disabled'/);
});

test('input-file actions live before the sweep and expose separate run modes', () => {
  const inputPanel = html.indexOf('id="input-files-box"');
  const sweep = html.indexOf('id="sweep-box"');
  assert.ok(inputPanel >= 0 && inputPanel < sweep, 'input files must appear before the sweep');
  assert.match(html, /id="btn-run-new-model"/);
  assert.match(html, /id="btn-run-uploaded-files"/);
  assert.match(html, /id="mapping-file-ack"/);
  assert.match(appSource, /launchWithMode/);
  assert.match(appSource, /create_default_files\s*=\s*true/);
  assert.match(appSource, /create_default_files\s*=\s*false/);
  assert.match(source, /mapping-file-ack/);
});

test('input files use collapsible download and upload subsections', () => {
  assert.match(html, /<details[^>]+id="input-files-box"/);
  assert.match(html, /Download input files/);
  assert.match(html, /Upload replacement files/);
  assert.match(html, /id="mapping-download-files"/);
  assert.match(html, /id="mapping-upload-files"/);
  assert.match(source, /mapping-download-files/);
  assert.match(source, /mapping-upload-files/);
  assert.match(apiSource, /mappingFilesOverview/);
  assert.match(source, /input-files-overview/);
});

test('local UI assets are versioned so browser caches cannot mix releases', () => {
  assert.match(html, /href="css\/app\.css\?v=[^"]+"/);
  for (const asset of ['api.js', 'format.js', 'charts.js', 'csv.js', 'mapping.js', 'controls.js', 'views.js', 'architecture.js', 'runs.js', 'app.js']) {
    assert.match(html, new RegExp(`src="js\\/${asset}\\?v=[^\"]+"`));
  }
  assert.match(html, /src="js\/architecture3d\.js\?v=[^"]+"/);
});

test('input file rendering tolerates an older API client during a cache transition', () => {
  assert.match(source, /typeof HISIM\.api\.mappingFilesOverview === 'function'/);
  assert.match(source, /Promise\.resolve\(\{\}\)/);
});

test('completed runs expose a CSV-derived model overview and replace the dashboard result', () => {
  assert.match(html, /id="model-input-overview"/);
  assert.match(viewsSource, /input_specs/);
  assert.match(viewsSource, /renderInputOverview/);
  assert.match(viewsSource, /input_file_overview/);
  assert.match(appSource, /result\.config\s*=\s*record\.config/);
  assert.match(appSource, /HISIM\.views\.setResult\(result\)/);
  assert.doesNotMatch(appSource, /find\(\(run\) => run\.status === 'completed'\) \|\| done\[0\]/);
});

test('input overviews present source, completeness, and hardware facts as separate fields', () => {
  assert.match(source, /input-mode-badge/);
  assert.match(source, /input-files-fact/);
  assert.match(viewsSource, /input-mode-badge/);
  assert.match(viewsSource, /input-files-fact/);
  assert.match(viewsSource, /input_mode_used/);
});

test('completed uploaded runs refresh the loaded result and logs before returning', () => {
  assert.match(appSource, /await refreshLogs\(runId\);\s*return result;/);
});

test('secondary critical-tile chart and noisy KPI tiles are removed', () => {
  assert.doesNotMatch(html, /Critical tiles by latency/);
  assert.doesNotMatch(viewsSource, /function drawTileLatency\(/);
  assert.doesNotMatch(viewsSource, /label: 'Simulation time'/);
  assert.doesNotMatch(viewsSource, /label: 'Perf \/ area'/);
});

test('download rows describe files as ready to download, not uploaded', () => {
  assert.match(source, /ready to download/);
  assert.match(source, /mode === 'download'/);
});
