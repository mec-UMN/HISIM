import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const views = fs.readFileSync(new URL('./views.js', import.meta.url), 'utf8');
const thermal = fs.readFileSync(new URL('./thermal.js', import.meta.url), 'utf8');

test('the dashboard no longer has the stage-timing and per-die-cost cards', () => {
  for (const gone of ['Simulator stage timing', 'Per-die cost', 'plot-stages', 'plot-cost-die']) {
    assert.ok(!html.includes(gone), `index.html still mentions ${gone}`);
    assert.ok(!views.includes(gone), `views.js still mentions ${gone}`);
  }
  assert.doesNotMatch(views, /drawStages/);
});

test('the dashboard has a Thermal section with a run button', () => {
  assert.match(html, /<h2 class="section-title">Thermal<\/h2>/);
  for (const id of ['btn-run-thermal', 'thermal-kpis', 'plot-thermal-dies', 'thermal-body', 'thermal-empty', 'thermal-empty-reason', 'thermal-note', 'thermal-status']) {
    assert.match(html, new RegExp(`id="${id}"`));
  }
});

function load() {
  const draws = [];
  const nodes = new Map();
  const node = () => ({ innerHTML: '', textContent: '', hidden: false, disabled: false, children: [], style: {},
    classList: { add() {}, remove() {}, toggle() {} }, dataset: {},
    appendChild(c) { this.children.push(c); return c; }, addEventListener() {} });
  const el = (id) => { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id); };
  const noop = () => {};
  const sandbox = {
    document: { querySelector: () => null, querySelectorAll: () => [] },
    Map, Math, Object, Number, String, JSON, Array, Set,
  };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  sandbox.HISIM = {
    dom: { el, make: (_t, _c, h = '') => ({ ...node(), innerHTML: h }), clear: (n) => { n.children = []; n.innerHTML = ''; }, esc: String },
    fmt: new Proxy({}, { get: () => (v) => String(v) }),
    charts: new Proxy({}, { get: (_t, name) => (name === 'draw' ? (id, data, layout) => draws.push({ id, data, layout }) : name === 'cssVar' ? () => '#123' : noop) }),
    architecture: { render: noop, drawNetworkGraph: noop },
  };
  vm.runInContext(thermal, sandbox);
  vm.runInContext(views, sandbox);
  return { views: sandbox.HISIM.views, el, draws };
}

const base = { counts: {}, topology: { tiles: [], noc_edges: [] }, warnings: [], errors: [], cost: {}, input_specs: {},
  metrics: { total_latency_s: 1, total_energy_j: 1 }, run_id: 'r1' };
const text = (n) => `${n.innerHTML} ${n.children.map((c) => c.innerHTML).join(' ')}`;

test('thermal section shows KPIs against ambient and a per-die chart', () => {
  const { views, el, draws } = load();
  views.setResult({ ...base, thermal_status: 'done',
    metrics: { ...base.metrics, peak_temperature_k: 350, avg_temperature_k: 330, min_temperature_k: 320 },
    thermal: { ambient_k: 298, peak_k: 350, avg_k: 330, min_k: 320, solve_time_s: 12, unknowns: 150000, coarsen_factor: 1,
      dies: [{ die: 0, peak_k: 350, avg_k: 330, min_k: 320 }, { die: 1, peak_k: 340, avg_k: 325, min_k: 318 }] } });
  assert.equal(el('thermal-body').hidden, false);
  assert.equal(el('thermal-empty').hidden, true);
  assert.match(text(el('thermal-kpis')), /350/);
  assert.match(text(el('thermal-kpis')), /\+52\.0 K above 298 K ambient/);
  assert.match(text(el('thermal-kpis')), /Die 0 \(top\)/);          // hottest die is named
  const chart = draws.find((d) => d.id === 'plot-thermal-dies');
  assert.ok(chart, 'per-die chart drawn');
  const plain = (v) => JSON.parse(JSON.stringify(v));
  assert.deepEqual(plain(chart.data[0].y), ['Die 1', 'Die 0 (top)']);
  assert.deepEqual(plain(chart.data[0].x), [42, 52]);                  // peak rise above ambient
  assert.equal(el('btn-run-thermal').disabled, false);
});

test('thermal section offers the run button when there are no thermal results yet', () => {
  const { views, el } = load();
  views.setResult({ ...base, thermal_status: 'none' });
  assert.equal(el('thermal-body').hidden, true);
  assert.equal(el('thermal-empty').hidden, false);
  assert.match(el('thermal-empty-reason').textContent, /Run thermal/);
  assert.equal(el('btn-run-thermal').disabled, false);
});

test('the run button is disabled while a thermal pass is queued or running, and without a run', () => {
  const { views, el } = load();
  views.setResult({ ...base, thermal_status: 'running' });
  assert.equal(el('btn-run-thermal').disabled, true);
  assert.match(el('thermal-status').textContent, /running/);
  views.setResult({ ...base, run_id: undefined, thermal_status: 'none' });
  assert.equal(el('btn-run-thermal').disabled, true);
});

test('a skipped or failed thermal analysis shows the reason and coarsening is called out', () => {
  const { views, el } = load();
  views.setResult({ ...base, thermal_status: 'failed', thermal_error: 'ThermalTooLarge: Thermal grid too large (14,276,064 cells).' });
  assert.match(el('thermal-empty-reason').textContent, /Thermal grid too large/);
  views.setResult({ ...base, thermal_status: 'done', metrics: { ...base.metrics, peak_temperature_k: 300 },
    thermal: { peak_k: 300, avg_k: 299, min_k: 298, coarsen_factor: 2.5, dies: [] } });
  assert.match(el('thermal-note').textContent, /coarsened ×2\.5/);
});

test('Run thermal is disabled when the design is a layout the thermal model cannot solve', () => {
  const { views, el } = load();
  views.setResult({ ...base, thermal_status: 'failed',
    thermal_error: '3.5D packages (2.5D placement combined with 3D stacking) are not supported by the thermal model yet.' });
  assert.equal(el('btn-run-thermal').disabled, true);
  assert.match(el('btn-run-thermal').title, /not supported/);
});

test('the per-die chart keeps its legend clear of the axis title', () => {
  const { views, draws } = load();
  views.setResult({ ...base, thermal_status: 'done', metrics: { ...base.metrics, peak_temperature_k: 300 },
    thermal: { ambient_k: 298, peak_k: 300, avg_k: 299, min_k: 298, dies: [{ die: 0, peak_k: 300, avg_k: 299, min_k: 298 }] } });
  const layout = draws.find((d) => d.id === 'plot-thermal-dies').layout;
  assert.ok(layout.legend.y > 1, 'legend sits above the plot');
});

test('the KPI strip keeps the thermal note short and points to the Thermal section', () => {
  const { views, el } = load();
  views.setResult({ ...base, thermal_status: 'failed', thermal_error: 'A very long explanation '.repeat(10) });
  const html = el('kpi-grid').children.map((c) => c.innerHTML).join(' ');
  assert.match(html, /not available, see the Thermal section/);
  assert.doesNotMatch(html, /A very long explanation/);
});
