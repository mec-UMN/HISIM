import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('./views.js', import.meta.url), 'utf8');

function makeNode(value = '') {
  const listeners = new Map();
  return {
    value,
    textContent: '',
    innerHTML: '',
    hidden: false,
    style: {},
    children: [],
    classList: {
      add() {},
      remove() {},
    },
    appendChild(child) { this.children.push(child); return child; },
    addEventListener(type, listener) { listeners.set(type, listener); },
    click() { listeners.get('click')?.(); },
  };
}

function loadViews() {
  const nodes = new Map();
  const el = (id) => {
    if (!nodes.has(id)) nodes.set(id, makeNode(id === 'tile-color' ? 'hw_class' : id === 'tile-size' ? 'area_mm2' : ''));
    return nodes.get(id);
  };
  const bar = makeNode();
  bar.dataset = { value: 'bar' };
  bar.classList = { add() {}, remove() {} };
  const pie = makeNode();
  pie.dataset = { value: 'pie' };
  pie.classList = { add() {}, remove() {} };
  const calls = [];
  const HISIM = {
    dom: {
      el,
      make: (_tag, _className, html = '') => ({ ...makeNode(), innerHTML: html }),
      clear: (node) => { node.children = []; node.innerHTML = ''; },
      esc: (value) => String(value),
    },
    fmt: {
      seconds: (value) => `${value} s`, joules: (value) => `${value} J`, area: (value) => `${value} mm²`,
      usd: (value) => `$${value ?? 0}`, watts: (value) => `${value} W`, pct: (value) => `${value ?? 0}%`,
      num: (value) => String(value), si: (value) => String(value), title: (value) => String(value),
    },
    charts: {
      draw: (id, data) => calls.push({ action: 'draw', id, type: data[0]?.type }),
      purge: (id) => calls.push({ action: 'purge', id }),
      empty: (id) => calls.push({ action: 'empty', id }),
      chartjs() {}, cssVar: () => '#123456', hwColor: () => '#123456', colorFor: () => '#123456',
      tooltip: () => ({ show() {}, hide() {} }),
    },
    architecture: { render() {}, drawNetworkGraph() {} },
    thermal: { formatTemp: (v) => (typeof v === 'number' ? `${v} K` : '—'), isNum: (v) => typeof v === 'number' && Number.isFinite(v), statusLabel: (v) => v || '—', toC: (k) => k - 273.15 },
  };
  const document = {
    querySelector: (selector) => selector.includes('button.active') ? bar : null,
    querySelectorAll: (selector) => selector === '[data-seg="latency-mode"] button' ? [bar, pie] : [],
  };
  vm.runInNewContext(source, { window: { HISIM }, HISIM, document, Map, Math, Object, Number, String, JSON });
  return { views: HISIM.views, bar, pie, calls, el };
}

test('switching latency visualizations clears incompatible Plotly state before drawing', () => {
  const { views, bar, pie, calls } = loadViews();
  views.setResult({
    metrics: {
      total_latency_s: 10, compute_latency_s: 8, memory_latency_s: 1, ddr_latency_s: 1,
      compute_energy_j: 2, memory_energy_j: 1, ddr_energy_j: 1,
    },
    counts: {}, topology: { tiles: [], noc_edges: [] }, warnings: [], errors: [], cost: {}, input_specs: {},
  });
  views.bindInteractions();

  calls.length = 0;
  pie.click();
  assert.deepEqual(calls.filter((call) => call.id === 'plot-latency'), [
    { action: 'purge', id: 'plot-latency' },
    { action: 'draw', id: 'plot-latency', type: 'pie' },
  ]);

  calls.length = 0;
  bar.click();
  assert.deepEqual(calls.filter((call) => call.id === 'plot-latency'), [
    { action: 'purge', id: 'plot-latency' },
    { action: 'draw', id: 'plot-latency', type: 'bar' },
  ]);
});

test('summary strip shows peak temperature, or a hint when thermal was not run', () => {
  const { views, el } = loadViews();
  const base = { counts: {}, topology: { tiles: [], noc_edges: [] }, warnings: [], errors: [], cost: {}, input_specs: {} };
  const kpiHtml = () => el('kpi-grid').children.map((child) => child.innerHTML).join('\n');

  views.setResult({ ...base, metrics: { total_latency_s: 1, total_energy_j: 1, peak_temperature_k: 350 } });
  assert.match(kpiHtml(), /350\.0<\/span><span class="unit">K[\s\S]*Peak temperature/);

  views.setResult({ ...base, metrics: { total_latency_s: 1, total_energy_j: 1 } });
  assert.match(kpiHtml(), /—[\s\S]*Peak temperature[\s\S]*Thermal analysis/);

  views.setResult({ ...base, thermal_error: 'boom', metrics: { total_latency_s: 1, total_energy_j: 1 } });
  assert.match(kpiHtml(), /not available, see the Thermal section/);

  // the coarsening note lives in the Thermal section (dashboard.thermal.test.mjs), not in the headline strip
});
