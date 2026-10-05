import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const css = fs.readFileSync(new URL('../css/app.css', import.meta.url), 'utf8');
const views = fs.readFileSync(new URL('./views.js', import.meta.url), 'utf8');
const charts = fs.readFileSync(new URL('./charts.js', import.meta.url), 'utf8');
const app = fs.readFileSync(new URL('./app.js', import.meta.url), 'utf8');

test('light "datasheet and substrate" theme is the default; dark stays available', () => {
  assert.match(html, /<html lang="en" data-theme="light">/);
  assert.match(css, /:root\s*\{[^}]*--bg:\s*#E8EBEE/i);            // wafer
  assert.match(css, /:root\s*\{[^}]*--surface:\s*#FFFFFF/i);        // die
  assert.match(css, /:root\s*\{[^}]*--text:\s*#1C2533/i);           // ink
  assert.match(css, /:root\s*\{[^}]*--accent:\s*#2F6FB0/i);         // blue actions (user preference over green)
  assert.match(css, /:root\s*\{[^}]*--gold:\s*#B8932F/i);           // bond gold
  assert.match(css, /html\[data-theme="dark"\]\s*\{/);
  assert.match(app, /'hisim-theme'/);
});

test('one family, Archivo, with its width axis carrying the roles', () => {
  assert.match(html, /fonts\.googleapis\.com\/css2\?family=Archivo:wdth,wght@62\.\.125,100\.\.900/);
  assert.match(css, /--sans:\s*"Archivo"/);
  assert.match(css, /#kpi-grid \.kpi\.primary \.value\s*\{[^}]*font-stretch:\s*1[12]\d%/);   // wide headline figures
  assert.match(css, /table\.data\s*\{[^}]*font-stretch:\s*8\d(\.\d)?%/);                  // narrow dense tables
  assert.match(charts, /Archivo/);
});

test('no all-caps labels anywhere, and data labels use tabular numerals rather than a monospace face', () => {
  assert.doesNotMatch(css, /text-transform:\s*uppercase/);
  for (const sel of ['.config-summary', '.knob-range', '.field .val']) {
    const rule = new RegExp(`${sel.replace('.', '\\.').replace(' ', '\\s+')}\\s*\\{[^}]*font-family:\\s*var\\(--mono\\)`);
    assert.doesNotMatch(css, rule, `${sel} still uses the monospace face`);
  }
});

test('the results strip is a datasheet block: a plain-language headline over five key specs', () => {
  assert.match(html, /<div class="spec-block">\s*<p class="spec-headline" id="kpi-headline"><\/p>\s*<div class="kpi-grid" id="kpi-grid"><\/div>/);
  assert.match(views, /function splitUnit\(/);
  assert.match(views, /function packageHeadline\(/);
});

function loadViews() {
  const nodes = new Map();
  const node = () => ({ innerHTML: '', textContent: '', hidden: false, disabled: false, children: [], style: {},
    classList: { add() {}, remove() {}, toggle() {} }, dataset: {}, className: '',
    appendChild(c) { this.children.push(c); return c; }, addEventListener() {} });
  const el = (id) => { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id); };
  const noop = () => {};
  const sandbox = { document: { querySelector: () => null, querySelectorAll: () => [] }, Map, Math, Object, Number, String, JSON, Array, Set };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  sandbox.HISIM = {
    dom: { el, make: (_t, c, h = '') => ({ ...node(), className: c, innerHTML: h }), clear: (n) => { n.children = []; n.innerHTML = ''; }, esc: String },
    fmt: { seconds: (v) => `${v * 1000}ms`, joules: () => '1.75mJ', area: () => '70.66 mm²', usd: () => '$12.64', watts: () => '6.5mW',
      pct: () => '97.7%', num: (v) => String(v), si: String, title: String },
    charts: new Proxy({}, { get: () => noop }),
    architecture: { render: noop, drawNetworkGraph: noop },
  };
  vm.runInContext(fs.readFileSync(new URL('./thermal.js', import.meta.url), 'utf8'), sandbox);
  vm.runInContext(views, sandbox);
  return { views: sandbox.HISIM.views, el };
}

test('the headline names the workload and the package in words, and five specs lead', () => {
  const { views, el } = loadViews();
  views.setResult({ aimodel: 'resnet50', counts: { stacks: 4, tiers: 2, tiles: 50, chiplets: 8, ai_layers: 153 },
    metrics: { total_latency_s: 0.27, total_energy_j: 0.00175, total_area_mm2: 70.66, cost_per_part_usd: 12.64 },
    topology: { tiles: [], noc_edges: [] }, warnings: [], errors: [], cost: {}, input_specs: {} });
  assert.equal(el('kpi-headline').textContent, 'resnet50 on a 3.5D package: 4 stacks of 2 tiers, 50 tiles in 8 chiplets');
  const kpis = el('kpi-grid').children;
  assert.equal(kpis.filter((k) => /primary/.test(k.className)).length, 5);
  assert.match(kpis[0].innerHTML, /<span class="num">270<\/span><span class="unit">ms<\/span>/);
  assert.ok(kpis.every((k) => !/ · /.test(k.innerHTML)), 'no middle-dot meta strings in the spec block');
});

test('spec items read figure, label, note; primary-button text stays readable in both themes', () => {
  assert.doesNotMatch(css, /#kpi-grid \.kpi\.primary \.label \{ order:/);
  assert.match(css, /:root\s*\{[^}]*--on-accent:\s*#FFFFFF/i);
  assert.match(css, /html\[data-theme="dark"\]\s*\{[^}]*--on-accent:\s*#0D1B2C/i);
  assert.match(css, /\.btn\.primary, \.btn\.primary:hover \{[^}]*color: var\(--on-accent\)/);
});

test('each key figure sits in its own box', () => {
  assert.match(css, /#kpi-grid \{[^}]*gap: 10px/);
  assert.match(css, /#kpi-grid \.kpi \{[^}]*border: 1px solid var\(--line\)[^}]*border-radius: var\(--radius\)/);
});

test('the dashboard no longer shows the Model health card (warnings stay in the Logs tab)', () => {
  assert.doesNotMatch(html, /Model health|id="diagnostics"|id="diag-count"/);
  assert.doesNotMatch(views, /renderDiagnostics/);
});
