/* ============================================================
   Result dashboards: summary strip · latency · energy · compute ·
   memory & DDR · interconnect · cost
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.views = (() => {
  const { el, make, clear, esc } = HISIM.dom;
  const { fmt, charts } = HISIM;

  let current = null;          // structured gui_result.json payload
  let latencyMode = null;

  const m = (key, fallback = null) => {
    const value = current?.metrics?.[key];
    return typeof value === 'number' ? value : fallback;
  };

  const LATENCY_PARTS = [
    ['Compute',      'compute_latency_s',   '--c1'],
    ['SRAM memory',  'memory_latency_s',    '--c3'],
    ['DDR',          'ddr_latency_s',       '--c4'],
    ['2D NoC',       'noc_2d_latency_s',    '--c2'],
    ['2.5D NoP',     'nop_2_5d_latency_s',  '--c6'],
    ['3D links',     'noc_3d_latency_s',    '--c5'],
  ];
  const ENERGY_PARTS = [
    ['Compute',      'compute_energy_j',    '--c1'],
    ['SRAM memory',  'memory_energy_j',     '--c3'],
    ['DDR',          'ddr_energy_j',        '--c4'],
    ['2D NoC',       'noc_2d_energy_j',     '--c2'],
    ['2.5D NoP',     'nop_2_5d_energy_j',   '--c6'],
    ['3D links',     'noc_3d_energy_j',     '--c5'],
  ];

  function setResult(payload) {
    current = payload;
    if (!payload) return;
    renderSummaryStrip();
    renderInputOverview();
    renderLatency();
    renderEnergy();
    renderComputeSection();
    renderMemorySection();
    renderNetwork();
    renderCost();
    HISIM.architecture.render(payload);
  }

  const getResult = () => current;

  function unique(rows, key) {
    return [...new Set((rows || []).map((row) => String(row?.[key] ?? '').trim()).filter(Boolean))].sort();
  }

  function numbers(rows, key) {
    return [...new Set((rows || []).map((row) => Number(row?.[key])).filter(Number.isFinite))].sort((a, b) => a - b);
  }

  function renderInputOverview() {
    const node = el('model-input-overview');
    const sourceNode = el('model-input-source');
    if (!node) return;

    const specs = current?.input_specs || {};
    const model = current?.aimodel || current?.config?.aimodel || 'selected model';
    const sourceKind = current?.input_mode_used || current?.input_file_overview?.source;
    const source = sourceKind === 'uploaded' ? 'uploaded custom CSVs'
      : sourceKind === 'mixed' ? 'uploaded + generated CSVs'
        : sourceKind === 'generated' ? 'generated HISIM CSVs'
          : current?.config?.create_default_files === false ? 'uploaded/custom CSVs' : 'generated HISIM CSVs';
    const requested = current?.input_mode_requested === 'uploaded' ? 'Uploaded-file run'
      : current?.input_mode_requested === 'generated' ? 'New generated run' : 'Run source not recorded';
    const chipRows = specs.chip_map || [];
    const sysRows = specs.sys_map || [];
    const layerRows = specs.layer_mapping || [];
    const saRows = specs.sa_spec || [];
    const memRows = specs.mem_spec || [];
    const networkRows = specs.network_spec || [];
    const hardware = {};
    chipRows.forEach((row) => {
      const type = String(row?.['HW Type'] ?? '').trim();
      if (type) hardware[type] = (hardware[type] || 0) + 1;
    });
    const hardwareText = Object.entries(hardware).map(([type, count]) => `${type} ×${count}`).join(', ') || 'hardware types unavailable';
    const sizes = [...new Set(saRows.map((row) => {
      const x = Number(row?.SA_size_x); const y = Number(row?.SA_size_y);
      return Number.isFinite(x) && Number.isFinite(y) ? `${x}×${y}` : null;
    }).filter(Boolean))].sort();
    const precisions = numbers(saRows, 'prec').map((value) => `INT${value}`).join('/');
    const banks = numbers(memRows, 'Nbank');
    const links = numbers(networkRows, 'N_2D_Links_per_tile').join('/');

    sourceNode.textContent = `${requested} · ${source}`;
    const facts = [
      ['Run source', `${requested} · verified ${source}`],
      ['Files', `${Object.keys(specs).length}/6 CSV files loaded`],
      ['Package', `${chipRows.length} tiles · ${unique(chipRows, 'Chiplet ID').length} chiplets · ${unique(sysRows, 'Stack ID').length} stacks · ${unique(sysRows, 'Tier ID').length} tiers`],
      ['Workload', `${unique(layerRows, 'Layer ID').length} AI layers`],
      ['Hardware', hardwareText],
      ['Compute', `${saRows.length} SA tiles${sizes.length ? ` · ${sizes.join(', ')}` : ''}${precisions ? ` · ${precisions}` : ''}`],
      ['Memory', `${memRows.length} memory tiles${banks.length ? ` · banks ${banks.join('/')}` : ''}`],
      ['Network', `${networkRows.length} network rows${links ? ` · 2D links ${links}` : ''}`],
    ];
    node.innerHTML = `<div class="model-overview-title"><strong>${esc(model)}</strong><span class="input-mode-badge">${esc(source)}</span></div><div class="model-overview-facts">${facts.map(([label, value]) => `<div class="input-files-fact"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join('')}</div>`;
  }

  /* ================================================  SUMMARY STRIP  ===== */
  function renderSummaryStrip() {
    el('overview-empty').hidden = true;
    el('overview-content').hidden = false;

    const latency = m('total_latency_s');
    const energy = m('total_energy_j');
    const area = m('total_area_mm2');
    const cost = m('cost_per_part_usd');
    const power = latency && energy ? energy / latency : null;
    const counts = current.counts || {};

    const kpis = [
      { label: 'End-to-end latency', value: fmt.seconds(latency), sub: latency ? `${fmt.num(1 / latency, 1)} inferences/s` : '', cls: '' },
      { label: 'Energy / inference', value: fmt.joules(energy), sub: power ? `≈ ${fmt.watts(power)} average power` : '', cls: 'accent-2' },
      { label: 'Total chip area', value: fmt.area(area), sub: `${counts.tiles ?? '—'} tiles · ${counts.chiplets ?? '—'} chiplets`, cls: '' },
      { label: 'Cost per part', value: fmt.usd(cost), sub: m('nre_cost_usd') ? `NRE ${fmt.usd(m('nre_cost_usd'))}` : '', cls: 'accent-3' },
      { label: 'Compute share', value: fmt.pct(latency && m('compute_latency_s') ? m('compute_latency_s') / latency : null), sub: 'of end-to-end latency', cls: '' },
      { label: 'AI layers mapped', value: counts.ai_layers ?? '—', sub: `${counts.stacks ?? '—'} stacks · ${counts.tiers ?? '—'} tiers`, cls: 'accent-2' },
    ];

    const host = el('kpi-grid');
    clear(host);
    kpis.forEach((kpi) => {
      host.appendChild(make('div', `kpi ${kpi.cls}`, `
        <div class="label">${esc(kpi.label)}</div>
        <div class="value">${kpi.value}</div>
        <div class="sub">${esc(kpi.sub || '')}</div>`));
    });

    drawStages();
    renderDiagnostics();
  }

  function drawStages() {
    const stages = current.stage_times || {};
    const keys = ['ai_mapping', 'compute', 'network', 'cost'];
    const values = keys.map((key) => stages[key] || 0);
    if (!values.some((v) => v > 0)) return charts.empty('plot-stages', 'no timing data');

    charts.draw('plot-stages', [{
      type: 'bar', orientation: 'h',
      x: values,
      y: ['AI mapping', 'Compute + memory', 'Network', 'Cost'],
      marker: { color: [charts.cssVar('--c1'), charts.cssVar('--c2'), charts.cssVar('--c3'), charts.cssVar('--c4')] },
      text: values.map((v) => `${v.toFixed(3)} s`),
      textposition: 'auto',
      hovertemplate: '%{y}: %{x:.3f} s<extra></extra>',
    }], { xaxis: { title: { text: 'seconds' } }, margin: { l: 118, r: 24, t: 20, b: 46 } });
  }

  // Error/warning log only — no per-item box styling, and nothing is shown
  // when the run is clean (the errors/warnings count in the card header
  // already says "0 errors · 0 warnings"; a reassuring checkmark row added
  // no information beyond that).
  function renderDiagnostics() {
    const host = el('diagnostics');
    clear(host);
    const warnings = current.warnings || [];
    const errors = current.errors || [];
    el('diag-count').textContent = current.partial
      ? 'recovered from logs · partial'
      : `${errors.length} errors · ${warnings.length} warnings`;

    if (current.partial) {
      host.appendChild(make('div', 'diag-line diag-line-warn',
        `⚑ ${esc(current.partial_reason || 'Only the printed summary was recovered for this run.')}`));
    }

    errors.slice(0, 12).forEach((message) => host.appendChild(make('div', 'diag-line diag-line-error', esc(message))));
    warnings.slice(0, 14).forEach((message) => host.appendChild(make('div', 'diag-line diag-line-warn', esc(message))));
    if (warnings.length > 14) host.appendChild(make('div', 'diag-line', `… ${warnings.length - 14} further warnings (see Logs tab)`));
  }

  const tiles = () => (current?.topology?.tiles || []).filter((tile) => (
    tile?.hw_class !== 'Empty' && String(tile?.hw_type || '').trim().toUpperCase() !== 'EMPTY'
  ));

  function partValues(parts) {
    return parts
      .map(([label, key, color]) => ({ label, value: m(key, 0) || 0, color: charts.cssVar(color, '#4cc2ff') }))
      .filter((entry) => entry.value > 0);
  }

  /* =====================================================  LATENCY  ===== */
  function renderLatency() {
    drawLatency(document.querySelector('[data-seg="latency-mode"] button.active')?.dataset.value || 'bar');
  }

  function drawLatency(mode) {
    const entries = partValues(LATENCY_PARTS);
    if (!entries.length) {
      latencyMode = null;
      return charts.empty('plot-latency', 'no latency data');
    }

    const nextMode = mode === 'pie' ? 'pie' : 'bar';
    // Pie and bar traces have incompatible Plotly state (pie has no axes,
    // while ranked bars use cartesian/log axes). A clean rebuild when moving
    // between them prevents a previous mode's axes or legend from leaking
    // into the next one.
    if (latencyMode && latencyMode !== nextMode) charts.purge('plot-latency');
    latencyMode = nextMode;

    if (nextMode === 'pie') {
      // Outside-positioned per-slice labels collide whenever several small
      // slices land near the same angle (exactly the case here: 2D NoC,
      // SRAM and DDR are all tiny next to Compute) — no fixed margin makes
      // that reliably safe. Put the label+value text in a real legend below
      // instead (predictable flow layout, same pattern as the bar chart's
      // legend), and keep each slice showing only a short percent.
      charts.draw('plot-latency', [{
        type: 'pie', hole: 0.55,
        labels: entries.map((e) => `${e.label} · ${fmt.seconds(e.value)}`),
        values: entries.map((e) => e.value),
        marker: { colors: entries.map((e) => e.color), line: { color: charts.cssVar('--surface'), width: 2 } },
        textinfo: 'percent',
        texttemplate: '%{percent:.1%}',
        textposition: 'inside',
        hovertemplate: '%{label}<br>%{percent:.2%}<extra></extra>',
      }], { legend: { y: -0.25 }, margin: { l: 20, r: 20, t: 20, b: 64 } });
      return;
    }

    // A stacked bar hides small contributors inside the dominant compute
    // segment. Rank each component as its own horizontal bar and use a log
    // axis so tiny but meaningful network/memory portions remain visible.
    const ranked = [...entries].sort((a, b) => a.value - b.value);
    charts.draw('plot-latency', [{
      type: 'bar', orientation: 'h',
      x: ranked.map((entry) => entry.value),
      y: ranked.map((entry) => entry.label),
      text: ranked.map((entry) => `${fmt.num(entry.value, 4)} s`),
      textposition: 'outside',
      cliponaxis: false,
      marker: { color: ranked.map((entry) => entry.color) },
      hovertemplate: '%{y}: %{text}<extra></extra>',
    }], {
      xaxis: { title: { text: 'latency (s, logarithmic)' }, type: 'log' },
      yaxis: { automargin: true },
      showlegend: false,
      margin: { l: 104, r: 96, t: 20, b: 58 },
    });
  }

  /* ======================================================  ENERGY  ===== */
  function renderEnergy() {
    drawEnergy();
  }

  function drawEnergy() {
    const entries = partValues(ENERGY_PARTS);
    if (!entries.length) return charts.empty('plot-energy', 'no energy data');
    charts.draw('plot-energy', [{
      type: 'bar',
      x: entries.map((e) => e.label),
      y: entries.map((e) => e.value),
      text: entries.map((e) => `${fmt.num(e.value, 4)} J`),
      marker: { color: entries.map((e) => e.color) },
      hovertemplate: '%{x}: %{text}<extra></extra>',
    }], {
      yaxis: { type: 'log', title: { text: 'joules (log)' } },
      margin: { l: 66, r: 20, t: 20, b: 60 },
    });
  }

  /* =====================================================  COMPUTE  ===== */
  function renderComputeSection() {
    drawTileScatter();
  }

  function drawTileScatter() {
    const data = tiles().filter((t) => t.area_mm2);
    if (!data.length) return charts.empty('plot-tiles', 'no tile data');

    const colorKey = el('tile-color').value;
    const sizeKey = el('tile-size').value;
    const groups = new Map();
    data.forEach((tile) => {
      const key = tile[colorKey] || 'unknown';
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(tile);
    });

    const sizes = data.map((t) => Number(t[sizeKey]) || 0);
    const maxSize = Math.max(...sizes, 1e-18);

    const traces = [...groups.entries()].map(([key, items], index) => ({
      type: 'scatter', mode: 'markers', name: String(key),
      x: items.map((t) => t.area_mm2),
      y: items.map((t) => Math.max(t.latency_s || 0, 1e-12)),
      text: items.map((t) => `${t.id}<br>${t.hw_type} · ${t.stack}<br>layer ${JSON.stringify(t.ai_layer)}`),
      customdata: items.map((t) => [fmt.num(t.area_mm2, 4), fmt.num(t.latency_s, 4)]),
      marker: {
        size: items.map((t) => 8 + 30 * Math.sqrt((Number(t[sizeKey]) || 0) / maxSize)),
        color: colorKey === 'hw_class' ? charts.hwColor(key) : charts.colorFor(index),
        opacity: 0.82,
        line: { width: 1, color: charts.cssVar('--surface') },
      },
      hovertemplate: '%{text}<br>area %{customdata[0]} mm²<br>latency %{customdata[1]} s<extra></extra>',
    }));

    charts.draw('plot-tiles', traces, {
      xaxis: { title: { text: 'tile area (mm²)' }, type: 'log' },
      yaxis: { title: { text: 'tile latency (s, log)' }, type: 'log' },
      margin: { l: 70, r: 24, t: 20, b: 54 },
    });
  }

  /* ================================================  MEMORY & DDR  ===== */
  function renderMemorySection() {
    drawAreaDonut();
    drawComputeSplit();
  }

  function drawAreaDonut() {
    const entries = [
      ['Compute', m('compute_area_mm2', 0), '--c1'],
      ['Memory', m('memory_area_mm2', 0), '--c3'],
      ['NoC routers', m('noc_area_mm2', 0), '--c2'],
      ['NoP routers', m('nop_router_area_mm2', 0), '--c6'],
      ['D2D interface', m('nop_interface_area_mm2', 0), '--c4'],
    ].filter(([, value]) => value > 0);

    charts.chartjs('chart-area', {
      type: 'doughnut',
      data: {
        labels: entries.map(([label]) => label),
        datasets: [{
          data: entries.map(([, value]) => value),
          backgroundColor: entries.map(([, , color]) => charts.cssVar(color, '#4cc2ff')),
          borderColor: charts.cssVar('--surface', '#131b2b'),
          borderWidth: 2,
          hoverOffset: 10,
        }],
      },
      options: {
        cutout: '58%',
        plugins: {
          legend: {
            position: 'right',
            // bake the value into the legend text itself — Chart.js's
            // default legend only shows the label, and the value is
            // otherwise only reachable by hovering the (sometimes tiny)
            // wedge.
            labels: {
              generateLabels: () => entries.map(([label, value, color], index) => ({
                text: `${label}: ${fmt.num(value, 2)} mm²`,
                fillStyle: charts.cssVar(color, '#4cc2ff'),
                strokeStyle: charts.cssVar(color, '#4cc2ff'),
                hidden: false,
                index,
              })),
            },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                return ` ${ctx.label}: ${ctx.parsed.toFixed(2)} mm² (${((ctx.parsed / total) * 100).toFixed(1)}%)`;
              },
            },
          },
        },
      },
    });
  }

  function drawComputeSplit() {
    const groups = ['Compute', 'SRAM memory', 'DDR'];
    const latency = [m('compute_latency_s', 0), m('memory_latency_s', 0), m('ddr_latency_s', 0)];
    const energy = [m('compute_energy_j', 0), m('memory_energy_j', 0), m('ddr_energy_j', 0)];
    const area = [m('compute_area_mm2', 0), m('memory_area_mm2', 0), 0];

    charts.draw('plot-compute-split', [
      { type: 'bar', name: 'Latency (s)', x: groups, y: latency, text: latency.map((v) => `${fmt.num(v, 4)} s`), marker: { color: charts.cssVar('--c1') },
        hovertemplate: '%{x} latency: %{text}<extra></extra>' },
      { type: 'bar', name: 'Energy (J)', x: groups, y: energy, text: energy.map((v) => `${fmt.num(v, 4)} J`), marker: { color: charts.cssVar('--c3') },
        hovertemplate: '%{x} energy: %{text}<extra></extra>' },
      { type: 'scatter', name: 'Area (mm²)', x: groups, y: area, yaxis: 'y2', mode: 'markers+lines',
        marker: { color: charts.cssVar('--c4'), size: 11, symbol: 'diamond' },
        hovertemplate: '%{x} area: %{y:.2f} mm²<extra></extra>' },
    ], {
      barmode: 'group',
      yaxis: { type: 'log', title: { text: 'latency (s) / energy (J), log' } },
      yaxis2: { overlaying: 'y', side: 'right', title: { text: 'area (mm²)' }, showgrid: false },
      margin: { l: 70, r: 62, t: 20, b: 50 },
    });
  }

  /* ================================================  INTERCONNECT  ====== */
  function renderNetwork() {
    drawNetworkTiers();
    drawNetworkArea();
    drawRouterArea();
    drawTraffic();
    renderNetWarnings();
    HISIM.architecture.drawNetworkGraph(current);
  }

  function drawNetworkTiers() {
    const labels = ['2D NoC', '2.5D NoP', '3D links'];
    const latency = [m('noc_2d_latency_s', 0), m('nop_2_5d_latency_s', 0), m('noc_3d_latency_s', 0)];
    const energy = [m('noc_2d_energy_j', 0), m('nop_2_5d_energy_j', 0), m('noc_3d_energy_j', 0)];

    charts.draw('plot-network', [
      { type: 'bar', name: 'Latency (s)', x: labels, y: latency, text: latency.map((v) => `${fmt.num(v, 4)} s`), marker: { color: charts.cssVar('--c1') },
        hovertemplate: '%{x} latency: %{text}<extra></extra>' },
      { type: 'scatter', name: 'Energy (J)', x: labels, y: energy, yaxis: 'y2',
        mode: 'markers+lines', marker: { color: charts.cssVar('--c5'), size: 13, symbol: 'star' },
        line: { dash: 'dot', width: 2 },
        text: energy.map((v) => `${fmt.num(v, 4)} J`),
        hovertemplate: '%{x} energy: %{text}<extra></extra>' },
    ], {
      yaxis: { title: { text: 'latency (s)' } },
      yaxis2: { overlaying: 'y', side: 'right', title: { text: 'energy (J)' }, showgrid: false, type: 'log' },
      margin: { l: 74, r: 74, t: 20, b: 50 },
    });
  }

  function drawNetworkArea() {
    const entries = [
      ['NoC routers', m('noc_area_mm2', 0), '--c1'],
      ['NoP routers', m('nop_router_area_mm2', 0), '--c2'],
      ['D2D (AIB) PHY', m('nop_interface_area_mm2', 0), '--c4'],
    ].filter(([, v]) => v > 0);

    charts.chartjs('chart-network-area', {
      type: 'bar',
      data: {
        labels: entries.map(([label]) => label),
        datasets: [{
          label: 'Area (mm²)',
          data: entries.map(([, value]) => value),
          backgroundColor: entries.map(([, , color]) => charts.cssVar(color, '#4cc2ff')),
          borderRadius: 6,
        }],
      },
      options: { plugins: { legend: { display: false } }, indexAxis: 'y' },
    });
  }

  function drawRouterArea() {
    const data = tiles().filter((t) => t.router_area_mm2)
      .sort((a, b) => (b.router_area_mm2 || 0) - (a.router_area_mm2 || 0)).slice(0, 20);
    if (!data.length) return charts.empty('plot-router', 'no router data');
    charts.draw('plot-router', [{
      type: 'bar',
      x: data.map((t) => t.id),
      y: data.map((t) => t.router_area_mm2),
      marker: { color: data.map((t) => charts.hwColor(t.hw_class)) },
      hovertemplate: '%{x}<br>router %{y:.5f} mm²<extra></extra>',
    }], {
      yaxis: { title: { text: 'router area (mm²)' } },
      xaxis: { tickangle: -45 },
      margin: { l: 70, r: 20, t: 18, b: 90 },
    });
  }

  function drawTraffic() {
    const edges = current?.topology?.noc_edges || [];
    if (!edges.length) {
      charts.empty('plot-traffic', 'no inter-tile transfers recorded');
      charts.empty('plot-hops', 'no hop data');
      return;
    }

    const top = [...edges]
      .filter((edge) => typeof edge.Q === 'number')
      .sort((a, b) => b.Q - a.Q)
      .slice(0, 18)
      .reverse();

    if (top.length) {
      charts.draw('plot-traffic', [{
        type: 'bar', orientation: 'h',
        x: top.map((edge) => edge.Q),
        y: top.map((edge) => `${edge.source} → ${edge.target}`),
        marker: {
          color: top.map((edge) => (edge.connection_type === '2d'
            ? charts.cssVar('--c1')
            : edge.connection_type === '3d' ? charts.cssVar('--c5') : charts.cssVar('--c6'))),
        },
        customdata: top.map((edge) => [edge.connection_type, edge.hops2d ?? 0, edge.hops2_5d ?? 0, edge.hops3d ?? 0]),
        hovertemplate: '%{y}<br>%{x:,.0f} bytes<br>type %{customdata[0]}<br>hops 2D %{customdata[1]} · 2.5D %{customdata[2]} · 3D %{customdata[3]}<extra></extra>',
      }], {
        xaxis: { title: { text: 'data volume (bytes, log)' }, type: 'log' },
        margin: { l: 168, r: 24, t: 18, b: 50 },
      });
    } else {
      charts.empty('plot-traffic', 'no data-volume information on the edges');
    }

    const series = [
      ['2D hops', 'hops2d', '--c1'],
      ['2.5D hops', 'hops2_5d', '--c6'],
      ['3D hops', 'hops3d', '--c5'],
    ].map(([name, key, color]) => ({
      type: 'histogram',
      name,
      x: edges.map((edge) => Number(edge[key]) || 0).filter((v) => v > 0),
      marker: { color: charts.cssVar(color) },
      opacity: 0.75,
      hovertemplate: `${name}: %{x} → %{y} transfers<extra></extra>`,
    })).filter((trace) => trace.x.length);

    if (series.length) {
      charts.draw('plot-hops', series, {
        barmode: 'overlay',
        xaxis: { title: { text: 'hops per transfer' } },
        yaxis: { title: { text: 'number of transfers' } },
        margin: { l: 66, r: 20, t: 18, b: 50 },
      });
    } else {
      charts.empty('plot-hops', 'all transfers are local (0 hops)');
    }
  }

  function renderNetWarnings() {
    const host = el('net-warnings');
    clear(host);
    const messages = (current.warnings || []).concat(current.errors || []);
    if (!messages.length) {
      host.appendChild(make('div', 'diag ok', '✓ All 2D / 2.5D / 3D links meet the required bandwidth.'));
      return;
    }
    messages.slice(0, 40).forEach((message) => {
      host.appendChild(make('div', `diag ${message.startsWith('Error') ? 'error' : 'warn'}`, esc(message)));
    });
  }

  /* =======================================================  COST  ======= */
  function flattenCost(node, prefix = 'Total') {
    const labels = [prefix];
    const parents = [''];
    const values = [0];
    // Every label's top-level ancestor (e.g. "Die · C3" -> "Die"), so the
    // sunburst can be colored — and legended — by top-level category
    // instead of Plotly's arbitrary per-wedge default palette.
    const topLevel = { [prefix]: null };

    const walk = (value, label, parent, top) => {
      topLevel[label] = top;
      if (typeof value === 'number') {
        labels.push(label); parents.push(parent); values.push(Math.max(value, 0));
        return Math.max(value, 0);
      }
      if (value && typeof value === 'object') {
        let sum = 0;
        labels.push(label); parents.push(parent); values.push(0);
        const index = values.length - 1;
        Object.entries(value).forEach(([key, child]) => {
          sum += walk(child, `${label} · ${key}`, label, top) || 0;
        });
        values[index] = sum;
        return sum;
      }
      return 0;
    };

    let total = 0;
    Object.entries(node || {}).forEach(([key, value]) => {
      const label = fmt.title(key);
      total += walk(value, label, prefix, label) || 0;
    });
    values[0] = total;
    return { labels, parents, values, topLevel };
  }

  function renderCost() {
    const cost = current.cost || {};
    const breakdown = cost.breakdown || {};
    const host = el('cost-kpis');
    clear(host);

    const kpis = [
      ['Cost per part', fmt.usd(m('cost_per_part_usd')), 'at the configured manufacturing volume'],
      ['Recurring cost', fmt.usd(m('recurring_cost_usd')), `${fmt.si(m('manufacturing_volume', 0))} units`],
      ['NRE cost', fmt.usd(m('nre_cost_usd')), 'masks · design · D2D IP'],
      ['Total programme cost', fmt.usd(cost.total_cost_usd), 'recurring + NRE'],
    ];
    kpis.forEach(([label, value, sub], index) => {
      host.appendChild(make('div', `kpi ${index % 2 ? 'accent-2' : ''}`, `
        <div class="label">${esc(label)}</div><div class="value">${value}</div><div class="sub">${esc(sub)}</div>`));
    });

    // Top-level cost categories (Die, Substrate, Interposer, NRE, ...) — used
    // for "Recurring cost by category" and as the sunburst's legend, since
    // small sunburst wedges are hard to hover: the legend gives every
    // category a large, easy-to-hover row with its exact value on hover.
    const categories = Object.entries(breakdown).map(([key, value]) => {
      const sum = (function total(node) {
        if (typeof node === 'number') return node;
        if (node && typeof node === 'object') return Object.values(node).reduce((acc, child) => acc + total(child), 0);
        return 0;
      })(value);
      return [fmt.title(key), sum];
    }).filter(([, value]) => value > 0).sort((a, b) => b[1] - a[1]);
    const categoryColor = new Map(categories.map(([label], index) => [label, charts.colorFor(index)]));

    const flat = flattenCost(breakdown);
    if (flat.values.length > 1) {
      const wedgeColors = flat.labels.map((label, index) => {
        if (index === 0) return charts.cssVar('--surface-2', '#212832'); // Total (root)
        return categoryColor.get(flat.topLevel[label]) || charts.colorFor(0);
      });
      charts.draw('plot-cost-sunburst', [{
        type: 'sunburst',
        labels: flat.labels,
        parents: flat.parents,
        values: flat.values,
        branchvalues: 'total',
        insidetextorientation: 'radial',
        marker: { colors: wedgeColors, line: { color: charts.cssVar('--surface'), width: 1.5 } },
        hovertemplate: '%{label}<br>$%{value:,.0f}<extra></extra>',
      }], { margin: { l: 10, r: 10, t: 10, b: 10 } });
      renderCostLegend(categories, categoryColor);
    } else {
      charts.empty('plot-cost-sunburst', 'no cost breakdown available');
      clear(el('cost-sunburst-legend'));
    }

    if (categories.length) {
      charts.draw('plot-cost-bars', [{
        type: 'bar', orientation: 'h',
        x: categories.map(([, value]) => value),
        y: categories.map(([label]) => label),
        marker: { color: categories.map((_, index) => charts.colorFor(index)) },
        hovertemplate: '%{y}: $%{x:,.0f}<extra></extra>',
      }], {
        xaxis: { title: { text: 'cost ($, log)' }, type: 'log' },
        margin: { l: 130, r: 24, t: 18, b: 50 },
      });
    } else {
      charts.empty('plot-cost-bars', 'no cost categories');
    }

    const perDie = cost.cost_per_die_usd;
    if (perDie && typeof perDie === 'object') {
      const entries = Object.entries(perDie).filter(([, v]) => typeof v === 'number');
      charts.draw('plot-cost-die', [{
        type: 'bar',
        x: entries.map(([key]) => key),
        y: entries.map(([, value]) => value),
        marker: { color: entries.map((_, index) => charts.colorFor(index)) },
        hovertemplate: '%{x}: $%{y:,.2f}<extra></extra>',
      }], { yaxis: { title: { text: 'cost per die ($)' } }, margin: { l: 68, r: 20, t: 18, b: 70 } });
    } else if (typeof perDie === 'number') {
      charts.draw('plot-cost-die', [{
        type: 'indicator', mode: 'number', value: perDie,
        number: { prefix: '$', valueformat: ',.2f' },
        title: { text: 'cost per die' },
      }], {});
    } else {
      charts.empty('plot-cost-die', 'no per-die cost data');
    }
  }

  function renderCostLegend(categories, categoryColor) {
    const host = el('cost-sunburst-legend');
    if (!host) return;
    clear(host);
    const total = categories.reduce((sum, [, value]) => sum + value, 0) || 1;
    const tip = charts.tooltip();
    categories.forEach(([label, value]) => {
      const row = make('div', 'sunburst-legend-row');
      const swatch = make('span', 'legend-swatch');
      swatch.style.background = categoryColor.get(label);
      row.appendChild(swatch);
      row.appendChild(make('span', 'legend-label', `${esc(label)} · ${fmt.usd(value)}`));
      row.addEventListener('mousemove', (event) => tip.show(
        `<b>${esc(label)}</b><br>${fmt.usd(value)}<br>${((value / total) * 100).toFixed(1)}% of recurring cost`,
        event,
      ));
      row.addEventListener('mouseleave', tip.hide);
      host.appendChild(row);
    });
  }

  function bindInteractions() {
    document.querySelectorAll('[data-seg="latency-mode"] button').forEach((button) => {
      button.addEventListener('click', () => {
        document.querySelectorAll('[data-seg="latency-mode"] button').forEach((b) => b.classList.remove('active'));
        button.classList.add('active');
        if (current) drawLatency(button.dataset.value);
      });
    });
    ['tile-color', 'tile-size'].forEach((id) => el(id).addEventListener('change', () => current && drawTileScatter()));
  }

  return { setResult, getResult, bindInteractions };
})();
