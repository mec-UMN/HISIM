/* ============================================================
   Architecture views — 3D package explorer, chiplet floorplan,
   interconnect graph, stack composition
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.architecture = (() => {
  const { el, make, clear, esc } = HISIM.dom;
  const { fmt, charts } = HISIM;

  let payload = null;
  let simulation = null;
  let tileSort = { key: 'latency_s', dir: -1 };

  const tiles = () => (payload?.topology?.tiles || []).filter((tile) => (
    tile?.hw_class !== 'Empty' && String(tile?.hw_type || '').trim().toUpperCase() !== 'EMPTY'
  ));
  const chiplets = () => payload?.topology?.chiplets || [];

  function render(result) {
    payload = result;
    showAllTiles = false;
    populateChipletSelect();
    draw3D();
    refreshThermal();
    drawFloorplan();
    drawStacks();
    renderTileTable();
    HISIM.mapping.render(result?.aimodel || result?.config?.aimodel);
  }

  /* -----------------------------------------------------  tile table  ----- */
  // Displayed columns: identity first, then the layer summary, then the numbers (kept on screen).
  const TILE_COLUMNS = [
    ['id', 'Tile'], ['hw_type', 'Type'], ['chiplet', 'Chiplet'], ['stack', 'Stack'],
    ['ai_layer', 'AI layers'], ['latency_s', 'Latency'], ['energy_j', 'Energy'],
    ['area_mm2', 'Area (mm²)'], ['router_area_mm2', 'Router (mm²)'],
  ];
  const NUMERIC_COLUMNS = new Set(['latency_s', 'energy_j', 'area_mm2', 'router_area_mm2']);
  // The CSV export keeps the full raw columns (complete layer lists, raw hardware type).
  const CSV_COLUMNS = [
    ['id', 'Tile'], ['hw_type', 'Type'], ['chiplet', 'Chiplet'], ['stack', 'Stack'],
    ['ai_layer', 'AI layer'], ['area_mm2', 'Area (mm²)'], ['latency_s', 'Latency (s)'],
    ['energy_j', 'Energy (J)'], ['router_area_mm2', 'Router (mm²)'],
  ];
  const TILE_ROW_LIMIT = 400;
  let showAllTiles = false;

  function tileSortValue(tile, key) {
    if (key === 'hw_type') return HISIM.tilefmt.typeLabel(tile);
    if (key === 'ai_layer') return HISIM.tilefmt.layerList(tile.ai_layer).length;
    return tile[key];
  }

  // Search against the identifying fields a user can actually see in the
  // table (id/type/chiplet/stack/layer) — not the raw JSON. Every tile
  // object carries keys like `ddr_latency_s` and `router_area_mm2`
  // regardless of its own type, so matching against JSON.stringify(tile)
  // would match a search term against property *names*, not values,
  // making most searches (e.g. "ddr", "router") match every tile.
  function tileSearchText(tile) {
    return [
      tile.id, tile.hw_type, tile.chiplet, tile.stack,
      Array.isArray(tile.ai_layer) ? tile.ai_layer.join(' ') : tile.ai_layer,
    ].filter(Boolean).join(' ').toLowerCase();
  }

  function filteredSortedTiles() {
    const query = (el('tile-search').value || '').toLowerCase();
    let rows = tiles();
    if (query) rows = rows.filter((tile) => tileSearchText(tile).includes(query));
    return [...rows].sort((a, b) => {
      const x = tileSortValue(a, tileSort.key); const y = tileSortValue(b, tileSort.key);
      if (typeof x === 'number' && typeof y === 'number') return (x - y) * tileSort.dir;
      return String(x).localeCompare(String(y)) * tileSort.dir;
    });
  }

  function renderTileTable() {
    const table = el('table-tiles');
    const all = filteredSortedTiles();
    const rows = showAllTiles ? all : all.slice(0, TILE_ROW_LIMIT);

    table.innerHTML = `
      <thead><tr>${TILE_COLUMNS.map(([key, label]) =>
        `<th data-key="${key}"${NUMERIC_COLUMNS.has(key) ? ' class="num"' : ''}>${esc(label)}${tileSort.key === key ? (tileSort.dir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr></thead>
      <tbody>${rows.map((tile) => {
        const layers = HISIM.tilefmt.layerSummary(HISIM.tilefmt.layerList(tile.ai_layer));
        const layerCell = layers.count
          ? `<strong>${layers.count}</strong> <span class="layer-preview">${esc(layers.preview)}</span>`
          : '<span class="layer-preview">none</span>';
        return `<tr>
          <td class="strong">${esc(tile.id)}</td>
          <td><span class="type-dot" style="background:${charts.hwColor(tile.hw_class || tile.hw_type)}"></span>${esc(HISIM.tilefmt.typeLabel(tile))}</td>
          <td>${esc(tile.chiplet)}</td>
          <td>${esc(tile.stack)}</td>
          <td class="layer-cell" title="${esc(layers.full || 'no AI layers mapped')}">${layerCell}</td>
          <td class="num">${fmt.seconds(tile.latency_s)}</td>
          <td class="num">${fmt.joules(tile.energy_j)}</td>
          <td class="num">${fmt.num(tile.area_mm2, 3)}</td>
          <td class="num">${fmt.num(tile.router_area_mm2, 3)}</td>
        </tr>`;
      }).join('')}</tbody>`;

    const truncated = all.length > rows.length;
    el('tile-count').textContent = truncated
      ? `${rows.length} of ${all.length} tiles shown (largest first by the sorted column)`
      : `${all.length} tile${all.length === 1 ? '' : 's'}`;
    const showAll = el('tile-show-all');
    showAll.hidden = !truncated;
    showAll.textContent = `Show all ${all.length}`;

    table.querySelectorAll('th').forEach((th) => th.addEventListener('click', () => {
      const key = th.dataset.key;
      tileSort = { key, dir: tileSort.key === key ? -tileSort.dir : -1 };
      renderTileTable();
    }));
  }

  function downloadTileCSV() {
    const rows = filteredSortedTiles();
    const csv = HISIM.csv.tilesToCSV(CSV_COLUMNS, rows);
    HISIM.csv.downloadCSV('hisim_tile_inventory.csv', csv);
  }

  /* -----------------------------------------------------------  3D  ---- */
  function draw3DFallback(message = '') {
    const container = el('plot-arch3d');
    container.querySelectorAll('canvas').forEach((canvas) => canvas.remove());
    const data = tiles().filter((tile) => Array.isArray(tile.noc) && Array.isArray(tile.nop));
    if (!data.length) {
      charts.empty('plot-arch3d', message || 'no package topology in this result');
      return;
    }
    charts.draw('plot-arch3d', [{
      type: 'scatter3d', mode: 'markers', name: 'tiles',
      x: data.map((tile) => tile.noc[0] + tile.nop[0] * 4),
      y: data.map((tile) => tile.nop[2] || 0),
      z: data.map((tile) => tile.noc[1] + tile.nop[1] * 4),
      text: data.map((tile) => `${tile.id}<br>${tile.hw_type || 'tile'}<br>${fmt.seconds(tile.latency_s)}`),
      hovertemplate: '%{text}<extra></extra>',
      marker: {
        size: 5,
        color: data.map((tile) => charts.hwColor(tile.hw_class)),
        opacity: 0.9,
      },
    }], {
      scene: {
        xaxis: { title: { text: 'NoC X / chiplet' } },
        yaxis: { title: { text: 'tier' } },
        zaxis: { title: { text: 'NoC Y / chiplet' } },
        aspectmode: 'data',
      },
      title: message ? { text: message, font: { size: 11 } } : undefined,
      margin: { l: 0, r: 0, t: 26, b: 0 },
    });
  }

  function draw3D() {
    const container = el('plot-arch3d');
    if (!tiles().length) {
      charts.empty('plot-arch3d', 'run a simulation to explore the package');
      return;
    }
    renderPackageLegend();
    if (typeof window.HISIM3D?.render !== 'function') {
      draw3DFallback('Interactive renderer unavailable · compatibility view');
      return;
    }
    try {
      window.HISIM3D.render(container, payload, {
        metric: el('arch-color-by').value,
        showLinks: el('arch-links').checked,
      });
    } catch (error) {
      console.warn('HISIM 3D renderer failed; using compatibility view.', error);
      draw3DFallback('Interactive renderer failed · compatibility view');
    }
  }

  function resetCamera() {
    if (typeof window.HISIM3D?.resetCamera === 'function') window.HISIM3D.resetCamera();
    else draw3D();
  }

  /* ---------------------------------------------------  floorplan  ----- */
  function populateChipletSelect() {
    const select = el('floorplan-chiplet');
    const ids = [...new Set(tiles().map((t) => t.chiplet))];
    const previous = select.value;
    clear(select);
    ids.forEach((id) => {
      const meta = chiplets().find((c) => c.id === id);
      const option = make('option', '', `${id}${meta ? ` · ${meta.stack} ${meta.tier}` : ''}`);
      option.value = id;
      select.appendChild(option);
    });
    if (ids.includes(previous)) select.value = previous;
  }

  function drawFloorplan() {
    const host = el('floorplan');
    clear(host);
    if (!window.d3 || !tiles().length) return;

    const chipletId = el('floorplan-chiplet').value || tiles()[0].chiplet;
    const data = tiles().filter((t) => t.chiplet === chipletId);
    if (!data.length) return;

    const width = host.clientWidth || 520;
    const height = host.clientHeight || 400;
    const maxX = Math.max(...data.map((t) => t.noc[0])) + 1;
    const maxY = Math.max(...data.map((t) => t.noc[1])) + 1;
    const pad = 26;
    const cell = Math.max(16, Math.min((width - pad * 2) / maxX, (height - pad * 2) / maxY));

    const svg = d3.select(host).append('svg').attr('viewBox', `0 0 ${width} ${height}`);
    const tip = charts.tooltip();

    const offsetX = (width - cell * maxX) / 2;
    const offsetY = (height - cell * maxY) / 2;

    const latencies = data.map((t) => t.latency_s || 0);
    const scale = d3.scaleSequential(d3.interpolateInferno).domain([0, Math.max(...latencies, 1e-12)]);

    const group = svg.append('g');

    group.selectAll('rect')
      .data(data)
      .join('rect')
      .attr('x', (d) => offsetX + d.noc[0] * cell + 2)
      .attr('y', (d) => offsetY + (maxY - 1 - d.noc[1]) * cell + 2)
      .attr('width', cell - 4)
      .attr('height', cell - 4)
      .attr('rx', 5)
      .attr('fill', (d) => charts.hwColor(d.hw_class))
      .attr('fill-opacity', 0.22)
      .attr('stroke', (d) => charts.hwColor(d.hw_class))
      .attr('stroke-width', 1.4)
      .style('cursor', 'pointer')
      .on('mousemove', (event, d) => tip.show(
        `<b>${esc(d.id)}</b><br>${esc(d.hw_type)}<br>layer ${esc(Array.isArray(d.ai_layer) ? d.ai_layer.join(', ') : d.ai_layer)}`
        + `<br>area ${fmt.num(d.area_mm2, 4)} mm²<br>latency ${fmt.seconds(d.latency_s)}<br>energy ${fmt.joules(d.energy_j)}`,
        event,
      ))
      .on('mouseleave', tip.hide)
      .transition().duration(420)
      .attr('fill-opacity', (d) => 0.18 + 0.55 * ((d.latency_s || 0) / Math.max(...latencies, 1e-12)));

    group.selectAll('text.label')
      .data(data)
      .join('text')
      .attr('class', 'label')
      .attr('x', (d) => offsetX + d.noc[0] * cell + cell / 2)
      .attr('y', (d) => offsetY + (maxY - 1 - d.noc[1]) * cell + cell / 2 + 3)
      .attr('text-anchor', 'middle')
      .attr('font-size', Math.max(7, Math.min(11, cell / 4.4)))
      .attr('fill', charts.cssVar('--text', '#e6edf7'))
      .attr('pointer-events', 'none')
      .text((d) => d.tile);

    svg.append('text')
      .attr('x', 12).attr('y', 20)
      .attr('font-size', 11)
      .attr('fill', charts.cssVar('--text-faint', '#63748f'))
      .text(`${chipletId} · ${data.length} tiles · ${maxX}×${maxY} NoC mesh · shade = latency`);

    void scale;
  }

  /* -----------------------------------------------  stack summary  ----- */
  function drawStacks() {
    const data = tiles();
    if (!data.length) return charts.empty('plot-stacks', 'no stack data');

    const stacks = [...new Set(data.map((t) => t.stack))].sort();
    const types = [...new Set(data.map((t) => t.hw_class))];

    const traces = types.map((type) => ({
      type: 'bar', name: type,
      x: stacks,
      y: stacks.map((stack) => data.filter((t) => t.stack === stack && t.hw_class === type).length),
      marker: { color: charts.hwColor(type) },
      hovertemplate: `%{x} · ${type}: %{y} tiles<extra></extra>`,
    }));

    const areaByStack = stacks.map((stack) =>
      data.filter((t) => t.stack === stack).reduce((acc, t) => acc + (t.area_mm2 || 0), 0));

    traces.push({
      type: 'scatter', name: 'Silicon area', x: stacks, y: areaByStack, yaxis: 'y2',
      mode: 'markers+lines', marker: { color: charts.cssVar('--c4'), size: 8 },
      line: { dash: 'dot' }, hovertemplate: '%{x}: %{y:.2f} mm²<extra></extra>',
    });

    charts.draw('plot-stacks', traces, {
      barmode: 'stack',
      yaxis: { title: { text: 'tiles' } },
      yaxis2: { overlaying: 'y', side: 'right', title: { text: 'area (mm²)' }, showgrid: false },
      xaxis: { title: { text: 'stack' } },
      margin: { l: 56, r: 62, t: 18, b: 50 },
    });
  }

  /* ------------------------------------------  interconnect graph  ----- */
  function drawNetworkGraph(result) {
    payload = result || payload;
    const host = el('graph-network');
    clear(host);
    if (!window.d3 || !chiplets().length) return;

    const width = host.clientWidth || 700;
    const height = host.clientHeight || 400;
    const showLabels = el('net-labels').checked;
    const tip = charts.tooltip();

    const tileCount = {};
    const areaByChiplet = {};
    tiles().forEach((t) => {
      tileCount[t.chiplet] = (tileCount[t.chiplet] || 0) + 1;
      areaByChiplet[t.chiplet] = (areaByChiplet[t.chiplet] || 0) + (t.area_mm2 || 0);
    });

    const nodes = chiplets().map((c) => ({
      id: c.id,
      stack: c.stack,
      tier: c.tier,
      nop: c.nop,
      tiles: tileCount[c.id] || 0,
      area: areaByChiplet[c.id] || 0,
      hasDDR: tiles().some((t) => t.chiplet === c.id && t.hw_class === 'DDR'),
    }));

    const index = new Map(nodes.map((n) => [`${n.nop[0]},${n.nop[1]},${n.nop[2]}`, n]));
    const links = [];
    nodes.forEach((n) => {
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach(([dx, dy, dz]) => {
        const neighbour = index.get(`${n.nop[0] + dx},${n.nop[1] + dy},${n.nop[2] + dz}`);
        if (neighbour) links.push({ source: n.id, target: neighbour.id, vertical: dz === 1 });
      });
    });

    const svg = d3.select(host).append('svg').attr('viewBox', `0 0 ${width} ${height}`);
    const container = svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.4, 4]).on('zoom', (event) => container.attr('transform', event.transform)));

    const maxArea = Math.max(...nodes.map((n) => n.area), 1);
    const radius = (n) => 9 + 20 * Math.sqrt(n.area / maxArea);

    const link = container.append('g').selectAll('line').data(links).join('line')
      .attr('stroke', (d) => (d.vertical ? charts.cssVar('--c5', '#ff6b81') : charts.cssVar('--line', '#22304a')))
      .attr('stroke-width', (d) => (d.vertical ? 2.6 : 1.6))
      .attr('stroke-dasharray', (d) => (d.vertical ? '4 3' : null));

    const node = container.append('g').selectAll('g').data(nodes).join('g').style('cursor', 'grab');

    node.append('circle')
      .attr('r', radius)
      .attr('fill', (d) => (d.hasDDR ? charts.cssVar('--c4', '#ffb547') : charts.cssVar('--c1', '#4cc2ff')))
      .attr('fill-opacity', 0.2)
      .attr('stroke', (d) => (d.hasDDR ? charts.cssVar('--c4', '#ffb547') : charts.cssVar('--c1', '#4cc2ff')))
      .attr('stroke-width', 1.8);

    if (showLabels) {
      node.append('text')
        .attr('text-anchor', 'middle').attr('dy', 3.5).attr('font-size', 10)
        .attr('fill', charts.cssVar('--text', '#e6edf7'))
        .text((d) => d.id);
      node.append('text')
        .attr('text-anchor', 'middle').attr('dy', (d) => radius(d) + 12).attr('font-size', 9)
        .attr('fill', charts.cssVar('--text-faint', '#63748f'))
        .text((d) => `${d.stack}·${d.tier}`);
    }

    node.on('mousemove', (event, d) => tip.show(
      `<b>${esc(d.id)}</b><br>${esc(d.stack)} · ${esc(d.tier)}<br>${d.tiles} tiles`
      + `<br>silicon ${fmt.num(d.area, 3)} mm²<br>NoP position ${d.nop.join(',')}`, event,
    )).on('mouseleave', tip.hide);

    if (simulation) simulation.stop();
    simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d) => d.id).distance(90).strength(0.6))
      .force('charge', d3.forceManyBody().strength(-320))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide().radius((d) => radius(d) + 12))
      .on('tick', () => {
        link.attr('x1', (d) => d.source.x).attr('y1', (d) => d.source.y)
          .attr('x2', (d) => d.target.x).attr('y2', (d) => d.target.y);
        node.attr('transform', (d) => `translate(${d.x},${d.y})`);
      });

    node.call(d3.drag()
      .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.25).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

    svg.append('text').attr('x', 12).attr('y', 20).attr('font-size', 11)
      .attr('fill', charts.cssVar('--text-faint', '#63748f'))
      .text(`${nodes.length} chiplets · ${links.length} package links (dashed = 3D / vertical) · drag & zoom`);
  }

  /* -------------------------------------------  package colour key  ----- */
  const METRIC_UNITS = {
    latency_s: ['Latency', (v) => fmt.seconds(v)], energy_j: ['Energy', (v) => fmt.joules(v)],
    area_mm2: ['Area', (v) => fmt.area(v)], router_area_mm2: ['Router area', (v) => fmt.area(v)],
  };

  function renderPackageLegend() {
    const host = el('arch-package-legend');
    const metric = el('arch-color-by').value;
    host.hidden = archView === 'thermal' || !tiles().length;
    if (host.hidden) return;
    const legend = window.HISIM3D.packageLegend(tiles(), metric);
    if (legend.kind === 'categorical') {
      host.innerHTML = '<span class="legend-title">Hardware type</span>' + legend.items.map((item) =>
        `<span class="legend-item"><i class="type-dot" style="background:${charts.hwColor(item.key)}"></i>${esc(item.key)}</span>`).join('');
      return;
    }
    const [title, format] = METRIC_UNITS[metric] || [metric, (v) => fmt.num(v, 3)];
    const ramp = window.HISIM3D.heatStops.map((stop) => `rgb(${stop.map((c) => Math.round(c * 255)).join(',')})`).join(',');
    host.innerHTML = `<span class="legend-title">${esc(title)}</span><span>${esc(format(legend.min))}</span>`
      + `<i class="legend-ramp" style="background:linear-gradient(90deg,${ramp})"></i><span>${esc(format(legend.max))}</span>`
      + '<span class="legend-note">per tile · colour scales from 0 to the largest tile</span>';
  }

  /* -------------------------------------------------  thermal layer  ----- */
  let thermalMaps = null;   // maps.json of the loaded run

  // hisim_thermal numbers dies from the heat-sink side: die 0 is the top die,
  // i.e. the highest tier of the package (verified against a real 2-tier run).
  function tierForDie(die) {
    const tiers = [...new Set(tiles().map((t) => t.nop[2] || 0))].sort((a, b) => b - a);
    return tiers[die] ?? 0;
  }

  const thermalDies = () => Object.keys(thermalMaps?.dies || {}).map(Number).sort((a, b) => a - b);
  const dieLabel = (d) => `Die ${d}${d === 0 ? ' (top)' : ''}`;

  let archView = 'package';
  let thermalReason = '';   // why the loaded run has no thermal map ('' when it has one)

  function setArchView(view) {
    archView = view;
    document.querySelectorAll('[data-seg="arch-view"] button').forEach((b) => {
      b.classList.toggle('active', b.dataset.value === view);
    });
    applyThermal();
  }

  async function refreshThermal() {
    thermalMaps = null;
    const hasThermal = HISIM.thermal.isNum(payload?.metrics?.peak_temperature_k);
    thermalReason = hasThermal ? '' : (payload?.thermal_error
      ? payload.thermal_error.replace(/^\w+: /, '')
      : 'This run was made without thermal analysis. Press Run thermal in the Thermal section of the dashboard to add it.');
    setArchView(archView);
    if (!hasThermal) return;
    const runId = payload.run_id;
    try {
      const maps = await HISIM.api.thermalMaps(runId);
      if (payload?.run_id !== runId) return;   // another run was loaded meanwhile
      thermalMaps = maps;
    } catch (_) {
      thermalReason = 'The thermal maps for this run could not be loaded.';
    }
    applyThermal();
  }

  function applyThermal() {
    renderPackageLegend();
    const legend = el('arch-thermal-legend');
    const note = el('arch-thermal-note');
    const on = archView === 'thermal' && thermalMaps;
    el('arch-color-field').hidden = archView === 'thermal';
    el('arch-thermal-png').hidden = !on;
    el('arch-thermal-csv').hidden = !on;
    el('arch-thermal-empty').hidden = !(archView === 'thermal' && !thermalMaps);
    el('arch-thermal-reason').textContent = thermalReason;
    const factor = payload?.thermal?.coarsen_factor;
    note.hidden = archView !== 'thermal' || !on || !(factor > 1);
    note.textContent = factor > 1 ? `Mesh coarsened ×${factor.toFixed(1)} to keep the solve fast — temperatures are approximate.` : '';
    if (!on) {
      window.HISIM3D.setThermal(null);
      legend.hidden = true;
      return;
    }
    const dies = thermalDies();
    const range = dies.map((d) => HISIM.thermal.mapRange(thermalMaps.dies[String(d)].grid));
    const min = Math.min(...range.map((r) => r.min));
    const max = Math.max(...range.map((r) => r.max));
    window.HISIM3D.setThermal({
      min, max,
      planes: dies.map((d) => ({ die: d, tier: tierForDie(d), grid: thermalMaps.dies[String(d)].grid })),
    });
    const ramp = [0, 0.25, 0.5, 0.75, 1].map((t) => `rgb(${HISIM.thermal.heatRgb(t).join(',')})`).join(',');
    legend.hidden = false;
    legend.innerHTML = `<span class="legend-title">Temperature</span><span>${min.toFixed(2)} K</span><i class="legend-ramp" style="background:linear-gradient(90deg,${ramp})"></i><span>${max.toFixed(2)} K</span><span class="legend-note">${dies.length} die${dies.length > 1 ? 's' : ''} · shared scale</span>`;
  }

  function saveBlob(name, blob) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function downloadThermalCsv() {
    const dies = thermalDies().map((d) => ({ die: d, grid: thermalMaps.dies[String(d)].grid }));
    const csv = HISIM.thermal.gridsToCsv(dies, thermalMaps.x_edges_mm, thermalMaps.y_edges_mm);
    saveBlob('thermal_all_dies.csv', new Blob([csv], { type: 'text/csv' }));
  }

  // One PNG with every die's map and a shared colour scale.
  function downloadThermalPng() {
    const dies = thermalDies();
    const grids = dies.map((d) => thermalMaps.dies[String(d)].grid);
    const ranges = grids.map((g) => HISIM.thermal.mapRange(g));
    const min = Math.min(...ranges.map((r) => r.min));
    const max = Math.max(...ranges.map((r) => r.max));
    const span = Math.max(max - min, 1e-9);
    const lay = HISIM.thermal.montageLayout(dies.length, grids[0][0].length, grids[0].length);
    const c = document.createElement('canvas');
    c.width = lay.width; c.height = lay.height;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.font = '12px sans-serif';
    grids.forEach((grid, k) => {
      const { x, y } = lay.positions[k];
      ctx.fillStyle = '#222';
      ctx.fillText(`${dieLabel(dies[k])} · peak ${ranges[k].max.toFixed(1)} K`, x, y + 12);
      grid.forEach((row, j) => row.forEach((v, i) => {
        ctx.fillStyle = `rgb(${HISIM.thermal.heatRgb((v - min) / span).join(',')})`;
        ctx.fillRect(x + i * lay.scale, y + lay.titleH + j * lay.scale, lay.scale, lay.scale);
      }));
    });
    const barX = lay.width - lay.barW + 8;
    const barH = lay.height - 2 * lay.pad - 16;
    for (let yy = 0; yy < barH; yy += 1) {
      ctx.fillStyle = `rgb(${HISIM.thermal.heatRgb(1 - yy / barH).join(',')})`;
      ctx.fillRect(barX, lay.pad + 8 + yy, 14, 1);
    }
    ctx.fillStyle = '#222';
    ctx.fillText(`${max.toFixed(1)} K`, barX + 18, lay.pad + 18);
    ctx.fillText(`${min.toFixed(1)} K`, barX + 18, lay.pad + 8 + barH);
    c.toBlob((blob) => saveBlob('thermal_all_dies.png', blob));
  }

  function bindInteractions() {
    // d3 panels are drawn to their container size, so redraw them when resized
    charts.registerRedraw('floorplan', () => payload && drawFloorplan());
    charts.registerRedraw('graph-network', () => payload && drawNetworkGraph(payload));

    el('arch-color-by').addEventListener('change', () => {
      if (!payload) return;
      if (typeof window.HISIM3D?.setMetric === 'function') window.HISIM3D.setMetric(el('arch-color-by').value);
      else draw3D();
      renderPackageLegend();
    });
    el('arch-links').addEventListener('change', () => {
      if (!payload) return;
      if (typeof window.HISIM3D?.setLinksVisible === 'function') window.HISIM3D.setLinksVisible(el('arch-links').checked);
      else draw3D();
    });
    el('arch-reset').addEventListener('click', resetCamera);
    document.querySelectorAll('[data-seg="arch-view"] button').forEach((button) => {
      button.addEventListener('click', () => { if (!button.disabled) setArchView(button.dataset.value); });
    });
    el('arch-thermal-goto').addEventListener('click', () => {
      document.querySelector('.tab[data-tab="design"]').click();
      setTimeout(() => el('thermal-section').scrollIntoView({ behavior: 'smooth', block: 'start' }), 60);
    });
    el('tile-show-all').addEventListener('click', () => { showAllTiles = true; renderTileTable(); });
    el('arch-thermal-png').addEventListener('click', downloadThermalPng);
    el('arch-thermal-csv').addEventListener('click', downloadThermalCsv);
    charts.registerRedraw('plot-arch3d', () => {
      if (typeof window.HISIM3D?.resize === 'function') window.HISIM3D.resize();
    });
    el('floorplan-chiplet').addEventListener('change', () => payload && drawFloorplan());
    el('net-labels').addEventListener('change', () => payload && drawNetworkGraph(payload));
    el('net-relayout').addEventListener('click', () => payload && drawNetworkGraph(payload));
    el('tile-search').addEventListener('input', () => payload && renderTileTable());
    el('tile-csv').addEventListener('click', downloadTileCSV);
  }

  return { render, drawNetworkGraph, drawFloorplan, bindInteractions, setView: (view) => setArchView(view) };
})();
