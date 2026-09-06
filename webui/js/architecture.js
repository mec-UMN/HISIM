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
    populateChipletSelect();
    draw3D();
    drawFloorplan();
    drawStacks();
    renderTileTable();
    HISIM.mapping.render(result?.aimodel || result?.config?.aimodel);
  }

  /* -----------------------------------------------------  tile table  ----- */
  const TILE_COLUMNS = [
    ['id', 'Tile'], ['hw_type', 'Type'], ['chiplet', 'Chiplet'], ['stack', 'Stack'],
    ['ai_layer', 'AI layer'], ['area_mm2', 'Area (mm²)'], ['latency_s', 'Latency (s)'],
    ['energy_j', 'Energy (J)'], ['router_area_mm2', 'Router (mm²)'],
  ];

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
      const x = a[tileSort.key]; const y = b[tileSort.key];
      if (typeof x === 'number' && typeof y === 'number') return (x - y) * tileSort.dir;
      return String(x).localeCompare(String(y)) * tileSort.dir;
    });
  }

  function renderTileTable() {
    const table = el('table-tiles');
    const rows = filteredSortedTiles().slice(0, 400);

    table.innerHTML = `
      <thead><tr>${TILE_COLUMNS.map(([key, label]) =>
        `<th data-key="${key}">${esc(label)}${tileSort.key === key ? (tileSort.dir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr></thead>
      <tbody>${rows.map((tile) => `<tr>
          <td class="strong">${esc(tile.id)}</td>
          <td><span class="badge type">${esc(tile.hw_type)}</span></td>
          <td>${esc(tile.chiplet)}</td>
          <td>${esc(tile.stack)}</td>
          <td>${esc(Array.isArray(tile.ai_layer) ? tile.ai_layer.join(', ') : tile.ai_layer)}</td>
          <td>${fmt.num(tile.area_mm2, 4)}</td>
          <td>${fmt.num(tile.latency_s, 4)}</td>
          <td>${fmt.num(tile.energy_j, 4)}</td>
          <td>${fmt.num(tile.router_area_mm2, 4)}</td>
        </tr>`).join('')}</tbody>`;

    table.querySelectorAll('th').forEach((th) => th.addEventListener('click', () => {
      const key = th.dataset.key;
      tileSort = { key, dir: tileSort.key === key ? -tileSort.dir : -1 };
      renderTileTable();
    }));
  }

  function downloadTileCSV() {
    const rows = filteredSortedTiles();
    const csv = HISIM.csv.tilesToCSV(TILE_COLUMNS, rows);
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
    if (typeof window.HISIM3D?.render !== 'function') {
      draw3DFallback('Interactive renderer unavailable · compatibility view');
      return;
    }
    try {
      window.HISIM3D.render(container, payload, {
        metric: el('arch-metric').value,
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

  function bindInteractions() {
    // d3 panels are drawn to their container size, so redraw them when resized
    charts.registerRedraw('floorplan', () => payload && drawFloorplan());
    charts.registerRedraw('graph-network', () => payload && drawNetworkGraph(payload));

    el('arch-metric').addEventListener('change', () => {
      if (!payload) return;
      if (typeof window.HISIM3D?.setMetric === 'function') window.HISIM3D.setMetric(el('arch-metric').value);
      else draw3D();
    });
    el('arch-links').addEventListener('change', () => {
      if (!payload) return;
      if (typeof window.HISIM3D?.setLinksVisible === 'function') window.HISIM3D.setLinksVisible(el('arch-links').checked);
      else draw3D();
    });
    el('arch-reset').addEventListener('click', resetCamera);
    charts.registerRedraw('plot-arch3d', () => {
      if (typeof window.HISIM3D?.resize === 'function') window.HISIM3D.resize();
    });
    el('floorplan-chiplet').addEventListener('change', () => payload && drawFloorplan());
    el('net-labels').addEventListener('change', () => payload && drawNetworkGraph(payload));
    el('net-relayout').addEventListener('click', () => payload && drawNetworkGraph(payload));
    el('tile-search').addEventListener('input', () => payload && renderTileTable());
    el('tile-csv').addEventListener('click', downloadTileCSV);
  }

  return { render, drawNetworkGraph, drawFloorplan, bindInteractions };
})();
