/* ============================================================
   Finished sweep history, Pareto trade-off, and radar compare
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.runs = (() => {
  const { el, make, clear, esc } = HISIM.dom;
  const { fmt, charts } = HISIM;

  let records = [];
  let selected = new Set();
  let baselineId = null;
  let activeGroup = null;
  let activeModels = new Set();
  let activeModelsGroup = null;
  let sort = { key: 'created_at', dir: -1 };

  const AXES = [
    ['total_latency_s', 'End-to-end latency (s)'],
    ['total_energy_j', 'Energy per inference (J)'],
    ['average_power_w', 'Average power (W)'],
    ['total_area_mm2', 'Total chip area (mm²)'],
    ['cost_per_part_usd', 'Cost per part ($)'],
    ['throughput_inferences_per_s', 'Throughput (inf/s)'],
    ['compute_latency_s', 'Compute latency (s)'],
    ['memory_latency_s', 'Memory latency (s)'],
    ['ddr_latency_s', 'DDR latency (s)'],
    ['noc_2d_latency_s', '2D NoC latency (s)'],
    ['nop_2_5d_latency_s', '2.5D NoP latency (s)'],
    ['noc_3d_latency_s', '3D link latency (s)'],
    ['noc_area_mm2', 'NoC router area (mm²)'],
    ['total_sim_time', 'Simulator runtime (s)'],
  ];

  function init() {
    const xSelect = el('pareto-x');
    const ySelect = el('pareto-y');
    [[xSelect, 'total_area_mm2'], [ySelect, 'total_latency_s']].forEach(([select, initial]) => {
      clear(select);
      AXES.forEach(([key, label]) => {
        const option = make('option', '', esc(label));
        option.value = key;
        select.appendChild(option);
      });
      select.value = initial;
      select.addEventListener('change', () => drawPareto());
    });

    const sweepMetric = el('sweep-metric');
    clear(sweepMetric);
    AXES.forEach(([key, label]) => {
      const option = make('option', '', esc(label));
      option.value = key;
      sweepMetric.appendChild(option);
    });
    sweepMetric.value = 'total_latency_s';
    sweepMetric.addEventListener('change', drawSweep);

    el('compare-group-filter').addEventListener('change', (event) => {
      activeGroup = event.target.value || null;
      activeModels = new Set();
      activeModelsGroup = null;
      selected = new Set();
      renderCompareModelChecklist();
      syncBaselineOptions();
      renderTable();
      drawSweep();
      drawPareto();
      drawRadar();
    });

    el('run-search').addEventListener('input', renderTable);
    el('btn-refresh-runs').addEventListener('click', () => HISIM.app.refreshRuns());
    el('select-visible-runs').addEventListener('change', (event) => selectVisibleRuns(event.target.checked));
    el('btn-delete-selected').addEventListener('click', deleteSelectedRuns);
    el('radar-baseline').addEventListener('change', (event) => {
      baselineId = event.target.value || null;
      drawRadar();
    });
  }

  function setRecords(list) {
    // Runs & Compare is intentionally scoped to finished sweep children.
    // Standalone runs belong in the dashboard, but are not comparable sweep
    // points and used to make this view noisy and misleading.
    records = (list || []).filter((run) => (
      run.group && (run.status === 'completed' || run.status === 'failed')
    ));
    syncCompareFilters();
    // Row selection is explicit: it powers the scorecard and bulk deletion,
    // while the charts are scoped by the selected launch and model chips.
    selected = new Set([...selected].filter((id) => visibleRecords().some((r) => r.id === id)));
    syncBaselineOptions();
    renderTable();
    drawSweep();
    drawPareto();
    drawRadar();
  }

  function visibleRecords() {
    if (!activeGroup) return [];
    return records.filter((run) => (
      run.group === activeGroup && activeModels.has(run.sweep_model || run.config?.aimodel)
    ));
  }

  function modelsForGroup(group) {
    const set = new Set();
    records.forEach((run) => {
      if (run.group === group) set.add(run.sweep_model || run.config?.aimodel);
    });
    return set;
  }

  function numericSweepValue(value) {
    if (value === null || value === undefined || String(value).trim() === '') return null;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
  }

  function displaySweepValue(value) {
    const numeric = numericSweepValue(value);
    return numeric === null ? String(value ?? '—') : fmt.num(numeric, 3);
  }

  function syncCompareFilters() {
    const select = el('compare-group-filter');
    if (!select) return;

    const groups = new Map();
    records.forEach((run) => {
      if (!run.group) return;
      if (!groups.has(run.group)) {
        groups.set(run.group, {
          label: run.sweep_knob_label || run.sweep_knob || 'Sweep',
          createdAt: run.created_at,
          runs: 0,
          models: new Set(),
        });
      }
      const group = groups.get(run.group);
      group.runs += 1;
      group.models.add(run.sweep_model || run.config?.aimodel || 'unknown');
    });

    clear(select);
    groups.forEach((group, id) => {
      const option = make('option', '', esc(
        `${group.label} · ${group.models.size} model${group.models.size !== 1 ? 's' : ''} · ${group.runs} point${group.runs !== 1 ? 's' : ''} · ${fmt.ago(group.createdAt)}`
      ));
      option.value = id;
      select.appendChild(option);
    });

    if (!groups.has(activeGroup)) {
      activeGroup = groups.keys().next().value || null;
      activeModels = new Set();
      activeModelsGroup = null;
    }
    select.value = activeGroup || '';
    select.disabled = !groups.size;

    renderCompareModelChecklist();
  }

  function renderCompareModelChecklist() {
    const host = el('compare-model-filter');
    if (!host) return;
    const available = activeGroup ? modelsForGroup(activeGroup) : new Set();
    // A new launch starts with every model selected. Once the user changes a
    // checkbox, even an intentionally empty selection is preserved.
    if (activeModelsGroup !== activeGroup) {
      activeModels = new Set(available);
      activeModelsGroup = activeGroup;
    } else {
      activeModels = new Set([...activeModels].filter((model) => available.has(model)));
    }
    clear(host);
    const actions = make('div', 'model-filter-actions');
    const all = make('button', 'chip mini-chip', 'All');
    all.type = 'button';
    all.setAttribute('data-models-all', 'true');
    const none = make('button', 'chip mini-chip', 'None');
    none.type = 'button';
    none.setAttribute('data-models-none', 'true');
    actions.append(all, none);
    host.appendChild(actions);

    const list = make('div', 'model-checklist');
    [...available].sort().forEach((name) => {
      const checked = activeModels.has(name);
      const row = make('label', `model-check${checked ? ' selected' : ''}`);
      const input = make('input');
      input.type = 'checkbox';
      input.checked = checked;
      input.setAttribute('data-model-filter', name);
      input.setAttribute('aria-label', `Include ${name} in charts`);
      row.appendChild(input);
      row.appendChild(make('span', 'model-check-name', esc(name)));
      row.appendChild(make('span', 'model-check-state', checked ? 'Selected' : 'Hidden'));
      input.addEventListener('change', () => {
        if (input.checked) activeModels.add(name);
        else activeModels.delete(name);
        selected = new Set();
        refreshComparisonScope();
      });
      list.appendChild(row);
    });
    host.appendChild(list);
    all.addEventListener('click', () => {
      activeModels = new Set(available);
      selected = new Set();
      refreshComparisonScope();
    });
    none.addEventListener('click', () => {
      activeModels = new Set();
      selected = new Set();
      refreshComparisonScope();
    });
  }

  function refreshComparisonScope() {
    renderCompareModelChecklist();
    syncBaselineOptions();
    renderTable();
    drawSweep();
    drawPareto();
    drawRadar();
  }

  const completed = () => visibleRecords().filter((run) => run.status === 'completed' && run.metrics);

  function syncBaselineOptions() {
    const select = el('radar-baseline');
    if (!select) return;
    const candidates = completed();
    clear(select);
    candidates.forEach((run) => {
      const option = make('option', '', esc(run.name));
      option.value = run.id;
      select.appendChild(option);
    });
    const preferred = candidates.find((run) => run.id === baselineId)
      || candidates.find((run) => selected.has(run.id))
      || candidates[0];
    baselineId = preferred?.id || null;
    select.value = baselineId || '';
    select.disabled = !candidates.length;
  }

  /* ------------------------------------------------------  table  ------ */
  function renderTable() {
    const table = el('table-runs');
    const query = (el('run-search').value || '').toLowerCase();

    const columns = [
      ['sel', ''], ['name', 'Run'], ['status', 'Status'], ['aimodel', 'Model'],
      ['type', 'Topology'], ['stacks', 'Stacks×Tiers'], ['sweep', 'Sweep knob'], ['sweep_value', 'Value'],
      ['total_latency_s', 'Latency'],
      ['total_energy_j', 'Energy'], ['total_area_mm2', 'Area'], ['cost_per_part_usd', 'Cost/part'],
      ['created_at', 'When'], ['actions', ''],
    ];

    let rows = visibleRecords();
    if (query) rows = rows.filter((run) => JSON.stringify(run).toLowerCase().includes(query));
    rows = [...rows].sort((a, b) => {
      const pick = (run) => {
        switch (sort.key) {
          case 'name': return run.name;
          case 'status': return run.status;
          case 'aimodel': return run.config?.aimodel;
          case 'type': return run.config?.type_default_files;
          case 'stacks': return (run.config?.stack_count || 0) * 100 + (run.config?.chip_count || 0);
          case 'sweep': return run.sweep_knob_label || run.sweep_knob || '';
          case 'sweep_value': return numericSweepValue(run.sweep_value) ?? String(run.sweep_value ?? '');
          case 'created_at': return new Date(run.created_at).getTime();
          default: return (run.metrics || {})[sort.key] ?? -Infinity;
        }
      };
      const x = pick(a); const y = pick(b);
      if (typeof x === 'number' && typeof y === 'number') return (x - y) * sort.dir;
      return String(x).localeCompare(String(y)) * sort.dir;
    });

    table.innerHTML = `
      <thead><tr>${columns.map(([key, label]) =>
        key === 'sel'
          ? `<th data-key="${key}"><input type="checkbox" data-select-all ${rows.length && rows.every((run) => selected.has(run.id)) ? 'checked' : ''} aria-label="Select visible runs" /></th>`
          :
        `<th data-key="${key}">${esc(label)}${sort.key === key ? (sort.dir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr></thead>
      <tbody>${rows.map((run) => {
        const metrics = run.metrics || {};
        const config = run.config || {};
        return `<tr class="${selected.has(run.id) ? 'selected' : ''}" data-id="${run.id}">
          <td><input type="checkbox" data-select="${run.id}" ${selected.has(run.id) ? 'checked' : ''} /></td>
          <td class="strong">${esc(run.name)}</td>
          <td><span class="badge ${run.status}">${run.status}</span></td>
          <td>${esc(config.aimodel)}</td>
          <td><span class="badge type">${esc(config.type_default_files)}</span></td>
          <td>${config.stack_count ?? '—'}×${config.chip_count ?? '—'}</td>
          <td>${esc(run.sweep_knob_label || run.sweep_knob || '—')}</td>
          <td>${esc(displaySweepValue(run.sweep_value))}</td>
          <td>${fmt.seconds(metrics.total_latency_s)}</td>
          <td>${fmt.joules(metrics.total_energy_j)}</td>
          <td>${metrics.total_area_mm2 ? fmt.num(metrics.total_area_mm2, 1) : '—'}</td>
          <td>${fmt.usd(metrics.cost_per_part_usd)}</td>
          <td>${fmt.ago(run.created_at)}</td>
          <td>
            <button class="icon-btn" data-open="${run.id}" title="Load into dashboards" aria-label="Load ${esc(run.name)} into dashboards">⤢</button>
            <button class="icon-btn" data-delete="${run.id}" title="Delete run" aria-label="Delete ${esc(run.name)}">✕</button>
          </td>
        </tr>`;
      }).join('')}</tbody>`;

    table.querySelectorAll('th').forEach((th) => th.addEventListener('click', () => {
      const key = th.dataset.key;
      if (key === 'sel' || key === 'actions') return;
      sort = { key, dir: sort.key === key ? -sort.dir : -1 };
      renderTable();
    }));

    table.querySelectorAll('[data-select]').forEach((box) => box.addEventListener('change', () => {
      const id = box.dataset.select;
      if (box.checked) selected.add(id); else selected.delete(id);
      syncSelectionActions();
      renderTable(); drawRadar();
    }));

    table.querySelectorAll('[data-select-all]').forEach((box) => box.addEventListener('change', () => selectVisibleRuns(box.checked)));

    table.querySelectorAll('[data-open]').forEach((button) => button.addEventListener('click', () => {
      HISIM.app.loadRun(button.dataset.open);
    }));

    table.querySelectorAll('[data-delete]').forEach((button) => button.addEventListener('click', async () => {
      const id = button.dataset.delete;
      try {
        await HISIM.api.deleteRun(id);
        selected.delete(id);
        syncSelectionActions();
        HISIM.dom.toast('Run deleted', id, 'ok');
        HISIM.app.refreshRuns();
      } catch (error) {
        HISIM.dom.toast('Delete failed', error.message, 'error');
      }
    }));
    syncSelectionActions(rows);
  }

  function selectVisibleRuns(selectAll) {
    const rows = filteredTableRecords();
    rows.forEach((run) => {
      if (selectAll) selected.add(run.id);
      else selected.delete(run.id);
    });
    renderTable();
    drawRadar();
  }

  function filteredTableRecords() {
    const query = (el('run-search').value || '').toLowerCase();
    return query
      ? visibleRecords().filter((run) => JSON.stringify(run).toLowerCase().includes(query))
      : visibleRecords();
  }

  function syncSelectionActions(rows = filteredTableRecords()) {
    const count = selected.size;
    const button = el('btn-delete-selected');
    if (button) {
      button.textContent = `Delete selected (${count})`;
      button.disabled = count === 0;
    }
    const selectVisible = el('select-visible-runs');
    if (selectVisible) {
      selectVisible.checked = Boolean(rows.length) && rows.every((run) => selected.has(run.id));
      selectVisible.indeterminate = rows.some((run) => selected.has(run.id)) && !selectVisible.checked;
    }
  }

  async function deleteSelectedRuns() {
    const ids = [...selected].filter((id) => records.some((run) => run.id === id));
    if (!ids.length) return;
    if (!window.confirm(`Delete ${ids.length} selected run${ids.length === 1 ? '' : 's'}? This cannot be undone.`)) return;
    const outcomes = await Promise.allSettled(ids.map((id) => HISIM.api.deleteRun(id)));
    const removed = outcomes.filter((outcome) => outcome.status === 'fulfilled').length;
    ids.forEach((id) => selected.delete(id));
    if (removed) HISIM.dom.toast('Runs deleted', `${removed} run${removed === 1 ? '' : 's'} removed.`, 'ok');
    if (removed !== ids.length) HISIM.dom.toast('Some runs could not be deleted', 'Refresh and try again.', 'error');
    syncSelectionActions();
    HISIM.app.refreshRuns();
  }

  /* -----------------------------------------------  sweep response  ------ */
  function drawSweep() {
    const data = completed();
    const hint = el('sweep-plot-hint');
    if (!activeModels.size) {
      if (hint) hint.textContent = 'choose at least one model';
      return charts.empty('plot-sweep', 'Select at least one model');
    }
    if (!data.length) {
      if (hint) hint.textContent = 'swept knob on X-axis';
      return charts.empty('plot-sweep', 'No completed runs yet');
    }

    const metricKey = el('sweep-metric').value;
    const metricLabel = (AXES.find(([key]) => key === metricKey) || [, metricKey])[1];
    const knobLabel = data[0].sweep_knob_label || data[0].sweep_knob || 'Sweep value';

    const byModel = new Map();
    data.forEach((run) => {
      const model = run.sweep_model || run.config?.aimodel || 'unknown';
      if (!byModel.has(model)) byModel.set(model, []);
      byModel.get(model).push(run);
    });

    let numericX = true;
    const traces = [...byModel.entries()].map(([model, runs]) => {
      const points = runs.map((run) => {
        const numericValue = numericSweepValue(run.sweep_value);
        const y = run.metrics?.[metricKey];
        return {
          run, y,
          x: numericValue === null ? displaySweepValue(run.sweep_value) : numericValue,
          xText: displaySweepValue(run.sweep_value),
        };
      }).filter((point) => Number.isFinite(point.y));
      if (!points.every((point) => typeof point.x === 'number')) numericX = false;
      points.sort((a, b) => (typeof a.x === 'number' && typeof b.x === 'number')
        ? a.x - b.x
        : String(a.x).localeCompare(String(b.x), undefined, { numeric: true }));
      return { model, points };
    }).filter((series) => series.points.length);

    const totalPoints = traces.reduce((sum, series) => sum + series.points.length, 0);
    if (!totalPoints) {
      if (hint) hint.textContent = `${knobLabel} → ${metricLabel}`;
      return charts.empty('plot-sweep', `No completed values for ${metricLabel}`);
    }
    if (hint) {
      hint.textContent = `${knobLabel} → ${metricLabel} · ${totalPoints} completed`
        + (byModel.size > 1 ? ` · ${byModel.size} models` : '');
    }

    const allY = traces.flatMap((series) => series.points.map((point) => point.y));
    const useLog = allY.every((y) => y > 0);
    const uniqueY = [...new Set(allY)].sort((a, b) => a - b);
    const axisTicks = uniqueY.length <= 6
      ? { tickvals: uniqueY, ticktext: uniqueY.map((value) => fmt.num(value, 3)) }
      : {};

    const plotTraces = traces.map(({ model, points }) => ({
      type: 'scatter', mode: 'lines+markers', name: model,
      x: points.map((point) => point.x),
      y: points.map((point) => point.y),
      text: points.map((point) => `${point.run.name}<br>${knobLabel}: ${point.xText}<br>${metricLabel}: ${fmt.num(point.y, 4)}`),
      hovertemplate: '%{text}<extra></extra>',
      line: { color: charts.modelColor(model), width: 2 },
      marker: { size: 9, color: charts.modelColor(model), symbol: charts.modelMarker(model) },
    }));

    charts.draw('plot-sweep', plotTraces, {
      showlegend: byModel.size > 1,
      xaxis: { title: { text: knobLabel }, type: numericX ? 'linear' : 'category' },
      yaxis: { title: { text: metricLabel }, type: useLog ? 'log' : 'linear', ...axisTicks },
      margin: { l: 78, r: 24, t: 20, b: 56 },
    });
  }

  /* -----------------------------------------------------  pareto  ------ */
  function drawPareto() {
    const data = completed();
    if (!activeModels.size) return charts.empty('plot-pareto', 'Select at least one model');
    if (!data.length) return charts.empty('plot-pareto', 'No completed runs yet');

    const xKey = el('pareto-x').value;
    const yKey = el('pareto-y').value;
    const points = data
      .map((run) => ({ run, x: run.metrics[xKey], y: run.metrics[yKey] }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));

    if (!points.length) return charts.empty('plot-pareto', 'selected metrics are not available for these runs');

    // minimise both axes → Pareto front
    const sorted = [...points].sort((a, b) => a.x - b.x);
    const front = [];
    let best = Infinity;
    sorted.forEach((point) => {
      if (point.y < best) { front.push(point); best = point.y; }
    });

    const traces = [
      {
        type: 'scatter', mode: 'markers', name: 'runs',
        x: points.map((p) => p.x), y: points.map((p) => p.y),
        text: points.map((p) => `${p.run.name}<br>${p.run.config?.aimodel || '—'} · ${p.run.config?.type_default_files || '—'}<br>x ${fmt.num(p.x, 4)} · y ${fmt.num(p.y, 4)}`),
        marker: {
          size: points.map((p) => (selected.has(p.run.id) ? 17 : 11)),
          color: points.map((p) => charts.modelColor(p.run.sweep_model || p.run.config?.aimodel)),
          symbol: points.map((p) => charts.modelMarker(p.run.sweep_model || p.run.config?.aimodel)),
          opacity: 0.85, line: { width: 1, color: charts.cssVar('--surface') },
        },
        hovertemplate: '%{text}<extra></extra>',
      },
      {
        type: 'scatter', mode: 'lines+markers', name: 'Pareto front',
        x: front.map((p) => p.x), y: front.map((p) => p.y),
        line: { color: charts.cssVar('--c5'), width: 2, dash: 'dot' },
        marker: { size: 9, symbol: 'star', color: charts.cssVar('--c5') },
        hoverinfo: 'skip',
      },
    ];

    const label = (key) => (AXES.find(([k]) => k === key) || [, key])[1];
    // Some valid HISIM metrics are exactly zero for a topology that does not
    // use that tier (for example 2.5D latency in a monolithic 2D run).  A log
    // axis silently drops those points, making the trade-off chart look dead.
    // Keep every valid point and use a linear axis whenever either selected
    // metric contains zero or a negative value.
    const useLog = points.every((point) => point.x > 0 && point.y > 0);
    const axisTicks = (values) => {
      const unique = [...new Set(values.filter((value) => Number.isFinite(value)))].sort((a, b) => a - b);
      if (unique.length <= 6) return { tickvals: unique, ticktext: unique.map((value) => fmt.num(value, 3)) };
      const sample = [0, 1, 2, unique.length - 3, unique.length - 2, unique.length - 1]
        .map((index) => unique[index]);
      return { tickvals: [...new Set(sample)], ticktext: [...new Set(sample)].map((value) => fmt.num(value, 3)) };
    };
    charts.draw('plot-pareto', traces, {
      xaxis: { title: { text: label(xKey) }, type: useLog ? 'log' : 'linear', ...axisTicks(points.map((p) => p.x)) },
      yaxis: { title: { text: label(yKey) }, type: useLog ? 'log' : 'linear', ...axisTicks(points.map((p) => p.y)) },
      margin: { l: 78, r: 24, t: 20, b: 56 },
    });
  }

  /* ------------------------------------------------------  radar  ------ */
  const RADAR_KEYS = [
    ['total_latency_s', 'Latency', true],
    ['total_energy_j', 'Energy', true],
    ['total_area_mm2', 'Area', true],
    ['cost_per_part_usd', 'Cost', true],
    ['throughput_inferences_per_s', 'Throughput', false],
    ['noc_2d_latency_s', 'NoC latency', true],
  ];

  function drawRadar() {
    const chosen = completed().filter((run) => selected.has(run.id));
    const baseline = completed().find((run) => run.id === baselineId);
    if (!chosen.length || !baseline) {
      charts.chartjs('chart-radar', { type: 'radar', data: { labels: [], datasets: [] } });
      return;
    }

    const datasets = chosen.map((run, index) => ({
      label: run.name.length > 28 ? `${run.name.slice(0, 26)}…` : run.name,
      data: RADAR_KEYS.map(([key, , lowerIsBetter]) => {
        const value = run.metrics[key];
        const reference = baseline.metrics[key];
        if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0
          || typeof reference !== 'number' || !Number.isFinite(reference) || reference <= 0) return 0;
        const score = lowerIsBetter ? reference / value : value / reference;
        return Number(score.toFixed(3));
      }),
      rawValues: RADAR_KEYS.map(([key]) => run.metrics[key]),
      borderColor: charts.colorFor(index),
      backgroundColor: `${charts.colorFor(index)}22`,
      pointBackgroundColor: charts.colorFor(index),
      borderWidth: 2,
    }));

    const maxScore = Math.max(1.2, ...datasets.flatMap((dataset) => dataset.data).filter((value) => Number.isFinite(value)));
    const chartMax = Math.ceil(maxScore * 10) / 10;
    charts.chartjs('chart-radar', {
      type: 'radar',
      data: { labels: RADAR_KEYS.map(([, label]) => label), datasets },
      options: {
        plugins: {
          tooltip: { callbacks: { label: (ctx) => {
            const raw = ctx.dataset.rawValues?.[ctx.dataIndex];
            const rawText = typeof raw === 'number' ? ` · raw ${fmt.num(raw, 3)}` : '';
            return ` ${ctx.dataset.label}: ${(ctx.parsed.r * 100).toFixed(0)}% vs reference${rawText}`;
          } } },
        },
        scales: {
          r: {
            min: 0,
            max: chartMax,
            ticks: { callback: (value) => `${Math.round(value * 100)}%` },
            pointLabels: { font: { size: 11 } },
          },
        },
      },
    });
  }

  return { init, setRecords, selected: () => selected };
})();
