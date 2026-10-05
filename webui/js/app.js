/* ============================================================
   Application bootstrap: tabs, run lifecycle, polling, console
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.app = (() => {
  const { el, make, clear, esc, toast } = HISIM.dom;
  const { fmt, charts } = HISIM;

  let runRecords = [];
  let activeIds = [];
  let focusId = null;
  let pollTimer = null;
  let logTimer = null;
  let inputFilesModel = null;

  const QUICK = [
    { label: '2D monolithic · GPT-2', patch: { aimodel: 'gpt2', type_default_files: '2D_Mesh', default_files_generic: true, stack_count: 1, chip_count: 1 } },
    { label: '2.5D chiplets · ViT-B', patch: { aimodel: 'vitbase', type_default_files: '2_5D_Mesh', default_files_generic: true, stack_count: 3, chip_count: 1 } },
    { label: '3.5D stacks · Llama', patch: { aimodel: 'llama', type_default_files: '3_5D_Mesh', default_files_generic: true, stack_count: 3, chip_count: 2 } },
    { label: 'Scaled 2.5D · MobileNetV2', patch: { aimodel: 'mobilenetv2', type_default_files: '2_5D_Mesh_Scaled', default_files_generic: true } },
  ];

  /* ------------------------------------------------------  boot  ------- */
  async function boot() {
    bindShell();
    HISIM.views.bindInteractions();
    HISIM.architecture.bindInteractions();

    try {
      await HISIM.api.health();
      setBackend(true);
    } catch (error) {
      setBackend(false, error.message);
      toast('Backend unreachable', 'Start the API with: uvicorn app.main:app --reload (from backend/)', 'error');
      return;
    }

    try {
      const [knobs, models, defaults] = await Promise.all([
        HISIM.api.knobs(), HISIM.api.models(), HISIM.api.defaults(),
      ]);
      HISIM.controls.init(knobs, defaults);
      HISIM.controls.onChange((next) => {
        if (next.aimodel === inputFilesModel) return;
        inputFilesModel = next.aimodel;
        HISIM.mapping.render(next.aimodel);
      });
      inputFilesModel = HISIM.controls.get().aimodel;
      HISIM.mapping.render(HISIM.controls.get().aimodel);
      el('models-text').textContent = `${(models.models || []).length} AI models`;
      renderQuickConfigs();
    } catch (error) {
      toast('Failed to load knob schema', error.message, 'error');
    }

    HISIM.runs.init();
    await refreshRuns(true);

    charts.restoreSizes();
    charts.observeResizable();
    setTimeout(charts.resizeAll, 200);
  }

  function setBackend(ok, message) {
    const pill = el('pill-backend');
    pill.querySelector('.dot').className = `dot ${ok ? '' : 'error'}`;
    el('backend-text').textContent = ok ? 'API connected' : `API offline${message ? ` · ${message}` : ''}`;
  }

  function setRunStatus(text, state = 'idle') {
    el('pill-run').querySelector('.dot').className = `dot ${state}`;
    el('run-text').textContent = text;
    const busy = state === 'busy';
    ['btn-run', 'btn-run-new-model', 'btn-run-uploaded-files'].forEach((id) => {
      const button = el(id);
      if (button) button.disabled = busy;
    });
  }

  function renderQuickConfigs() {
    const host = el('quick-configs');
    clear(host);
    QUICK.forEach((quick) => {
      const button = make('button', 'btn ghost', esc(quick.label));
      button.addEventListener('click', () => {
        HISIM.controls.set(quick.patch);
        toast('Configuration loaded', quick.label, 'ok');
      });
      host.appendChild(button);
    });
  }

  /* Demo: a pre-computed bundle (3.5D resnet50 showcase, a 2.5D thermal example and three sweeps)
     installed by the backend from demo/ — nothing is simulated, so it works even if the engine is busy
     or broken in the middle of a presentation. Rebuild it with backend/scripts/build_demo.py. */
  let demoManifest = null;

  async function openDemo(view) {
    el('demo-menu').hidden = true;
    const tab = (name) => document.querySelector(`.tab[data-tab="${name}"]`).click();
    try {
      if (!demoManifest) {
        demoManifest = await HISIM.api.loadDemo();
        await refreshRuns();
      }
      if (view === 'sweeps') {
        tab('runs');
        const select = el('compare-group-filter');
        const group = demoManifest.sweep_groups?.[0];
        if (select && group) {
          select.value = group;
          select.dispatchEvent(new Event('change'));
        }
        toast('Demo sweeps', 'Pick another sweep in “Sweep launch”: SA count, stack count or 2.5D channels.', 'ok');
        return;
      }
      const runId = view === 'thermal' ? demoManifest.thermal : demoManifest.showcase;
      await loadRun(runId);
      // put the demo design in the knob panel too, so "Run simulation" re-runs exactly what is shown
      const demoConfig = runRecords.find((run) => run.id === runId)?.config;
      if (demoConfig) HISIM.controls.set(demoConfig, true);
      if (view === 'showcase') {
        tab('design');
      } else {
        tab('architecture');
        HISIM.architecture.setView(view === 'thermal' ? 'thermal' : 'package');
      }
    } catch (error) {
      toast('Demo not available', error.message, 'error');
    }
  }

  /* -------------------------------------------------  shell events  ---- */
  function bindShell() {
    document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      document.querySelectorAll('.view').forEach((v) => v.classList.remove('active'));
      tab.classList.add('active');
      el(`view-${tab.dataset.tab}`).classList.add('active');
      setTimeout(charts.resizeAll, 60);
      if (tab.dataset.tab === 'design' || tab.dataset.tab === 'architecture') {
        const result = HISIM.views.getResult();
        if (result) { HISIM.architecture.drawNetworkGraph(result); HISIM.architecture.drawFloorplan(); }
      }
    }));

    el('btn-run').addEventListener('click', launch);
    el('btn-run-new-model').addEventListener('click', () => launchWithMode('generated'));
    el('btn-run-uploaded-files').addEventListener('click', () => launchWithMode('uploaded'));
    el('btn-run-thermal').addEventListener('click', runThermalForLoadedRun);
    el('btn-reset').addEventListener('click', () => HISIM.controls.reset());
    el('btn-demo').addEventListener('click', (event) => {
      event.stopPropagation();
      el('demo-menu').hidden = !el('demo-menu').hidden;
    });
    document.querySelectorAll('[data-demo]').forEach((button) => {
      button.addEventListener('click', () => openDemo(button.dataset.demo));
    });
    document.addEventListener('click', (event) => {
      if (!event.target.closest('.demo-wrap')) el('demo-menu').hidden = true;
    });
    el('btn-expand-all').addEventListener('click', () => {
      const groups = document.querySelectorAll('.knob-group');
      const anyClosed = [...groups].some((g) => !g.open);
      groups.forEach((g) => { g.open = anyClosed; });
    });
    el('btn-panel').addEventListener('click', () => {
      document.querySelector('.shell').classList.toggle('collapsed');
      setTimeout(charts.resizeAll, 220);
    });
    el('btn-theme').addEventListener('click', toggleTheme);
    el('btn-layout').addEventListener('click', () => {
      charts.resetSizes();
      setPanelWidth(330);
      toast('Layout reset', 'Panel width and chart heights restored to defaults.');
    });

    bindSplitter();

    el('sweep-enable').addEventListener('change', (event) => {
      const enabled = event.target.checked;
      el('sweep-body').hidden = !enabled;
      el('sweep-mode-label').textContent = enabled
        ? 'on · multiple values'
        : 'off · one run';
      el('btn-run').textContent = enabled ? '▶ Launch sweep' : '▶ Run simulation';
    });

    el('log-stream').addEventListener('change', () => focusId && refreshLogs(focusId));
    el('log-copy').addEventListener('click', () => {
      navigator.clipboard.writeText(el('console').textContent || '')
        .then(() => toast('Copied', 'Console output copied to clipboard.', 'ok'))
        .catch(() => toast('Copy failed', 'Clipboard permission denied.', 'error'));
    });

    try {
      const stored = localStorage.getItem('hisim-theme');
      if (stored) document.documentElement.dataset.theme = stored;
    } catch (_) { /* storage unavailable */ }
  }

  /* ------------------------------------------------  resizable shell  --- */
  const PANEL_MIN = 240;
  const PANEL_MAX = 620;

  function setPanelWidth(px) {
    const width = Math.min(PANEL_MAX, Math.max(PANEL_MIN, px));
    document.documentElement.style.setProperty('--panel-w', `${width}px`);
    try { localStorage.setItem('hisim-panel-w', String(width)); } catch (_) { /* ignore */ }
  }

  function bindSplitter() {
    const splitter = el('splitter');
    if (!splitter) return;
    let dragging = false;

    const move = (event) => {
      if (!dragging) return;
      const shell = document.querySelector('.shell');
      setPanelWidth(event.clientX - shell.getBoundingClientRect().left);
    };
    const stop = () => {
      if (!dragging) return;
      dragging = false;
      splitter.classList.remove('dragging');
      document.body.classList.remove('resizing');
      setTimeout(charts.resizeAll, 120);
    };

    splitter.addEventListener('mousedown', (event) => {
      dragging = true;
      event.preventDefault();
      splitter.classList.add('dragging');
      document.body.classList.add('resizing');
    });
    splitter.addEventListener('dblclick', () => { setPanelWidth(330); setTimeout(charts.resizeAll, 120); });
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', stop);

    try {
      const stored = Number(localStorage.getItem('hisim-panel-w'));
      if (stored) setPanelWidth(stored);
    } catch (_) { /* ignore */ }
  }

  function toggleTheme() {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('hisim-theme', next); } catch (_) { /* ignore */ }
    const result = HISIM.views.getResult();
    if (result) HISIM.views.setResult(result);
    HISIM.runs.setRecords(runRecords);
    setTimeout(charts.resizeAll, 80);
  }

  /* ---------------------------------------------------  launching  ----- */
  function setInputFileAck(message, state = '') {
    HISIM.mapping?.acknowledge(message, state);
  }

  function launchWithMode(mode) {
    return launch(mode);
  }

  async function launch(mode = null) {
    const config = HISIM.controls.get();
    const explicitMode = mode === 'generated' || mode === 'uploaded';
    if (mode === 'generated') config.create_default_files = true;
    if (mode === 'uploaded') config.create_default_files = false;
    const modeLabel = mode === 'generated' ? 'new generated model' : mode === 'uploaded' ? 'uploaded input files' : 'selected configuration';
    try {
      if (!explicitMode && HISIM.controls.sweepEnabled()) {
        const { knob, knob_label, values, compare_models, models } = HISIM.controls.sweepRequest();
        const created = await HISIM.api.startSweep({ config, knob, knob_label, values, compare_models, models });
        activeIds = created.map((run) => run.id);
        focusId = activeIds[0];
        const modelNote = knob === 'aimodel'
          ? 'selected comparison models'
          : compare_models ? `${models.length} selected models` : config.aimodel;
        toast('Sweep launched', `${created.length} points · ${modelNote} · ${knob}`, 'ok');
        setInputFileAck(`${config.aimodel}: ${created.length}-point sweep queued. Input-file actions below remain single-run actions.`, 'ok');
        document.querySelector('.tab[data-tab="runs"]').click();
      } else {
        const created = await HISIM.api.startRun({ config });
        activeIds = [created.id];
        focusId = created.id;
        toast('Simulation started', `${created.name} · ${modeLabel}`, 'ok');
        setInputFileAck(`${config.aimodel}: ${modeLabel} queued. HISIM will validate inputs before simulation.`, 'ok');
      }
      setRunStatus('queued…', 'busy');
      startPolling();
      startLogPolling();
    } catch (error) {
      toast('Launch failed', error.message, 'error');
      setInputFileAck(`${config.aimodel}: launch failed — ${error.message}`, 'error');
      setRunStatus('launch failed', 'error');
    }
  }

  function startPolling() {
    clearInterval(pollTimer);
    pollTimer = setInterval(tick, 1600);
    tick();
  }

  async function tick() {
    await refreshRuns();
    const active = runRecords.filter((run) => activeIds.includes(run.id));
    if (!active.length) return;

    const done = active.filter((run) => run.status === 'completed' || run.status === 'failed');
    const running = active.find((run) => run.status === 'running') || active.find((run) => run.status === 'queued');

    if (running) {
      focusId = running.id;
      setRunStatus(
        active.length > 1 ? `running ${done.length + 1}/${active.length} · ${running.config.aimodel}` : `running · ${running.config.aimodel}`,
        'busy',
      );
      return;
    }

    clearInterval(pollTimer);
    clearInterval(logTimer);
    const failed = done.filter((run) => run.status === 'failed');
    setRunStatus(failed.length ? `${failed.length} failed` : `${done.length} run${done.length > 1 ? 's' : ''} complete`,
      failed.length ? 'error' : 'idle');

      const last = done.find((run) => run.status === 'completed');
      if (last) {
        const loadedResult = await loadRun(last.id);
        if (last.config?.aimodel) {
          const uploaded = last.config.create_default_files === false;
          const mode = uploaded ? 'uploaded-file run' : 'new generated run';
          const resultState = loadedResult
            ? `Dashboard updated from ${uploaded ? 'the uploaded CSVs' : 'newly generated HISIM files'}.`
            : 'The run completed, but its structured result could not be loaded; open Logs for details.';
          setInputFileAck(`${last.config.aimodel}: ${mode} completed. ${resultState}`, loadedResult ? 'ok' : 'error');
          HISIM.mapping.render(last.config.aimodel);
        }
        toast('Simulation complete', `${last.name} · ${last.config.create_default_files === false ? 'uploaded CSVs' : 'new generated files'}`, 'ok');
      } else if (done.length) {
        const failed = done[done.length - 1];
        const mode = failed.config?.create_default_files === false ? 'uploaded-file run' : 'new generated run';
        setInputFileAck(`${failed.config?.aimodel || 'Selected model'}: ${mode} failed — ${(failed.error || 'see Logs for details').split('\n')[0]}`, 'error');
        toast('Simulation failed', (failed.error || '').split('\n').slice(-2).join(' ').slice(0, 220), 'error');
    }
  }

  function startLogPolling() {
    clearInterval(logTimer);
    logTimer = setInterval(() => focusId && refreshLogs(focusId), 1800);
  }

  /* -----------------------------------------------------  loading  ----- */
  async function refreshRuns(autoload = false) {
    try {
      runRecords = await HISIM.api.runs();
      HISIM.runs.setRecords(runRecords);
      watchThermal();
      if (autoload) {
        const newest = runRecords.find((run) => run.status === 'completed');
        if (newest) await loadRun(newest.id, true);
      }
    } catch (error) {
      setBackend(false, error.message);
    }
    return runRecords;
  }

  // The dashboard's thermal pass is slow and runs in the background: keep
  // refreshing while any run has it queued/running, and reload the loaded
  // run once it finishes (or fails) so the dashboard and 3D page update.
  let thermalTimer = null;
  const thermalPending = new Set();
  function watchThermal() {
    const nowPending = new Set(runRecords
      .filter((run) => ['queued', 'running'].includes(run.thermal_status)).map((run) => run.id));
    thermalPending.forEach((id) => {
      if (nowPending.has(id)) return;
      const run = runRecords.find((r) => r.id === id);
      if (id === focusId) loadRun(id, true);
      if (run?.thermal_status === 'failed') toast('Thermal analysis failed', (run.thermal_error || '').slice(0, 200), 'error');
      if (run?.thermal_status === 'done') toast('Thermal analysis complete', run.name, 'ok');
    });
    thermalPending.clear();
    nowPending.forEach((id) => thermalPending.add(id));
    if (nowPending.size && !thermalTimer) {
      thermalTimer = setInterval(() => refreshRuns(), 5000);
    } else if (!nowPending.size && thermalTimer) {
      clearInterval(thermalTimer);
      thermalTimer = null;
    }
  }

  async function runThermalForLoadedRun() {
    const runId = focusId;
    if (!runId) return;
    try {
      await HISIM.api.runThermal(runId);
      toast('Thermal analysis started', 'Results appear on the dashboard and Architecture 3D when it finishes.', 'ok');
      await refreshRuns();
      await loadRun(runId, true);
    } catch (error) {
      toast('Thermal could not start', error.message, 'error');
    }
  }

  async function loadRun(runId, quiet = false) {
    focusId = runId;
    let result = null;
    try {
      result = await HISIM.api.result(runId);
      const record = runRecords.find((run) => run.id === runId);
      // the dashboard and 3D page need the run id (thermal maps, Run thermal) and the live thermal status
      result = { ...result, run_id: runId, thermal_status: record?.thermal_status };
      if (record) {
        result.config = record.config;
        HISIM.views.setResult(result);
        el('pill-models').innerHTML = `<span>${esc(record.config.aimodel)} · ${esc(record.config.type_default_files)}</span>`;
      } else HISIM.views.setResult(result);
      document.querySelectorAll(`#table-runs tr`).forEach((row) => row.classList.toggle('active-run', row.dataset.id === runId));
    } catch (error) {
      if (!quiet) {
        toast('No structured result', 'This run finished before the instrumented driver was available, or it failed early.', 'error');
      }
    }
    await refreshLogs(runId);
    return result;
  }

  async function refreshLogs(runId) {
    try {
      const stream = el('log-stream').value;
      const text = await HISIM.api.logs(runId, stream, 4000);
      const node = el('console');
      const record = runRecords.find((run) => run.id === runId);
      const label = el('log-run-label');
      if (label) label.textContent = record ? `${record.name} · ${record.status}` : runId;
      node.innerHTML = (text || '(no output yet)')
        .split('\n')
        .map((line) => {
          const safe = esc(line);
          if (/^\s*Error/.test(line)) return `<span class="l-error">${safe}</span>`;
          if (/^\s*Warning/.test(line)) return `<span class="l-warn">${safe}</span>`;
          if (/^-{3,}|^Total |sim time is:/.test(line)) return `<span class="l-key">${safe}</span>`;
          return safe;
        })
        .join('\n');
      if (el('log-follow').checked) node.scrollTop = node.scrollHeight;
    } catch (_) { /* run may not have logs yet */ }
  }

  document.addEventListener('DOMContentLoaded', () => setTimeout(boot, 60));

  return { refreshRuns, loadRun, runs: () => runRecords };
})();
