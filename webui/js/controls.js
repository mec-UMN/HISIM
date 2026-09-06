/* ============================================================
   Knob panel — rendered straight from /api/knobs
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.controls = (() => {
  const { el, make, clear, esc } = HISIM.dom;
  const { fmt } = HISIM;

  const ICONS = { brain: '◈', layers: '≡', cpu: '▦', database: '▤', network: '⌗', settings: '⚙' };

  let schema = null;
  let config = {};
  let defaults = {};
  let listeners = [];

  const DEFAULT_CONFIG = {
    aimodel: 'gpt2',
    create_default_files: true,
    default_files_generic: true,
    type_default_files: '2D_Mesh',
    set_suff_banks: true,
    stack_count: 4,
    chip_count: 1,
    tile_count_dict: { SA: 2, CPU: 2, Mem_I: 1, Mem_W: 1, Mem_O: 1 },
    sa_size_x: 16, sa_size_y: 16, n_sa: 2, precision_bits: 8, clock_hz: 1e9,
    n_bank: 1, n_word: 1024, n_bit: 320, col_mux: 4,
    n_2d_links_per_tile: 80, n_3d_links_per_tile: 80, n_2_5d_channels_per_chiplet_edge: 1,
  };

  function onChange(fn) { listeners.push(fn); }
  function emit() { listeners.forEach((fn) => fn(config)); }

  function get() { return JSON.parse(JSON.stringify(config)); }

  function set(patch, rerender = true) {
    config = { ...config, ...patch };
    if (rerender) render();
    emit();
  }

  function init(knobSchema, backendDefaults) {
    schema = knobSchema;
    defaults = backendDefaults || {};
    config = { ...DEFAULT_CONFIG };
    // seed GUI fields from the repo's config.py so the panel mirrors the CLI defaults
    schema.groups.forEach((group) => group.fields.forEach((field) => {
      const raw = defaults[field.config_name];
      if (raw === undefined || raw === null) return;
      if (field.type === 'bool') config[field.id] = Boolean(raw);
      else if (field.type === 'dict_int' && typeof raw === 'object') config[field.id] = { ...config[field.id], ...raw };
      else if (field.type === 'int' || field.type === 'select_int') config[field.id] = Number(raw);
      else if (field.type === 'select_float') config[field.id] = Number(raw);
      else config[field.id] = raw;
    }));

    render();
    renderSweepKnobs();
    emit();
  }

  function reset() {
    config = { ...DEFAULT_CONFIG };
    render();
    emit();
    HISIM.dom.toast('Knobs reset', 'Back to HISIM defaults.');
  }

  /* ---------------------------------------------------------------- render */
  function render() {
    const host = el('knob-groups');
    if (!host || !schema) return;
    const openState = {};
    host.querySelectorAll('details').forEach((node) => { openState[node.dataset.group] = node.open; });
    clear(host);

    schema.groups.forEach((group, index) => {
      const details = make('details', 'knob-group');
      details.dataset.group = group.id;
      details.open = openState[group.id] !== undefined ? openState[group.id] : index < 3;

      const summary = make('summary', '', `
        <span class="gicon">${ICONS[group.icon] || '•'}</span>
        <span>${esc(group.label)}</span>`);
      summary.title = group.description || '';
      details.appendChild(summary);

      // compact: no descriptions, no help paragraphs — everything lives in tooltips
      const body = make('div', 'group-body');
      group.fields.forEach((field) => body.appendChild(renderField(field)));
      details.appendChild(body);
      host.appendChild(details);
    });
  }

  /* Some knobs only take effect for certain other knob values (e.g. stack/tier
     counts are ignored while generic auto-mapping derives them from the AI
     model). disabled_when marks those so the panel doesn't imply a value is
     "wrong" when it was simply never applied. */
  function isDisabled(field) {
    const rule = field.disabled_when;
    if (!rule) return false;
    return config[rule.field] === rule.equals;
  }

  /* Every control is one compact row: label on the left, input on the right.
     Explanatory copy lives in the native tooltip so the panel stays scannable. */
  function renderField(field) {
    const value = config[field.id];
    const disabled = isDisabled(field);

    if (field.type === 'bool') {
      const row = make('div', 'row');
      row.title = field.help || '';
      row.appendChild(make('span', 'row-label', esc(field.label)));
      const label = make('label', 'switch');
      const input = make('input');
      input.type = 'checkbox';
      input.setAttribute('aria-label', field.label);
      input.checked = Boolean(value);
      input.addEventListener('change', () => set({ [field.id]: input.checked }, field.id === 'default_files_generic'));
      label.appendChild(input);
      label.appendChild(make('span'));
      row.appendChild(label);
      return row;
    }

    if (field.type === 'dict_int') {
      const wrap = make('div', `row stack${disabled ? ' disabled' : ''}`);
      wrap.title = disabled ? 'Ignored while generic auto-mapping is ON.' : (field.help || '');
      wrap.appendChild(make('span', 'row-label', esc(field.label)));
      const grid = make('div', 'tile-counts');
      field.keys.forEach((entry) => {
        const cell = make('label', 'tile-count');
        cell.title = entry.label;
        cell.appendChild(make('span', '', esc(entry.key.replace('Mem_', 'M'))));
        const input = make('input');
        input.type = 'number';
        input.setAttribute('aria-label', `${field.label} ${entry.label}`);
        input.min = field.min ?? 0;
        input.max = field.max ?? 9999;
        input.disabled = disabled;
        input.value = (value && value[entry.key]) ?? 1;
        input.addEventListener('change', () => {
          const clamped = Math.min(field.max ?? Infinity, Math.max(field.min ?? 0, Number(input.value) || 0));
          input.value = clamped;
          const next = { ...config[field.id], [entry.key]: clamped };
          set({ [field.id]: next }, false);
        });
        cell.appendChild(input);
        grid.appendChild(cell);
      });
      wrap.appendChild(grid);
      if (disabled) wrap.appendChild(make('span', 'knob-hint', 'auto-derived from the AI model'));
      return wrap;
    }

    if (field.type === 'select' || field.type === 'select_int' || field.type === 'select_float') {
      const row = make('div', 'row');
      const chosen = (field.options || []).find((o) => String(o.value) === String(value));
      row.title = (chosen && chosen.hint) || field.help || '';
      row.appendChild(make('span', 'row-label', esc(field.label)));
      const select = make('select');
      select.setAttribute('aria-label', field.label);
      (field.options || []).forEach((option) => {
        const node = make('option', '', esc(option.label));
        node.value = option.value;
        select.appendChild(node);
      });
      select.value = value;
      select.addEventListener('change', () => {
        const cast = field.type === 'select' ? select.value : Number(select.value);
        const patch = { [field.id]: cast };
        // A plain 3D mesh needs at least two tiers.  Keep the first-time path
        // runnable while still leaving advanced users free to edit the value
        // when they turn off generic auto-mapping.
        if (field.id === 'type_default_files' && cast === '3D_Mesh' && config.chip_count < 2) {
          patch.chip_count = 2;
        }
        set(patch, field.id === 'type_default_files');
      });
      row.appendChild(select);
      return row;
    }

    // numeric: number box with a hairline slider underneath for quick scrubbing
    const row = make('div', `row stack${disabled ? ' disabled' : ''}`);
    row.title = disabled ? 'Ignored while generic auto-mapping is ON.' : (field.help || '');

    const head = make('div', 'row-head');
    head.appendChild(make('span', 'row-label', esc(field.label)));
    if (field.min !== undefined && field.max !== undefined) {
      head.appendChild(make('span', 'knob-range', `${fmt.num(field.min)}–${fmt.num(field.max)}`));
    }

    const number = make('input', 'num');
    number.type = 'number';
    number.setAttribute('aria-label', `${field.label} value`);
    number.min = field.min ?? 1;
    number.max = field.max ?? 100000;
    number.step = field.step ?? 1;
    number.value = value;
    number.disabled = disabled;
    head.appendChild(number);
    row.appendChild(head);

    const range = make('input', 'scrub');
    range.type = 'range';
    range.setAttribute('aria-label', `${field.label} slider`);
    range.min = field.min ?? 1;
    range.max = field.max ?? 128;
    range.step = field.step ?? 1;
    range.value = value;
    range.disabled = disabled;
    row.appendChild(range);

    const sync = (next) => {
      const clamped = Math.min(field.max ?? Infinity, Math.max(field.min ?? 0, Number(next) || 0));
      range.value = clamped;
      number.value = clamped;
      set({ [field.id]: clamped }, false);
    };
    range.addEventListener('input', () => sync(range.value));
    number.addEventListener('change', () => sync(number.value));

    if (field.presets && field.presets.length && !disabled) {
      const presetRow = make('div', 'knob-presets');
      field.presets.forEach((preset) => {
        const chip = make('button', `chip mini-chip${Number(value) === preset ? ' active' : ''}`, String(preset));
        chip.type = 'button';
        chip.addEventListener('click', () => sync(preset));
        presetRow.appendChild(chip);
      });
      row.appendChild(presetRow);
    }
    if (disabled) row.appendChild(make('span', 'knob-hint', 'auto-derived from the AI model'));

    return row;
  }

  /* ---------------------------------------------------------------- sweeps */
  const knobLabels = () => {
    const labels = {};
    (schema?.groups || []).forEach((g) => g.fields.forEach((f) => { labels[f.id] = f.label; }));
    return labels;
  };

  // "Group · Field" — the full label shown in the sweep-knob dropdown itself.
  // Sent as knob_label on launch so a run's stored label always matches what
  // this dropdown displayed, keeping Runs & Compare's Knob filter consistent
  // with the panel a sweep was actually launched from.
  let sweepKnobGroupLabels = {};

  function renderSweepKnobs() {
    const select = el('sweep-knob');
    if (!select || !schema) return;
    clear(select);
    sweepKnobGroupLabels = {};
    schema.groups.forEach((g) => g.fields.forEach((f) => { sweepKnobGroupLabels[f.id] = `${g.label} · ${f.label}`; }));
    (schema.sweepable || []).forEach((knob) => {
      const option = make('option', '', esc(sweepKnobGroupLabels[knob] || knob));
      option.value = knob;
      select.appendChild(option);
    });
    select.value = 'stack_count';
    select.addEventListener('change', () => {
      updateSweepValueLabel(sweepKnobGroupLabels); renderSweepPresets(true); renderSweepModelChips(); updateSweepCount();
    });
    el('sweep-values').addEventListener('input', updateSweepCount);
    el('sweep-compare-models').addEventListener('change', () => {
      renderSweepModelChips();
      updateSweepCount();
    });
    updateSweepValueLabel(sweepKnobGroupLabels);
    renderSweepPresets();
    renderSweepModelChips();
    updateSweepCount();
  }

  /* ---------------------------------------------------- sweep: models --- */
  // Extra models only participate after the user explicitly turns on
  // comparison. The Design Knobs model always remains selected.
  let sweepModels = new Set();

  const sweepModelOptions = () => {
    const field = (schema?.groups || []).flatMap((g) => g.fields).find((f) => f.id === 'aimodel');
    return (field?.options || []).map((opt) => opt.value);
  };

  function compareModelsEnabled() {
    return el('sweep-compare-models')?.checked === true && el('sweep-knob')?.value !== 'aimodel';
  }

  function renderSweepModelChips() {
    const host = el('sweep-models');
    const field = el('sweep-models-field');
    const note = el('sweep-model-note');
    const toggleField = el('sweep-compare-models-field');
    if (!host || !field) return;
    const isModelKnob = el('sweep-knob')?.value === 'aimodel';
    if (toggleField) toggleField.hidden = isModelKnob;
    const enabled = compareModelsEnabled();
    field.hidden = !enabled;
    if (note) {
      note.textContent = isModelKnob
        ? 'Each AI model value is one comparison run; the multi-model switch is not needed.'
        : enabled
          ? 'Each selected model runs every listed value. The Design Knobs model is always included.'
          : 'Runs the model currently selected in Design Knobs. Enable comparison to add models.';
    }
    if (!enabled) return;

    sweepModels.add(config.aimodel);
    clear(host);
    sweepModelOptions().forEach((name) => {
      const isDesignModel = name === config.aimodel;
      const chip = make('button', `chip${sweepModels.has(name) ? ' active' : ''}`, esc(name));
      chip.type = 'button';
      chip.disabled = isDesignModel;
      chip.title = isDesignModel ? 'Selected in Design Knobs; always included.' : `Include ${name} in this comparison`;
      chip.addEventListener('click', () => {
        if (sweepModels.has(name)) sweepModels.delete(name);
        else sweepModels.add(name);
        renderSweepModelChips();
        updateSweepCount();
      });
      host.appendChild(chip);
    });
    const hint = el('sweep-models-hint');
    if (hint) hint.textContent = `${sweepModels.size} selected`;
  }

  function updateSweepValueLabel(groupLabels = {}) {
    const label = el('sweep-values-label');
    const knob = el('sweep-knob')?.value;
    if (label) label.textContent = groupLabels[knob] || knob || 'selected knob';
  }

  function parseValues(raw) {
    return (raw || '').split(',').map((token) => {
      const trimmed = token.trim();
      if (!trimmed) return null;
      return /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/.test(trimmed) ? Number(trimmed) : trimmed;
    }).filter((value) => value !== null);
  }

  function updateSweepCount() {
    const host = el('sweep-count');
    if (!host) return;
    const values = parseValues(el('sweep-values').value);
    const knob = el('sweep-knob').value;
    const knobLabel = knobLabels()[knob] || knob;
    const comparingModels = compareModelsEnabled();
    if (comparingModels) sweepModels.add(config.aimodel);
    const modelCount = comparingModels ? sweepModels.size : 1;
    const totalRuns = values.length * modelCount;
    const valueWord = `${values.length} value${values.length > 1 ? 's' : ''}`;

    let text = `enter at least one value for ${knobLabel}`;
    if (values.length) {
      if (knob === 'aimodel') text = `${valueWord}; each value is one model to compare`;
      else if (comparingModels) text = `${values.length} values × ${modelCount} models = ${totalRuns} runs`;
      else text = `${valueWord} for ${knobLabel} · ${config.aimodel}`;
    }
    host.textContent = text;
    host.classList.toggle('over', totalRuns > 96);
  }

  const SWEEP_PRESETS = {
    stack_count: ['1, 2, 4, 8, 16', '1, 2, 4, 8, 16, 32, 64'],
    chip_count: ['1, 2, 3, 4'],
    n_bank: ['1, 5, 9, 13, 17'],
    n_2d_links_per_tile: ['20, 40, 60, 80'],
    n_3d_links_per_tile: ['20, 40, 80, 160'],
    n_2_5d_channels_per_chiplet_edge: ['1, 2, 4, 8'],
    sa_size_x: ['8, 16, 32, 64, 128'],
    sa_size_y: ['8, 16, 32, 64, 128'],
    n_sa: ['1, 2, 4, 8, 16'],
    precision_bits: ['4, 8, 16, 32'],
    clock_hz: ['250000000, 500000000, 1000000000, 2000000000'],
    aimodel: ['mobilenetv2, resnet50, vitbase, gpt2', 'gpt2, llama, gemma1b, qwen0.6b'],
    type_default_files: ['2D_Mesh, 2_5D_Mesh, 3_5D_Mesh', '2_5D_Mesh, 2_5D_Mesh_Scaled, 3_5D_Mesh, 3_5D_Mesh_Scaled'],
    n_word: ['128, 256, 512, 1024'],
    n_bit: ['64, 160, 240, 320'],
    col_mux: ['4'],
  };

  function renderSweepPresets(resetValue = false) {
    const host = el('sweep-presets');
    const knob = el('sweep-knob').value;
    const input = el('sweep-values');
    if (!host) return;
    clear(host);
    const presets = SWEEP_PRESETS[knob] || [];
    presets.forEach((preset, index) => {
      const chip = make('button', 'chip', preset.length > 26 ? `${preset.slice(0, 24)}…` : preset);
      chip.type = 'button';
      chip.title = preset;
      chip.addEventListener('click', () => {
        input.value = preset;
        updateSweepCount();
      });
      host.appendChild(chip);
      if (index === 0 && (resetValue || !input.value)) input.value = preset;
    });
    if (presets[0]) input.placeholder = presets[0];
  }

  function sweepRequest() {
    const knob = el('sweep-knob').value;
    const values = parseValues(el('sweep-values').value);
    if (!values.length) throw new Error('Enter at least one sweep value.');
    const compare_models = compareModelsEnabled();
    if (compare_models) sweepModels.add(config.aimodel);
    const models = compare_models ? [...sweepModels] : [];
    const total = values.length * (compare_models ? models.length : 1);
    if (total > 96) {
      throw new Error(`That is ${total} runs — more than 96. Trim the value list.`);
    }
    return {
      knob,
      knob_label: sweepKnobGroupLabels[knob] || knobLabels()[knob] || knob,
      values,
      compare_models,
      models,
    };
  }

  const sweepEnabled = () => el('sweep-enable')?.checked === true;

  return { init, get, set, reset, onChange, sweepEnabled, sweepRequest };
})();
