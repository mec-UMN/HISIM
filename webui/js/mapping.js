/* ============================================================
   HISIM input files — download or upload the six CSVs used by a run.
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.mapping = (() => {
  const { el, make, clear, esc } = HISIM.dom;

  const KINDS = [
    ['chip_map', 'Chip Map', 'per-tile hardware type, NoC position, and AI layer assignment'],
    ['sys_map', 'System Map', 'chiplet → stack / tier / NoP organisation'],
    ['layer_map', 'Layer Map', 'AI layer → systolic array mapping parameters'],
    ['sa_spec', 'SA Spec', 'systolic-array geometry and precision per SA tile'],
    ['mem_spec', 'Memory Spec', 'memory capacity, banks, width, and precision per memory tile'],
    ['network_spec', 'Network Spec', 'per-stack network and link parameters'],
  ];

  let currentModel = null;
  let renderToken = 0;

  function acknowledge(message, state = '') {
    const node = el('mapping-file-ack');
    if (!node) return;
    node.className = `input-files-ack${state ? ` ${state}` : ''}`;
    node.textContent = message;
  }

  function fmtBytes(n) {
    if (n === null || n === undefined) return '';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }

  function sourceLabel(source) {
    return source === 'uploaded' ? 'uploaded custom CSVs'
      : source === 'mixed' ? 'uploaded + generated CSVs'
        : source === 'generated' ? 'generated CSVs' : 'templates only';
  }

  function renderOverview(summary) {
    const node = el('input-files-overview');
    if (!node) return;
    if (!summary || summary.source === 'template') {
      node.innerHTML = `<div class="input-files-empty">${esc(currentModel
        ? `${currentModel}: no complete input set yet. Download templates or run the new model.`
        : 'Select a model to inspect its input files.')}</div>`;
      return;
    }
    const hardware = Object.entries(summary.hardware_types || {})
      .map(([type, count]) => `${type} ×${count}`).join(', ') || 'hardware types not detected';
    const sa = summary.sa || {};
    const memory = summary.memory || {};
    const network = summary.network || {};
    const saText = sa.tiles ? `SA ${sa.tiles} tiles${sa.sizes?.length ? ` · ${sa.sizes.join(', ')}` : ''}${sa.precisions?.length ? ` · INT${sa.precisions.join('/')}` : ''}` : 'SA data not detected';
    const memoryText = memory.tiles ? `memory ${memory.tiles} tiles · ${memory.total_banks || 0} banks` : 'memory data not detected';
    const networkText = network.stacks ? `network ${network.stacks} stack${network.stacks > 1 ? 's' : ''} · 2D links ${network.noc_2d_links?.join('/') || '—'}` : 'network data not detected';
    const facts = [
      ['Files', `${summary.files_available}/${summary.files_total} available`],
      ['Package', `${summary.chiplets} chiplets · ${summary.tiles} tiles · ${summary.stacks} stacks · ${summary.tiers} tiers`],
      ['Workload', `${summary.ai_layers} AI layers`],
      ['Hardware', hardware],
      ['Compute', saText],
      ['Memory', memoryText],
      ['Network', networkText],
    ];
    node.innerHTML = `<div class="input-files-overview-title"><strong>${esc(summary.model)}</strong><span class="input-mode-badge">${esc(sourceLabel(summary.source))}</span></div><div class="input-files-facts">${facts.map(([label, value]) => `<div class="input-files-fact"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join('')}</div>`;
  }

  function renderRows(hostId, status, mode) {
    const host = el(hostId);
    if (!host) return;
    clear(host);
    KINDS.forEach(([kind, label, hint]) => {
      const info = status[kind] || { exists: false };
      const row = make('div', 'mapping-row');
      row.title = hint;

      const meta = make('div', 'mapping-meta');
      meta.appendChild(make('span', 'mapping-label', esc(label)));
      const statusText = info.exists
        ? `${fmtBytes(info.size_bytes)} · ${mode === 'download' ? 'ready to download' : info.source === 'uploaded' ? 'uploaded' : 'generated'}`
        : mode === 'download' ? 'template available to download' : 'waiting for upload';
      meta.appendChild(make('span', 'mapping-status', statusText));
      row.appendChild(meta);

      const actions = make('div', 'mapping-actions');
      if (mode === 'download') {
        const download = make('a', 'chip mini-chip', info.exists ? '⤓ Download' : '⤓ Template');
        download.href = HISIM.api.mappingFileDownloadUrl(currentModel, kind);
        download.setAttribute('download', '');
        download.title = info.exists
          ? `Download the current ${label} CSV`
          : `Download a ${label} CSV header template; fill it before uploading`;
        actions.appendChild(download);
      } else {
        const uploadButton = make('button', 'chip mini-chip upload-chip', '⤒ Choose CSV');
        uploadButton.type = 'button';
        uploadButton.setAttribute('aria-label', `Choose ${label} CSV to upload`);
        const input = make('input');
        input.type = 'file';
        input.accept = '.csv,text/csv';
        input.hidden = true;
        uploadButton.addEventListener('click', () => input.click());
        input.addEventListener('change', () => {
          if (input.files[0]) upload(kind, label, input.files[0]);
          input.value = '';
        });
        actions.appendChild(uploadButton);
        actions.appendChild(input);
      }
      row.appendChild(actions);
      host.appendChild(row);
    });
  }

  async function render(aimodel) {
    const token = ++renderToken;
    currentModel = aimodel || null;
    const downloadHost = el('mapping-download-files');
    const uploadHost = el('mapping-upload-files');
    if (!downloadHost && !uploadHost) return;
    if (!currentModel) {
      clear(downloadHost);
      clear(uploadHost);
      renderOverview(null);
      acknowledge('Select a model to see its input-file status.');
      return;
    }

    let status = {};
    let overview = null;
    // Keep the panel usable if an older cached API client is briefly paired
    // with this HTML.  The asset version on index.html prevents this in normal
    // use; the guard makes the transition fail soft instead of stopping all
    // rows and buttons from rendering.
    const statusRequest = typeof HISIM.api.mappingFiles === 'function'
      ? HISIM.api.mappingFiles(currentModel)
      : Promise.resolve({});
    const overviewRequest = typeof HISIM.api.mappingFilesOverview === 'function'
      ? HISIM.api.mappingFilesOverview(currentModel)
      : Promise.resolve({});
    const results = await Promise.allSettled([statusRequest, overviewRequest]);
    if (token !== renderToken || currentModel !== aimodel) return;
    if (results[0].status === 'fulfilled') status = results[0].value;
    if (results[1].status === 'fulfilled') overview = results[1].value;

    const uploaded = Object.values(status).filter((info) => info.source === 'uploaded').length;
    const generated = Object.values(status).filter((info) => info.source === 'generated').length;
    renderOverview(overview);
    if (uploaded) acknowledge(`${currentModel}: ${uploaded} uploaded file${uploaded > 1 ? 's' : ''} active; use “Run uploaded files” to validate them.`, 'ok');
    else if (generated) acknowledge(`${currentModel}: generated input files available; use “Run new model” to refresh them.`, '');
    else acknowledge(`${currentModel}: no run files yet; download templates or run the new model to create them.`, 'warn');

    renderRows('mapping-download-files', status, 'download');
    renderRows('mapping-upload-files', status, 'upload');
  }

  async function upload(kind, label, file) {
    if (!currentModel) return;
    const model = currentModel;
    acknowledge(`${model}: uploading ${label}…`, '');
    try {
      await HISIM.api.uploadMappingFile(model, kind, file);
      if (currentModel !== model) return;
      acknowledge(`${model}: ${label} uploaded and ready for “Run uploaded files”.`, 'ok');
      HISIM.dom.toast(
        'Input file uploaded',
        `${label} for ${model} is stored in uploaded_files. Use “Run uploaded files” to validate and run it.`,
        'ok',
      );
      render(model);
    } catch (error) {
      if (currentModel === model) {
        acknowledge(`${model}: ${label} upload failed — ${error.message}`, 'error');
        HISIM.dom.toast('Upload failed', error.message, 'error');
      }
    }
  }

  return { render, acknowledge };
})();
