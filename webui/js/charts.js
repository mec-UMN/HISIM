/* ============================================================
   Charting helpers — Plotly (scientific), Chart.js (compact), D3 (custom)
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.charts = (() => {
  const cssVar = (name, fallback) => {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  };

  const palette = () => [
    cssVar('--c1', '#4cc2ff'), cssVar('--c2', '#7c5cff'), cssVar('--c3', '#16e0a3'),
    cssVar('--c4', '#ffb547'), cssVar('--c5', '#ff6b81'), cssVar('--c6', '#c084fc'),
    cssVar('--c7', '#38bdf8'), cssVar('--c8', '#facc15'),
  ];

  const colorFor = (index) => palette()[index % palette().length];

  const HW_VARS = {
    'SA': '--hw-sa',
    'CPU': '--hw-cpu',
    'Memory': '--hw-mem',
    'Weight mem': '--hw-wmem',
    'Output mem': '--hw-omem',
    'Input mem': '--hw-imem',
    'DDR': '--hw-ddr',
    'Empty': '--hw-empty',
  };
  const hwColor = (type) => cssVar(HW_VARS[type] || '--c8', '#ff8fc8');

  // stable colour + marker per AI model, so a model looks the same on every chart
  const modelOrder = [];
  const MODEL_MARKERS = ['square', 'triangle-up', 'diamond', 'circle', 'star', 'hexagon', 'cross', 'x', 'pentagon', 'triangle-down'];

  function modelIndex(model) {
    const key = String(model).toLowerCase();
    let index = modelOrder.indexOf(key);
    if (index === -1) { modelOrder.push(key); index = modelOrder.length - 1; }
    return index;
  }
  const modelColor = (model) => colorFor(modelIndex(model));
  const modelMarker = (model) => MODEL_MARKERS[modelIndex(model) % MODEL_MARKERS.length];
  const seedModels = (models) => (models || []).forEach(modelIndex);

  const config = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
    toImageButtonOptions: { format: 'png', scale: 3, filename: 'hisim' },
  };

  function layout(overrides = {}) {
    const text = cssVar('--text-dim', '#93a3bd');
    const grid = cssVar('--line-soft', '#1a2436');
    const base = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Inter, system-ui, sans-serif', size: 11, color: text },
      margin: { l: 62, r: 26, t: 26, b: 52 },
      hovermode: 'closest',
      hoverlabel: {
        bgcolor: cssVar('--surface', '#131b2b'),
        bordercolor: cssVar('--line', '#22304a'),
        font: { color: cssVar('--text', '#e6edf7'), size: 11 },
      },
      legend: { orientation: 'h', y: -0.18, font: { size: 10.5 } },
      xaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid, automargin: true },
      yaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid, automargin: true },
      colorway: palette(),
      transition: { duration: 260, easing: 'cubic-in-out' },
    };
    return deepMerge(base, overrides);
  }

  function deepMerge(target, source) {
    const out = { ...target };
    Object.entries(source || {}).forEach(([key, value]) => {
      if (value && typeof value === 'object' && !Array.isArray(value) && typeof out[key] === 'object') {
        out[key] = deepMerge(out[key], value);
      } else {
        out[key] = value;
      }
    });
    return out;
  }

  const drawn = new Set();

  function draw(nodeId, data, layoutOverrides = {}) {
    const node = document.getElementById(nodeId);
    if (!node || !window.Plotly) return;
    Plotly.react(node, data, layout(layoutOverrides), config);
    drawn.add(nodeId);
  }

  function purge(nodeId) {
    const node = document.getElementById(nodeId);
    if (node && window.Plotly) Plotly.purge(node);
    drawn.delete(nodeId);
  }

  function empty(nodeId, message) {
    const node = document.getElementById(nodeId);
    if (!node) return;
    if (window.Plotly) Plotly.purge(node);
    drawn.delete(nodeId);
    node.innerHTML = `<div style="height:100%;display:grid;place-items:center;color:var(--text-faint);font-size:12px">${message}</div>`;
  }

  function resizeAll() {
    if (!window.Plotly) return;
    drawn.forEach((id) => {
      const node = document.getElementById(id);
      if (node && node.offsetParent !== null) Plotly.Plots.resize(node);
    });
  }

  // ---- Chart.js ---------------------------------------------------------- #
  const chartRegistry = new Map();

  function chartjs(canvasId, spec) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !window.Chart) return null;
    if (chartRegistry.has(canvasId)) chartRegistry.get(canvasId).destroy();

    const text = cssVar('--text-dim', '#93a3bd');
    const grid = cssVar('--line-soft', '#1a2436');
    const defaults = {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500 },
      plugins: {
        legend: { labels: { color: text, boxWidth: 10, boxHeight: 10, font: { size: 10.5 } } },
        tooltip: {
          backgroundColor: cssVar('--surface', '#131b2b'),
          borderColor: cssVar('--line', '#22304a'),
          borderWidth: 1,
          titleColor: cssVar('--text', '#e6edf7'),
          bodyColor: text,
          padding: 10,
        },
      },
      scales: spec.type === 'doughnut' || spec.type === 'pie' || spec.type === 'radar' ? undefined : {
        x: { ticks: { color: text, font: { size: 10 } }, grid: { color: grid } },
        y: { ticks: { color: text, font: { size: 10 } }, grid: { color: grid } },
      },
    };
    if (spec.type === 'radar') {
      defaults.scales = {
        r: {
          angleLines: { color: grid },
          grid: { color: grid },
          pointLabels: { color: text, font: { size: 10 } },
          ticks: { display: false },
          suggestedMin: 0,
        },
      };
    }

    const chart = new Chart(canvas, {
      ...spec,
      options: deepMerge(defaults, spec.options || {}),
    });
    chartRegistry.set(canvasId, chart);
    return chart;
  }

  // ---- shared D3 tooltip ------------------------------------------------- #
  let tooltipNode = null;
  function tooltip() {
    if (!tooltipNode) {
      tooltipNode = document.createElement('div');
      tooltipNode.className = 'd3-tooltip';
      tooltipNode.style.display = 'none';
      document.body.appendChild(tooltipNode);
    }
    return {
      show(html, event) {
        tooltipNode.innerHTML = html;
        tooltipNode.style.display = 'block';
        tooltipNode.style.left = `${Math.min(event.clientX + 14, window.innerWidth - 260)}px`;
        tooltipNode.style.top = `${Math.max(12, event.clientY - 12)}px`;
      },
      hide() { tooltipNode.style.display = 'none'; },
    };
  }

  // ---- per-container resizing ------------------------------------------- #
  const redrawHandlers = new Map();   // element id -> callback (d3 panels)

  function registerRedraw(nodeId, callback) {
    redrawHandlers.set(nodeId, callback);
  }

  let observer = null;
  const debounce = new Map();

  function observeResizable() {
    if (!window.ResizeObserver) return;
    if (!observer) {
      observer = new ResizeObserver((entries) => {
        entries.forEach((entry) => {
          const node = entry.target;
          const id = node.id;
          clearTimeout(debounce.get(node));
          debounce.set(node, setTimeout(() => {
            if (node.offsetParent === null) return;
            if (id && drawn.has(id) && window.Plotly) Plotly.Plots.resize(node);
            const handler = id && redrawHandlers.get(id);
            if (handler) handler();
            persist(node);
          }, 90));
        });
      });
    }
    document.querySelectorAll('.resizable').forEach((node) => observer.observe(node));
  }

  // remember the heights the user dragged to
  const STORE_KEY = 'hisim-panel-sizes';

  function readStore() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); } catch (_) { return {}; }
  }

  function persist(node) {
    const key = node.id || node.dataset.sizeKey;
    if (!key) return;
    try {
      const store = readStore();
      store[key] = node.style.height || `${node.clientHeight}px`;
      localStorage.setItem(STORE_KEY, JSON.stringify(store));
    } catch (_) { /* storage unavailable */ }
  }

  function restoreSizes() {
    const store = readStore();
    document.querySelectorAll('.resizable').forEach((node, index) => {
      if (!node.id) node.dataset.sizeKey = `resizable-${index}`;
      const key = node.id || node.dataset.sizeKey;
      if (store[key]) node.style.height = store[key];
    });
  }

  function resetSizes() {
    try { localStorage.removeItem(STORE_KEY); } catch (_) { /* ignore */ }
    document.querySelectorAll('.resizable').forEach((node) => { node.style.height = ''; });
    setTimeout(resizeAll, 120);
  }

  window.addEventListener('resize', () => {
    clearTimeout(window.__hisimResize);
    window.__hisimResize = setTimeout(resizeAll, 140);
  });

  return {
    draw, purge, empty, layout, config, palette, colorFor, hwColor, chartjs,
    modelColor, modelMarker, seedModels,
    resizeAll, tooltip, cssVar, registerRedraw, observeResizable, restoreSizes, resetSizes,
  };
})();
