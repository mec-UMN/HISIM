/* ============================================================
   Thermal helpers: formatting, Pareto front, heat colours, CSV
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.thermal = (() => {
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const toC = (k) => (isNum(k) ? k - 273.15 : null);

  function formatTemp(k) {
    if (!isNum(k)) return '—';
    return `${k.toFixed(1)} K (${toC(k).toFixed(1)} °C)`;
  }

  const STATUS = { queued: 'queued', running: 'running…', done: 'done', failed: 'failed' };
  const statusLabel = (status) => STATUS[status] || '—';

  const STOPS = [[10, 20, 46], [20, 107, 140], [191, 122, 26], [184, 31, 41], [255, 235, 160]];
  function heatRgb(t) {
    const c = Math.min(1, Math.max(0, t));
    const seg = Math.min(STOPS.length - 2, Math.floor(c * (STOPS.length - 1)));
    const f = c * (STOPS.length - 1) - seg;
    return [0, 1, 2].map((i) => Math.round(STOPS[seg][i] + (STOPS[seg + 1][i] - STOPS[seg][i]) * f));
  }

  function mapRange(grid) {
    let min = Infinity; let max = -Infinity;
    grid.forEach((row) => row.forEach((v) => { if (v < min) min = v; if (v > max) max = v; }));
    return { min, max };
  }

  function gridToCsv(grid, xEdges, yEdges) {
    const mid = (e, i) => (e[i] + e[i + 1]) / 2;
    const header = ['y_mm\\x_mm', ...grid[0].map((_, i) => mid(xEdges, i))].join(',');
    const rows = grid.map((row, j) => [mid(yEdges, j), ...row].join(','));
    return `${header}\n${rows.join('\n')}\n`;
  }

  // Long-format CSV for every die: die,x_mm,y_mm,temperature_k
  function gridsToCsv(dies, xEdges, yEdges) {
    const mid = (e, i) => (e[i] + e[i + 1]) / 2;
    const rows = ['die,x_mm,y_mm,temperature_k'];
    dies.forEach(({ die, grid }) => grid.forEach((row, j) => row.forEach((v, i) => {
      rows.push(`${die},${mid(xEdges, i)},${mid(yEdges, j)},${v}`);
    })));
    return `${rows.join('\n')}\n`;
  }

  // Geometry of one PNG holding every die's map side by side plus a shared colour bar.
  function montageLayout(n, w, h, maxCell = 360) {
    const scale = Math.max(1, Math.floor(maxCell / Math.max(w, h)));
    const titleH = 18; const pad = 10; const barW = 78;
    const cols = Math.max(1, Math.ceil(Math.sqrt(n)));
    const rows = Math.max(1, Math.ceil(n / cols));
    const cellW = w * scale; const cellH = h * scale + titleH;
    const positions = Array.from({ length: n }, (_, i) => ({
      x: pad + (i % cols) * (cellW + pad), y: pad + Math.floor(i / cols) * (cellH + pad),
    }));
    return { scale, titleH, pad, barW, cols, rows, cellW, cellH, positions,
      width: pad + cols * (cellW + pad) + barW, height: pad + rows * (cellH + pad) };
  }

  return { isNum, toC, formatTemp, statusLabel, heatRgb, mapRange, gridToCsv, gridsToCsv, montageLayout };
})();
