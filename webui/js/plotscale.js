/* ============================================================
   Axis calibration for the comparison charts: choose linear vs log, pad
   degenerate ranges, and pick tick formats from the unit in the label.
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.plotscale = (() => {
  const LOG_SPREAD = 20;   // use a log axis only when max/min spans more than this

  function unitFormat(label) {
    const unit = (/\(([^)]*)\)\s*$/.exec(String(label)) || [])[1] || '';
    if (unit === '$') return { tickprefix: '$', tickformat: '.4~g' };
    if (['s', 'J', 'W'].includes(unit)) return { ticksuffix: unit, tickformat: '.3~s' };
    if (unit) return { ticksuffix: ` ${unit}`, tickformat: '.4~g' };
    return { tickformat: '.4~g' };
  }

  // values: numbers plotted on the axis; label: axis title such as "Latency (s)".
  function axisFor(values, label = '') {
    const finite = (values || []).filter((v) => Number.isFinite(v));
    const format = unitFormat(label);
    if (!finite.length) return { type: 'linear', constant: false, ...format };
    const min = Math.min(...finite);
    const max = Math.max(...finite);

    if (max === min) {
      const pad = Math.abs(min) * 0.1 || 1;
      return { type: 'linear', constant: true, range: [min - pad, max + pad], tickvals: [min], ...format };
    }
    if (min > 0 && max / min > LOG_SPREAD) return { type: 'log', constant: false, ...format };
    const pad = (max - min) * 0.1;
    return { type: 'linear', constant: false, range: [min - pad, max + pad], ...format };
  }

  return { axisFor };
})();
