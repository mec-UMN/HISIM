/* ============================================================
   Tile labelling helpers: type label, layer list and compact layer summary
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.tilefmt = (() => {
  // ai_layer arrives as an array or as a printed python list: "['L84', 'L85']"
  function layerList(raw) {
    const text = Array.isArray(raw) ? raw.join(' ') : String(raw ?? '');
    return text.match(/L\d+/g) || [];
  }

  // hw_type carries every mapped layer in front of the type ("L123 L120 CPU"); keep only the type.
  function typeLabel(tile) {
    const type = String(tile.hw_type || '').replace(/\bL\d+\b\s*/g, '').trim();
    return type || tile.hw_class || '—';
  }

  // ['L1','L2','L3','L6'] → "L1–L3, L6"
  function layerRanges(layers) {
    const nums = [...new Set(layers.map((l) => Number(String(l).slice(1))))].sort((a, b) => a - b);
    const parts = [];
    for (let i = 0; i < nums.length;) {
      let j = i;
      while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j += 1;
      parts.push(j > i ? `L${nums[i]}–L${nums[j]}` : `L${nums[i]}`);
      i = j + 1;
    }
    return parts.join(', ');
  }

  function layerSummary(layers, maxChars = 26) {
    const full = layerRanges(layers);
    const preview = full.length > maxChars ? `${full.slice(0, maxChars).replace(/[,\s]+$/, '')}…` : full;
    return { count: new Set(layers).size, preview, full };
  }

  return { layerList, typeLabel, layerRanges, layerSummary };
})();
