/* ============================================================
   Pure package-layout math for the 3D visualizer — no THREE.js or
   DOM dependency, so it can be exercised directly with plain Node.
   ============================================================ */

export function computeLayout(tiles, opts = {}) {
  const tileSize = opts.tileSize ?? 0.82;
  const gap = opts.gap ?? 0.18;
  const chipletGapFactor = opts.chipletGapFactor ?? 2.4;
  const tierHeight = opts.tierHeight ?? 1.4;
  const step = tileSize + gap;

  const maxNoc = tiles.length
    ? Math.max(...tiles.map((t) => Math.max(t.noc[0], t.noc[1]))) + 1
    : 1;
  const chipletSpan = maxNoc * step - gap;
  const pitch = chipletSpan + tileSize * chipletGapFactor;

  const position = (tile) => ({
    x: tile.noc[0] * step + tile.nop[0] * pitch,
    y: (tile.nop[2] || 0) * tierHeight,
    z: tile.noc[1] * step + tile.nop[1] * pitch,
  });

  return { tileSize, gap, step, chipletSpan, pitch, tierHeight, position };
}

export function heatColor(t, stops) {
  const clamped = Math.min(1, Math.max(0, t));
  const seg = Math.min(stops.length - 2, Math.floor(clamped * (stops.length - 1)));
  const localT = clamped * (stops.length - 1) - seg;
  const a = stops[seg];
  const b = stops[seg + 1];
  return [0, 1, 2].map((i) => a[i] + (b[i] - a[i]) * localT);
}
