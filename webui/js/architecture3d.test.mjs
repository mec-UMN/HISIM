import assert from 'node:assert/strict';
import { computeLayout, computeTierFootprint, heatColor, packageLegend, HEAT_STOPS } from './architecture3d-layout.js';

// Two tiles in the same chiplet, adjacent on the NoC x-axis, should sit
// exactly one `step` apart and share the same z/y.
{
  const tiles = [
    { noc: [0, 0], nop: [0, 0, 0] },
    { noc: [1, 0], nop: [0, 0, 0] },
  ];
  const layout = computeLayout(tiles);
  const a = layout.position(tiles[0]);
  const b = layout.position(tiles[1]);
  assert.equal(b.x - a.x, layout.step, 'adjacent NoC tiles should be one step apart on x');
  assert.equal(a.z, b.z, 'tiles at the same noc[1] should share z');
  assert.equal(a.y, b.y, 'tiles in the same tier should share y');
}

// Two chiplets side by side (nop[0] differs) should be exactly `pitch` apart,
// and a chiplet must be wide enough that its tiles never overlap the next
// chiplet's tiles.
{
  const tiles = [
    { noc: [0, 0], nop: [0, 0, 0] },
    { noc: [0, 0], nop: [1, 0, 0] },
  ];
  const layout = computeLayout(tiles);
  const a = layout.position(tiles[0]);
  const b = layout.position(tiles[1]);
  assert.equal(b.x - a.x, layout.pitch, 'adjacent chiplets should be one pitch apart on x');
  assert.ok(layout.pitch > layout.chipletSpan, 'pitch must exceed chiplet span so chiplets do not overlap');
}

// A tile one tier up (nop[2] = 1) should be exactly one tierHeight above tier 0.
{
  const tiles = [
    { noc: [0, 0], nop: [0, 0, 0] },
    { noc: [0, 0], nop: [0, 0, 1] },
  ];
  const layout = computeLayout(tiles, { tierHeight: 2 });
  const a = layout.position(tiles[0]);
  const b = layout.position(tiles[1]);
  assert.equal(b.y - a.y, 2, 'one tier up should be exactly tierHeight higher');
  assert.equal(layout.tierHeight, 2, 'computeLayout should return the tierHeight it used');
}

// heatColor should return the low-stop color at t=0 and the high-stop color at t=1.
{
  const stops = [[0, 0, 0], [1, 1, 1]];
  assert.deepEqual(heatColor(0, stops), [0, 0, 0]);
  assert.deepEqual(heatColor(1, stops), [1, 1, 1]);
  assert.deepEqual(heatColor(0.5, stops), [0.5, 0.5, 0.5]);
}

console.log('architecture3d layout/color tests passed');

// The thermal overlay covers only the tiles of the selected tier.
{
  const tiles = [
    { noc: [0, 0], nop: [0, 0, 0] },
    { noc: [1, 0], nop: [0, 0, 0] },
    { noc: [0, 0], nop: [0, 0, 1] },
  ];
  const layout = computeLayout(tiles);
  const t0 = computeTierFootprint(tiles, layout, 0);
  const t1 = computeTierFootprint(tiles, layout, 1);
  assert.ok(t0.x1 - t0.x0 > t1.x1 - t1.x0, 'tier 0 has two tiles, tier 1 has one');
  assert.equal(t1.y, layout.tierHeight);
  assert.equal(computeTierFootprint(tiles, layout, 7), null);
}

// The Package view needs a key: categories for HW type, a 0..max gradient for numeric colourings.
{
  const tiles = [
    { hw_class: 'SA', latency_s: 0.5, area_mm2: 2 }, { hw_class: 'CPU', latency_s: 2, area_mm2: 1 },
    { hw_class: 'SA', latency_s: 1, area_mm2: 4 },
  ];
  const cat = packageLegend(tiles, 'hw_class');
  assert.equal(cat.kind, 'categorical');
  assert.deepEqual(cat.items.map((i) => i.key), ['CPU', 'SA']);
  const grad = packageLegend(tiles, 'latency_s');
  assert.equal(grad.kind, 'gradient');
  assert.equal(grad.min, 0);
  assert.equal(grad.max, 2);
  assert.equal(grad.metric, 'latency_s');
  assert.equal(packageLegend(tiles, 'area_mm2').max, 4);
  assert.equal(packageLegend([], 'latency_s').max, 0);
  assert.ok(Array.isArray(HEAT_STOPS) && HEAT_STOPS.length >= 3);
}
