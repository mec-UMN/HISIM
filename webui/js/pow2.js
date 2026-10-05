/* ============================================================
   Power-of-two knobs (SA rows / columns, arrays per tile): the slider
   moves in exponent steps and typed values snap to the nearest power.
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.pow2 = (() => {
  const toExponent = (v) => Math.round(Math.log2(Math.max(1, Number(v) || 1)));
  const fromExponent = (e) => 2 ** e;
  const exponentRange = (min, max) => ({ min: toExponent(min), max: Math.floor(Math.log2(max)) });
  const snap = (v, min, max) => {
    const { min: lo, max: hi } = exponentRange(min, max);
    return fromExponent(Math.min(hi, Math.max(lo, toExponent(v))));
  };
  const values = (min, max) => {
    const { min: lo, max: hi } = exponentRange(min, max);
    return Array.from({ length: hi - lo + 1 }, (_, i) => fromExponent(lo + i));
  };
  return { snap, toExponent, fromExponent, exponentRange, values };
})();
