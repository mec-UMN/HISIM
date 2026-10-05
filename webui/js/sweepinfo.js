/* ============================================================
   Plain-language warnings about sweeps that cannot change the result
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.sweepinfo = (() => {
  // With "Size banks for full on-package memory" on, HISIM derives each memory tile's bank count from the
  // AI model and ignores the Banks knob, so every point of an n_bank sweep comes out identical.
  function sweepWarning(knob, config = {}) {
    if (knob === 'n_bank' && config.set_suff_banks) {
      return 'Bank count is auto-sized while “Size banks for full on-package memory” is on, so every point of this sweep will give identical results. Turn that option off to sweep banks.';
    }
    return '';
  }
  return { sweepWarning };
})();
