/* ============================================================
   Number / unit formatting + small DOM helpers
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.fmt = (() => {
  const SI = [
    { limit: 1e-12, suffix: 'p', factor: 1e12 },
    { limit: 1e-9,  suffix: 'n', factor: 1e9  },
    { limit: 1e-6,  suffix: 'µ', factor: 1e6  },
    { limit: 1e-3,  suffix: 'm', factor: 1e3  },
    { limit: 1,     suffix: '',  factor: 1    },
    { limit: 1e3,   suffix: 'k', factor: 1e-3 },
    { limit: 1e6,   suffix: 'M', factor: 1e-6 },
    { limit: 1e9,   suffix: 'G', factor: 1e-9 },
  ];

  function si(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    const abs = Math.abs(value);
    if (abs === 0) return '0';
    let chosen = SI[SI.length - 1];
    for (const entry of SI) {
      if (abs < entry.limit * 1000) { chosen = entry; break; }
    }
    const scaled = value * chosen.factor;
    const precision = Math.abs(scaled) >= 100 ? 0 : Math.abs(scaled) >= 10 ? 1 : digits - 1;
    return `${scaled.toFixed(precision)}${chosen.suffix}`;
  }

  function scientific(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    const numeric = Number(value);
    if (numeric === 0) return '0';
    let exponent = Math.floor(Math.log10(Math.abs(numeric)));
    let mantissa = numeric / (10 ** exponent);
    if (Math.abs(mantissa) >= 10) {
      mantissa /= 10;
      exponent += 1;
    }
    return `${mantissa.toFixed(digits)} × 10^${exponent}`;
  }

  const num = (value, digits = 3) => {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-4 || abs >= 1e6)) return scientific(value, 2);
    return value.toLocaleString(undefined, { maximumFractionDigits: digits });
  };

  const seconds  = (v) => (v === null || v === undefined ? '—' : `${si(v)}s`);
  const joules   = (v) => (v === null || v === undefined ? '—' : `${si(v)}J`);
  const watts    = (v) => (v === null || v === undefined ? '—' : `${si(v)}W`);
  const area     = (v) => (v === null || v === undefined ? '—' : `${num(v, 2)} mm²`);
  const usd      = (v) => (v === null || v === undefined ? '—' : `$${num(v, 2)}`);
  const hz       = (v) => (v === null || v === undefined ? '—' : `${si(v)}Hz`);
  const pct      = (v) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`);

  const ago = (iso) => {
    const then = new Date(iso).getTime();
    const delta = Math.max(0, (Date.now() - then) / 1000);
    if (delta < 60) return `${Math.round(delta)}s ago`;
    if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
    if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
    return new Date(iso).toLocaleDateString();
  };

  const bytes = (n) => {
    if (!n) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
    return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
  };

  const title = (s) => String(s || '').replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  return { si, scientific, num, seconds, joules, watts, area, usd, hz, pct, ago, bytes, title };
})();

HISIM.dom = (() => {
  const el = (id) => document.getElementById(id);
  const make = (tag, className, html) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (html !== undefined) node.innerHTML = html;
    return node;
  };
  const clear = (node) => { while (node && node.firstChild) node.removeChild(node.firstChild); };
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));

  let toastId = 0;
  function toast(title, message, kind = '') {
    const host = el('toasts');
    if (!host) return;
    const node = make('div', `toast ${kind}`, `<b>${esc(title)}</b><span>${esc(message || '')}</span>`);
    node.dataset.id = ++toastId;
    host.appendChild(node);
    setTimeout(() => {
      node.style.transition = 'opacity .3s, transform .3s';
      node.style.opacity = '0';
      node.style.transform = 'translateX(18px)';
      setTimeout(() => node.remove(), 320);
    }, kind === 'error' ? 8000 : 4200);
  }

  return { el, make, clear, esc, toast };
})();
