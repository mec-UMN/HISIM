window.HISIM = window.HISIM || {};

HISIM.csv = (() => {
  function escapeCell(value) {
    const s = value === null || value === undefined ? '' : String(value);
    if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
  }

  // columns: [[key, label], ...]  rows: array of plain objects
  function tilesToCSV(columns, rows) {
    const lines = [columns.map(([, label]) => escapeCell(label)).join(',')];
    rows.forEach((row) => {
      lines.push(columns.map(([key]) => {
        const value = row[key];
        return escapeCell(Array.isArray(value) ? value.join(' ') : value);
      }).join(','));
    });
    return lines.join('\r\n');
  }

  function downloadCSV(filename, csvText) {
    const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return { tilesToCSV, downloadCSV };
})();
