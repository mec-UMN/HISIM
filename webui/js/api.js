/* ============================================================
   API client for the HISIM GUI backend
   ============================================================ */
window.HISIM = window.HISIM || {};

HISIM.api = (() => {
  const base = `${location.origin}/api`;

  async function request(path, options = {}) {
    const response = await fetch(`${base}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        if (body && body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
      } catch (_) { /* non-JSON error body */ }
      throw new Error(detail);
    }
    const type = response.headers.get('content-type') || '';
    return type.includes('application/json') ? response.json() : response.text();
  }

  return {
    base,
    health:        () => request('/health'),
    models:        () => request('/models'),
    knobs:         () => request('/knobs'),
    defaults:      () => request('/config/defaults'),

    startRun:      (payload) => request('/runs',   { method: 'POST', body: JSON.stringify(payload) }),
    startSweep:    (payload) => request('/sweeps', { method: 'POST', body: JSON.stringify(payload) }),
    runs:          () => request('/runs'),
    run:           (id) => request(`/runs/${id}`),
    deleteRun:     (id) => request(`/runs/${id}`, { method: 'DELETE' }),
    result:        (id) => request(`/runs/${id}/result`),
    logs:          (id, stream = 'stdout', tail = 4000) => request(`/runs/${id}/logs?stream=${stream}&tail=${tail}`),
    // artifacts are still archived per run on disk and exposed by the API
    // (/api/runs/{id}/artifacts) — the UI just does not preview them.

    plots:         () => request('/plots'),
    plot:          (id) => request(`/plots/${id}`),

    mappingFiles:  (aimodel) => request(`/mapping-files/${encodeURIComponent(aimodel)}`),
    mappingFilesOverview: (aimodel) => request(`/mapping-files/${encodeURIComponent(aimodel)}/overview`),
    mappingFileDownloadUrl: (aimodel, kind) => `${base}/mapping-files/${encodeURIComponent(aimodel)}/${kind}`,
    // FormData uploads must not set Content-Type themselves (the browser adds
    // the multipart boundary), so this bypasses request()'s JSON default.
    uploadMappingFile: async (aimodel, kind, file) => {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch(`${base}/mapping-files/${encodeURIComponent(aimodel)}/${kind}`, {
        method: 'POST',
        body: form,
      });
      if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
          const body = await response.json();
          if (body && body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
        } catch (_) { /* non-JSON error body */ }
        throw new Error(detail);
      }
      return response.json();
    },
  };
})();
