// Reverse proxy: forwards every request received at sim.ganaptewary.com
// to the HISIM Python backend, reached via a public URL (a Cloudflare
// Tunnel from the Hostinger SSH box, since this container can't reach
// that box's localhost directly). Set BACKEND_URL in this app's
// environment variables to update the target without a redeploy.
const http = require('http');
const https = require('https');
const { URL } = require('url');

const PORT = process.env.PORT || 3002;
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8001';

const server = http.createServer((req, res) => {
  const target = new URL(req.url, BACKEND_URL);
  const client = target.protocol === 'https:' ? https : http;

  const forwardedHeaders = { ...req.headers, host: target.host };

  const proxyReq = client.request(
    {
      hostname: target.hostname,
      port: target.port || (target.protocol === 'https:' ? 443 : 80),
      path: target.pathname + target.search,
      method: req.method,
      headers: forwardedHeaders,
    },
    (proxyRes) => {
      res.writeHead(proxyRes.statusCode, proxyRes.headers);
      proxyRes.pipe(res, { end: true });
    },
  );

  proxyReq.on('error', (err) => {
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Backend unreachable', detail: err.message }));
  });

  req.pipe(proxyReq, { end: true });
});

server.listen(PORT, () => {
  console.log(`Simulator proxy listening on ${PORT}, forwarding to ${BACKEND_URL}`);
});
