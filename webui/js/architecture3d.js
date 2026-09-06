/* ============================================================
   3D package visualizer — photoreal Three.js renderer.
   Loaded as an ES module; publishes window.HISIM3D so the classic
   script webui/js/architecture.js can drive it.
   ============================================================ */

import { computeLayout, heatColor } from './architecture3d-layout.js';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const HEAT_STOPS = [
  [0.04, 0.08, 0.18],
  [0.08, 0.42, 0.55],
  [0.75, 0.48, 0.10],
  [0.72, 0.12, 0.16],
];

let renderer = null;
let scene = null;
let camera = null;
let controls = null;
let composer = null;
let bloom = null;
let containerEl = null;
let raf = null;

let instMesh = null;
let orderedTiles = [];
let disposables = [];   // geometries/materials/textures to dispose on rebuild
let sceneObjects = [];  // meshes/lines added to `scene` this render, minus lights

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let tooltip = null;

function charts() { return window.HISIM.charts; }
function fmt() { return window.HISIM.fmt; }

function circuitTexture() {
  const size = 256;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#7a828f';
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = 'rgba(0,0,0,0.22)';
  ctx.lineWidth = 1.4;
  for (let i = 0; i <= size; i += 12) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(size, i); ctx.stroke();
  }
  const grad = ctx.createRadialGradient(size * 0.35, size * 0.3, size * 0.05, size * 0.5, size * 0.5, size * 0.72);
  grad.addColorStop(0, 'rgba(255,255,255,0.16)');
  grad.addColorStop(1, 'rgba(0,0,0,0.22)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function substrateTexture() {
  const size = 256;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#0c0e13';
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = 'rgba(120,160,220,0.10)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= size; i += 16) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(size, i); ctx.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function ensureInit(container) {
  if (renderer && containerEl === container) return;
  containerEl = container;

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;
  renderer.domElement.style.position = 'absolute';
  renderer.domElement.style.inset = '0';
  renderer.domElement.style.display = 'block';
  container.style.position = 'relative';
  container.style.overflow = 'hidden';
  container.innerHTML = '';
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(35, 1, 0.1, 200);

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.48;

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.22, 0.7, 0.94);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  tooltip = charts().tooltip();

  renderer.domElement.addEventListener('mousemove', onPointerMove);
  renderer.domElement.addEventListener('mouseleave', () => tooltip.hide());

  resize();
  if (!raf) {
    const loop = () => {
      controls.update();
      composer.render();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
  }
}

function disposeSceneContents() {
  sceneObjects.forEach((obj) => scene.remove(obj));
  sceneObjects = [];
  disposables.forEach((d) => d.dispose && d.dispose());
  disposables = [];
  instMesh = null;
  scene.children
    .filter((c) => c.isLight)
    .forEach((light) => scene.remove(light));
}

function onPointerMove(event) {
  const rect = containerEl.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  if (!instMesh) return;
  const hit = raycaster.intersectObject(instMesh)[0];
  if (hit && hit.instanceId != null) {
    const t = orderedTiles[hit.instanceId];
    const layer = Array.isArray(t.ai_layer) ? t.ai_layer.join(', ') : t.ai_layer;
    tooltip.show(
      `<b>${t.hw_type}</b><br>${t.stack} · tier ${t.nop[2] || 0}`
      + `<br>layer ${layer}`
      + `<br>area ${fmt().num(t.area_mm2, 4)} mm²`
      + `<br>latency ${fmt().seconds(t.latency_s)}`
      + `<br>energy ${fmt().joules(t.energy_j)}`,
      event,
    );
  } else {
    tooltip.hide();
  }
}

function colorForTile(tile, metric, maxValue) {
  if (metric === 'hw_class') {
    return new THREE.Color(charts().hwColor(tile.hw_class || tile.hw_type));
  }
  const value = Number(tile[metric]) || 0;
  const [r, g, b] = heatColor(maxValue ? value / maxValue : 0, HEAT_STOPS);
  return new THREE.Color(r, g, b);
}

function buildSubstrates(tiles, layout) {
  const keys = [...new Set(tiles.map((t) => `${t.nop[0]},${t.nop[1]},${t.nop[2] || 0}`))];
  const tex = substrateTexture();
  disposables.push(tex);
  const geo = new THREE.BoxGeometry(layout.chipletSpan + layout.tileSize * 0.3, 0.14, layout.chipletSpan + layout.tileSize * 0.3);
  const mat = new THREE.MeshStandardMaterial({ color: 0x14171d, metalness: 0.3, roughness: 0.82, map: tex, envMapIntensity: 0.3 });
  disposables.push(geo, mat);
  keys.forEach((key) => {
    const [nx, ny, nz] = key.split(',').map(Number);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(
      nx * layout.pitch + layout.chipletSpan / 2 - layout.step / 2,
      nz * layout.tierHeight - 0.08,
      ny * layout.pitch + layout.chipletSpan / 2 - layout.step / 2,
    );
    mesh.receiveShadow = true;
    scene.add(mesh);
    sceneObjects.push(mesh);
  });
}

function buildTiles(tiles, layout, metric) {
  const values = tiles.map((t) => Number(t[metric]) || 0);
  const maxValue = Math.max(...values, 1e-18);
  const tex = circuitTexture();
  const geo = new THREE.BoxGeometry(layout.tileSize, 0.16, layout.tileSize);
  const mat = new THREE.MeshStandardMaterial({ map: tex, metalness: 0.6, roughness: 0.36, envMapIntensity: 0.9 });
  disposables.push(tex, geo, mat);

  instMesh = new THREE.InstancedMesh(geo, mat, tiles.length);
  instMesh.castShadow = true;
  instMesh.receiveShadow = true;
  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  orderedTiles = tiles;
  tiles.forEach((tile, i) => {
    const p = layout.position(tile);
    dummy.position.set(p.x, p.y + 0.09, p.z);
    dummy.updateMatrix();
    instMesh.setMatrixAt(i, dummy.matrix);
    color.copy(colorForTile(tile, metric, maxValue));
    instMesh.setColorAt(i, color);
  });
  instMesh.instanceMatrix.needsUpdate = true;
  if (instMesh.instanceColor) instMesh.instanceColor.needsUpdate = true;
  scene.add(instMesh);
  sceneObjects.push(instMesh);
}

function buildLinks(tiles, layout) {
  const lookup = new Map(tiles.map((t) => [`${t.chiplet}|${t.noc[0]},${t.noc[1]}`, t]));
  const points = [];
  tiles.forEach((t) => {
    [[1, 0], [0, 1]].forEach(([dx, dy]) => {
      const n = lookup.get(`${t.chiplet}|${t.noc[0] + dx},${t.noc[1] + dy}`);
      if (!n) return;
      const p1 = layout.position(t);
      const p2 = layout.position(n);
      points.push(new THREE.Vector3(p1.x, p1.y + 0.18, p1.z), new THREE.Vector3(p2.x, p2.y + 0.18, p2.z));
    });
  });
  const tierLookup = new Map(tiles.map((t) => [`${t.stack}|${t.noc[0]},${t.noc[1]}|${t.nop[2] || 0}`, t]));
  tiles.forEach((t) => {
    const above = tierLookup.get(`${t.stack}|${t.noc[0]},${t.noc[1]}|${(t.nop[2] || 0) + 1}`);
    if (!above) return;
    const p1 = layout.position(t);
    const p2 = layout.position(above);
    points.push(new THREE.Vector3(p1.x, p1.y + 0.18, p1.z), new THREE.Vector3(p2.x, p2.y + 0.18, p2.z));
  });
  if (!points.length) return;
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineBasicMaterial({ color: charts().cssVar('--line', '#5c6e88'), transparent: true, opacity: 0.35 });
  disposables.push(geo, mat);
  const lines = new THREE.LineSegments(geo, mat);
  lines.name = 'links';
  scene.add(lines);
  sceneObjects.push(lines);
}

function buildLights(extent) {
  const ambient = new THREE.AmbientLight(0x2b3444, 0.38);
  const key = new THREE.DirectionalLight(0xdfeaff, 2.6);
  key.position.set(-extent * 0.55, extent * 0.65, extent * 0.9);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.near = 0.1; key.shadow.camera.far = extent * 6;
  key.shadow.camera.left = -extent * 1.3; key.shadow.camera.right = extent * 1.3;
  key.shadow.camera.top = extent * 1.3; key.shadow.camera.bottom = -extent * 1.3;
  key.shadow.bias = -0.0015;
  const rim = new THREE.DirectionalLight(0x4ea8f5, 0.7);
  rim.position.set(extent * 0.9, extent * 0.35, -extent * 0.7);
  scene.add(ambient, key, rim);
}

// Bounding box (x/z) of every tile position, so framing scales with the
// actual package footprint instead of assuming a fixed grid size.
function tileBounds(tiles, layout) {
  let minX = Infinity; let maxX = -Infinity;
  let minZ = Infinity; let maxZ = -Infinity;
  tiles.forEach((t) => {
    const p = layout.position(t);
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z);
  });
  return {
    centerX: (minX + maxX) / 2,
    centerZ: (minZ + maxZ) / 2,
    span: Math.max(maxX - minX, maxZ - minZ, layout.chipletSpan),
  };
}

let currentPayload = null;
let currentOptions = { metric: 'hw_class', showLinks: true };

function render(container, payload, options = {}) {
  currentPayload = payload;
  currentOptions = { ...currentOptions, ...options };
  ensureInit(container);

  const tiles = (payload?.topology?.tiles || []).filter((tile) => (
    tile?.hw_class !== 'Empty' && String(tile?.hw_type || '').trim().toUpperCase() !== 'EMPTY'
  ));
  disposeSceneContents();
  scene.background = new THREE.Color(charts().cssVar('--bg', '#0a0e17'));

  if (!tiles.length) { buildLights(10); return; }

  const layout = computeLayout(tiles);
  const bounds = tileBounds(tiles, layout);
  buildLights(bounds.span);
  buildSubstrates(tiles, layout);
  buildTiles(tiles, layout, currentOptions.metric);
  if (currentOptions.showLinks) buildLinks(tiles, layout);

  // Low, raking "product shot" angle: distance and height both scale with
  // the package footprint so a 3-tile demo and a 1000-tile package both
  // frame the whole thing without the tiles reading as a flat gradient.
  camera.position.set(
    bounds.centerX - bounds.span * 0.55,
    bounds.span * 0.32,
    bounds.centerZ + bounds.span * 0.85,
  );
  controls.target.set(bounds.centerX, layout.tileSize * 0.5, bounds.centerZ);
  controls.update();
}

function setMetric(metric) {
  if (currentPayload) render(containerEl, currentPayload, { ...currentOptions, metric });
}

function setLinksVisible(showLinks) {
  if (currentPayload) render(containerEl, currentPayload, { ...currentOptions, showLinks });
}

function resetCamera() {
  if (!currentPayload) return;
  render(containerEl, currentPayload, currentOptions);
}

function resize() {
  if (!renderer || !containerEl) return;
  const w = Math.max(1, containerEl.clientWidth);
  const h = Math.max(1, containerEl.clientHeight);
  renderer.setSize(w, h);
  composer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

window.HISIM3D = Object.assign(window.HISIM3D || {}, {
  render, setMetric, setLinksVisible, resetCamera, resize,
});
