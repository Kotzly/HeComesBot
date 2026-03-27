'use strict';

// ── State ────────────────────────────────────────────────────────────────────
let treeId        = null;
let treeData      = null;
let selectedId    = null;
let collapsedIds  = new Set();
let functionsData = null;

// ── D3 setup ─────────────────────────────────────────────────────────────────
const svg  = d3.select('#tree-svg');
const gRoot = svg.append('g').attr('class', 'root-g');
const zoom = d3.zoom().scaleExtent([0.05, 4]).on('zoom', e => gRoot.attr('transform', e.transform));
svg.call(zoom);

const arityColor = a => a === 0 ? '#4ec9b0' : a === 1 ? '#569cd6' : '#ce9178';

// ── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  const [funcsRes, persRes] = await Promise.all([
    fetch('/api/functions'),
    fetch('/api/personalities'),
  ]);
  functionsData = await funcsRes.json();
  const personalities = await persRes.json();

  const pSel = document.getElementById('personality-select');
  personalities.forEach(p => pSel.appendChild(new Option(p, p)));

  const fSel = document.getElementById('func-select');
  const arityLabels = {'0':'Leaf (arity 0)','1':'Unary (arity 1)','2':'Binary (arity 2)'};
  Object.entries(functionsData).sort().forEach(([arity, funcs]) => {
    const group = document.createElement('optgroup');
    group.label = arityLabels[arity] || `Arity ${arity}`;
    funcs.forEach(f => group.appendChild(new Option(f, f)));
    fSel.appendChild(group);
  });

  document.getElementById('regen-seed-input').value = randInt();

  // Build / save / load
  document.getElementById('build-btn')        .addEventListener('click', onBuild);
  document.getElementById('save-btn')         .addEventListener('click', onSave);
  document.getElementById('load-btn')         .addEventListener('click', onLoad);
  document.getElementById('refresh-trees-btn').addEventListener('click', refreshTreeList);

  // Preview
  document.getElementById('preview-btn')      .addEventListener('click', onPreview);
  document.getElementById('preview-img')      .addEventListener('click', openModal);
  document.getElementById('modal-close-btn')  .addEventListener('click', closeModal);
  document.getElementById('preview-modal')    .addEventListener('click', e => { if (e.target.id === 'preview-modal') closeModal(); });

  // Node editor
  document.getElementById('apply-func-btn')   .addEventListener('click', onApplyFunc);
  document.getElementById('regen-btn')        .addEventListener('click', onRegen);
  document.getElementById('update-leaf-btn')  .addEventListener('click', onUpdateLeaf);

  // Seed random buttons
  document.getElementById('random-seed-btn')  .addEventListener('click', () => { document.getElementById('seed-input').value = randInt(); });
  document.getElementById('regen-random-btn') .addEventListener('click', () => { document.getElementById('regen-seed-input').value = randInt(); });

  // Delta slider ↔ number sync
  const slider = document.getElementById('delta-slider');
  const num    = document.getElementById('delta-num');
  slider.addEventListener('input', () => { num.value = parseFloat(slider.value).toFixed(4); });
  num.addEventListener('input', () => { slider.value = num.value; });

  window.addEventListener('resize', () => { if (treeData) renderTree(); });
  await refreshTreeList();
}

function randInt() { return Math.floor(Math.random() * 1e9); }

// ── Save / Load ───────────────────────────────────────────────────────────────
async function refreshTreeList() {
  const res   = await fetch('/api/trees');
  const trees = await res.json();
  const sel   = document.getElementById('load-select');
  sel.innerHTML = '';
  if (!trees.length) sel.appendChild(new Option('(none saved)', ''));
  else trees.forEach(t => sel.appendChild(new Option(t, t)));
}

async function onSave() {
  const name = document.getElementById('save-name-input').value.trim();
  if (!name) { alert('Enter a name first.'); return; }
  const res  = await fetch('/api/save', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({tree_id: treeId, name}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  await refreshTreeList();
  document.getElementById('load-select').value = data.name;
}

async function onLoad() {
  const name = document.getElementById('load-select').value;
  if (!name) return;
  const res  = await fetch('/api/load', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({name}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  applyLoadedTree(data);
}

function applyLoadedTree(data) {
  treeId    = data.tree_id;
  treeData  = data.tree;
  selectedId = null;
  collapsedIds.clear();
  const meta = data.meta || {};
  if (meta.seed != null)      document.getElementById('seed-input').value      = meta.seed;
  if (meta.dx   != null)      document.getElementById('width-input').value     = meta.dx;
  if (meta.dy   != null)      document.getElementById('height-input').value    = meta.dy;
  if (meta.min_depth != null) document.getElementById('min-depth-input').value = meta.min_depth;
  if (meta.max_depth != null) document.getElementById('max-depth-input').value = meta.max_depth;

  document.getElementById('tree-hint').style.display = 'none';
  document.getElementById('preview-btn').disabled = false;
  document.getElementById('save-btn').disabled    = false;
  document.getElementById('node-panel').classList.add('hidden');

  renderTree();
  fitTree();
  updateStats(treeData);
}

// ── Build ─────────────────────────────────────────────────────────────────────
async function onBuild() {
  const btn = document.getElementById('build-btn');
  btn.disabled = true; btn.textContent = 'Building…';
  try {
    const res  = await fetch('/api/build', {
      method: 'POST', headers: jsonHdr(),
      body: JSON.stringify({
        seed:        parseInt(document.getElementById('seed-input').value),
        width:       parseInt(document.getElementById('width-input').value),
        height:      parseInt(document.getElementById('height-input').value),
        min_depth:   parseInt(document.getElementById('min-depth-input').value),
        max_depth:   parseInt(document.getElementById('max-depth-input').value),
        personality: document.getElementById('personality-select').value,
      }),
    });
    applyLoadedTree(await res.json());
  } finally {
    btn.disabled = false; btn.textContent = 'Build Tree';
  }
}

// ── Preview ───────────────────────────────────────────────────────────────────
async function onPreview() {
  const btn = document.getElementById('preview-btn');
  btn.disabled = true; btn.textContent = 'Rendering…';
  try {
    const res  = await fetch('/api/preview', {
      method: 'POST', headers: jsonHdr(), body: JSON.stringify({tree_id: treeId}),
    });
    const data = await res.json();
    const img  = document.getElementById('preview-img');
    img.src = data.image;
    document.getElementById('preview-wrap').style.display = '';
    document.getElementById('modal-img').src = data.image;
  } finally {
    btn.disabled = false; btn.textContent = 'Render Frame';
  }
}

function openModal()  { document.getElementById('preview-modal').classList.remove('hidden'); }
function closeModal() { document.getElementById('preview-modal').classList.add('hidden'); }

// ── Node editing ──────────────────────────────────────────────────────────────
async function onApplyFunc() {
  if (!selectedId) return;
  const funcName = document.getElementById('func-select').value;
  const res  = await fetch('/api/node/set-func', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({tree_id: treeId, node_id: selectedId, func_name: funcName}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  treeData = data.tree;
  const node = findNode(treeData, selectedId);
  if (node) selectNode(node); else renderTree();
  updateStats(treeData);
}

async function onRegen() {
  if (!selectedId) return;
  const seed = parseInt(document.getElementById('regen-seed-input').value);
  const res  = await fetch('/api/node/regenerate', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({tree_id: treeId, node_id: selectedId, seed}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  treeData   = data.tree;
  selectedId = data.new_node_id;
  collapsedIds.clear();
  const node = findNode(treeData, selectedId);
  if (node) selectNode(node); else renderTree();
  updateStats(treeData);
}

async function onUpdateLeaf() {
  if (!selectedId) return;
  const node = findNode(treeData, selectedId);
  if (!node || node.arity !== 0) return;

  const delta  = parseFloat(document.getElementById('delta-num').value);
  const params = collectLeafParams(node.func);

  const res  = await fetch('/api/leaf/set-params', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({tree_id: treeId, node_id: selectedId, params, delta}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  treeData = data.tree;
  const updated = findNode(treeData, selectedId);
  if (updated) selectNode(updated);
}

function collectLeafParams(func) {
  const params = {};
  if (func === 'rand_color') {
    const hex = document.getElementById('lp-color').value;
    params.color = hexToRgb(hex);
  } else if (func === 'cone') {
    params.cx = parseFloat(document.getElementById('lp-cx').value);
    params.cy = parseFloat(document.getElementById('lp-cy').value);
    params.rx = parseFloat(document.getElementById('lp-rx').value);
    params.ry = parseFloat(document.getElementById('lp-ry').value);
  } else if (func === 'circle') {
    params.cx = parseFloat(document.getElementById('lp-cx').value);
    params.cy = parseFloat(document.getElementById('lp-cy').value);
    params.rx = parseFloat(document.getElementById('lp-rx').value);
    params.ry = parseFloat(document.getElementById('lp-ry').value);
    params.color = hexToRgb(document.getElementById('lp-color').value);
  }
  return params;
}

// ── Tree rendering ────────────────────────────────────────────────────────────
function renderTree() {
  const container = document.getElementById('tree-container');
  const W = container.clientWidth;
  const H = container.clientHeight;
  svg.attr('width', W).attr('height', H);
  gRoot.selectAll('*').remove();

  const visible = visibleSubtree(treeData);
  const hier    = d3.hierarchy(visible, d => d._kids.length ? d._kids : null);

  const leafCount   = hier.leaves().length;
  const nodeSpacing = Math.max(16, Math.min(32, 800 / (leafCount + 1)));
  d3.tree().nodeSize([nodeSpacing, 180])(hier);

  const nodes = hier.descendants();
  const minX  = d3.min(nodes, d => d.x);
  const offY  = H / 2 - minX;
  const offX  = 80;

  gRoot.selectAll('.link')
    .data(hier.links())
    .join('path').attr('class', 'link')
    .attr('d', d3.linkHorizontal().x(d => d.y + offX).y(d => d.x + offY));

  const nodeG = gRoot.selectAll('.node')
    .data(nodes).join('g').attr('class', 'node')
    .attr('transform', d => `translate(${d.y + offX},${d.x + offY})`);

  nodeG.append('circle')
    .attr('r', 11)
    .attr('fill', d => arityColor(d.data.arity))
    .attr('stroke', d => d.data.id === selectedId ? '#fff' : '#333')
    .attr('stroke-width', d => d.data.id === selectedId ? 2.5 : 1)
    .on('click',  (e, d) => { e.stopPropagation(); selectNode(d.data); })
    .on('dblclick', (e, d) => { e.stopPropagation(); toggleCollapse(d.data); });

  nodeG.filter(d => collapsedIds.has(d.data.id) && d.data.arity > 0)
    .append('text')
    .attr('dy', '0.35em').attr('text-anchor', 'middle')
    .attr('font-size', '9px').attr('fill', '#fff').attr('pointer-events', 'none')
    .text('…');

  const hasKids = d => d.data._kids.length > 0 && !collapsedIds.has(d.data.id);
  nodeG.append('text').attr('class', 'node-label')
    .attr('x', d => hasKids(d) ? -16 : 16)
    .attr('dy', '0.35em')
    .attr('text-anchor', d => hasKids(d) ? 'end' : 'start')
    .text(d => d.data.func);
}

function visibleSubtree(node) {
  const copy = {id: node.id, func: node.func, arity: node.arity, _kids: []};
  if (node.children.length && !collapsedIds.has(node.id))
    copy._kids = node.children.map(visibleSubtree);
  return copy;
}

function toggleCollapse(nodeData) {
  if (nodeData.arity === 0) return;
  collapsedIds.has(nodeData.id) ? collapsedIds.delete(nodeData.id) : collapsedIds.add(nodeData.id);
  renderTree();
}

function selectNode(nodeData) {
  selectedId = nodeData.id;

  document.getElementById('node-panel').classList.remove('hidden');
  document.getElementById('node-id-display')   .textContent = nodeData.id;
  document.getElementById('node-func-display') .textContent = nodeData.func;
  document.getElementById('node-arity-display').textContent =
    nodeData.arity === 0 ? 'leaf' : nodeData.arity === 1 ? '1 (unary)' : '2 (binary)';
  document.getElementById('func-select').value = nodeData.func;

  const leafEditor = document.getElementById('leaf-editor');
  if (nodeData.arity === 0) {
    leafEditor.classList.remove('hidden');
    buildLeafEditor(nodeData);
  } else {
    leafEditor.classList.add('hidden');
  }

  renderTree();
}

function buildLeafEditor(node) {
  const controls = document.getElementById('leaf-controls');
  controls.innerHTML = '';

  const delta = node.delta ?? 0;
  document.getElementById('delta-slider').value = delta;
  document.getElementById('delta-num').value    = delta.toFixed(4);

  const p = node.params || {};

  if (node.func === 'rand_color') {
    controls.appendChild(makeColorRow('Color', 'lp-color', p.color ?? [0.5, 0.5, 0.5]));

  } else if (node.func === 'circle') {
    controls.appendChild(makeSliderRow('Center X', 'lp-cx', -2, 2, 0.01, p.cx ?? 0));
    controls.appendChild(makeSliderRow('Center Y', 'lp-cy', -2, 2, 0.01, p.cy ?? 0));
    controls.appendChild(makeSliderRow('Radius X', 'lp-rx', 0.01, 1, 0.01, p.rx ?? 0.5));
    controls.appendChild(makeSliderRow('Radius Y', 'lp-ry', 0.01, 1, 0.01, p.ry ?? 0.5));
    controls.appendChild(makeColorRow('Color', 'lp-color', p.color ?? [0.5, 0.5, 0.5]));

  } else if (node.func === 'cone') {
    controls.appendChild(makeSliderRow('Center X', 'lp-cx', -2, 2, 0.01, p.cx ?? 0));
    controls.appendChild(makeSliderRow('Center Y', 'lp-cy', -2, 2, 0.01, p.cy ?? 0));
    controls.appendChild(makeSliderRow('Radius X', 'lp-rx', 0.01, 1, 0.01, p.rx ?? 0.5));
    controls.appendChild(makeSliderRow('Radius Y', 'lp-ry', 0.01, 1, 0.01, p.ry ?? 0.5));

  } else {
    const msg = document.createElement('div');
    msg.className = 'muted';
    msg.textContent = node.func === 'x_var' || node.func === 'y_var'
      ? 'Spatial grid — values fixed by dimensions.'
      : 'No editable params for this leaf.';
    controls.appendChild(msg);
  }
}

function makeSliderRow(label, id, min, max, step, value) {
  const wrap = document.createElement('label');
  wrap.textContent = label;
  const row = document.createElement('div');
  row.className = 'input-row';

  const slider = document.createElement('input');
  slider.type = 'range'; slider.id = id;
  slider.min = min; slider.max = max; slider.step = step;
  slider.value = value; slider.style.flex = '1';

  const num = document.createElement('input');
  num.type = 'number'; num.step = step;
  num.value = parseFloat(value).toFixed(2);
  num.style.width = '60px';

  slider.addEventListener('input', () => { num.value = parseFloat(slider.value).toFixed(2); });
  num.addEventListener('input', () => { slider.value = num.value; });

  row.appendChild(slider); row.appendChild(num);
  wrap.appendChild(row);
  return wrap;
}

function makeColorRow(label, id, rgb) {
  const wrap = document.createElement('label');
  wrap.textContent = label;
  const input = document.createElement('input');
  input.type = 'color'; input.id = id;
  input.value = rgbToHex(rgb[0], rgb[1], rgb[2]);
  input.style.width = '100%'; input.style.height = '32px'; input.style.cursor = 'pointer';
  wrap.appendChild(input);
  return wrap;
}

// ── Fit tree ──────────────────────────────────────────────────────────────────
function fitTree() {
  const container = document.getElementById('tree-container');
  const W = container.clientWidth;
  const H = container.clientHeight;
  const bounds = gRoot.node().getBBox();
  if (!bounds.width || !bounds.height) return;
  const scale = Math.min(0.9, 0.9 / Math.max(bounds.width / W, bounds.height / H));
  const tx = W / 2 - scale * (bounds.x + bounds.width  / 2);
  const ty = H / 2 - scale * (bounds.y + bounds.height / 2);
  svg.transition().duration(600).call(
    zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale)
  );
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats(node) {
  let total = 0, leaves = 0, depth = 0;
  (function walk(n, d) {
    total++; if (n.arity === 0) leaves++; if (d > depth) depth = d;
    (n.children || []).forEach(c => walk(c, d + 1));
  })(node, 0);
  document.getElementById('tree-stats').innerHTML =
    `<small>${total} nodes · ${leaves} leaves · depth ${depth}</small>`;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function findNode(tree, id) {
  if (tree.id === id) return tree;
  for (const c of tree.children || []) {
    const found = findNode(c, id);
    if (found) return found;
  }
  return null;
}

function jsonHdr() { return {'Content-Type': 'application/json'}; }

function hexToRgb(hex) {
  return [
    parseInt(hex.slice(1,3), 16) / 255,
    parseInt(hex.slice(3,5), 16) / 255,
    parseInt(hex.slice(5,7), 16) / 255,
  ];
}

function rgbToHex(r, g, b) {
  const h = v => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2, '0');
  return `#${h(r)}${h(g)}${h(b)}`;
}

init();
