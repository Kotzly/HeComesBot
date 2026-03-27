'use strict';

// ── State ────────────────────────────────────────────────────────────────────
let treeId        = null;
let treeData      = null;
let selectedId    = null;
let collapsedIds  = new Set();
let functionsData = null;

// ── D3 setup ─────────────────────────────────────────────────────────────────
const svg  = d3.select('#tree-svg');
const root = svg.append('g').attr('class', 'root-g');
const zoom = d3.zoom().scaleExtent([0.05, 4]).on('zoom', e => root.attr('transform', e.transform));
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

  // Populate personality dropdown
  const pSel = document.getElementById('personality-select');
  personalities.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p; opt.textContent = p;
    pSel.appendChild(opt);
  });

  // Populate function select (grouped by arity)
  const fSel = document.getElementById('func-select');
  const arityLabels = {'0': 'Leaf (arity 0)', '1': 'Unary (arity 1)', '2': 'Binary (arity 2)'};
  Object.entries(functionsData).sort().forEach(([arity, funcs]) => {
    const group = document.createElement('optgroup');
    group.label = arityLabels[arity] || `Arity ${arity}`;
    funcs.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f; opt.textContent = f;
      group.appendChild(opt);
    });
    fSel.appendChild(group);
  });

  // Seed inputs — randomise regen seed on load
  document.getElementById('regen-seed-input').value = randInt();

  // Wire events
  document.getElementById('build-btn')       .addEventListener('click', onBuild);
  document.getElementById('preview-btn')     .addEventListener('click', onPreview);
  document.getElementById('apply-func-btn')  .addEventListener('click', onApplyFunc);
  document.getElementById('regen-btn')       .addEventListener('click', onRegen);
  document.getElementById('random-seed-btn') .addEventListener('click', () => {
    document.getElementById('seed-input').value = randInt();
  });
  document.getElementById('regen-random-btn').addEventListener('click', () => {
    document.getElementById('regen-seed-input').value = randInt();
  });
  document.getElementById('modal-close-btn') .addEventListener('click', closeModal);
  document.getElementById('preview-modal')   .addEventListener('click', e => {
    if (e.target.id === 'preview-modal') closeModal();
  });

  // Resize
  window.addEventListener('resize', () => { if (treeData) renderTree(); });
}

function randInt() { return Math.floor(Math.random() * 1e9); }

// ── Build ─────────────────────────────────────────────────────────────────────
async function onBuild() {
  const btn = document.getElementById('build-btn');
  btn.disabled = true;
  btn.textContent = 'Building…';

  const body = {
    seed:        parseInt(document.getElementById('seed-input').value),
    width:       parseInt(document.getElementById('width-input').value),
    height:      parseInt(document.getElementById('height-input').value),
    min_depth:   parseInt(document.getElementById('min-depth-input').value),
    max_depth:   parseInt(document.getElementById('max-depth-input').value),
    personality: document.getElementById('personality-select').value,
  };

  try {
    const res  = await fetch('/api/build', {method: 'POST', headers: jsonHeaders(), body: JSON.stringify(body)});
    const data = await res.json();
    treeId    = data.tree_id;
    treeData  = data.tree;
    selectedId = null;
    collapsedIds.clear();

    document.getElementById('tree-hint').style.display = 'none';
    document.getElementById('preview-btn').disabled = false;
    document.getElementById('node-panel').classList.add('hidden');

    renderTree();
    fitTree();
    updateStats(treeData);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Build Tree';
  }
}

// ── Preview ───────────────────────────────────────────────────────────────────
async function onPreview() {
  const btn = document.getElementById('preview-btn');
  btn.disabled = true;
  btn.textContent = 'Rendering…';
  try {
    const res  = await fetch('/api/preview', {method: 'POST', headers: jsonHeaders(), body: JSON.stringify({tree_id: treeId})});
    const data = await res.json();
    document.getElementById('modal-img').src = data.image;
    document.getElementById('preview-modal').classList.remove('hidden');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Preview Frame';
  }
}

function closeModal() { document.getElementById('preview-modal').classList.add('hidden'); }

// ── Node editing ──────────────────────────────────────────────────────────────
async function onApplyFunc() {
  if (!selectedId) return;
  const funcName = document.getElementById('func-select').value;
  const res  = await fetch('/api/node/set-func', {
    method: 'POST', headers: jsonHeaders(),
    body: JSON.stringify({tree_id: treeId, node_id: selectedId, func_name: funcName}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  treeData = data.tree;
  const node = findNode(treeData, selectedId);
  if (node) selectNode(node);
  renderTree();
}

async function onRegen() {
  if (!selectedId) return;
  const seed = parseInt(document.getElementById('regen-seed-input').value);
  const res  = await fetch('/api/node/regenerate', {
    method: 'POST', headers: jsonHeaders(),
    body: JSON.stringify({tree_id: treeId, node_id: selectedId, seed}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  treeData   = data.tree;
  selectedId = data.new_node_id;
  // Remove collapsed state for replaced subtree (new IDs)
  collapsedIds.clear();
  const node = findNode(treeData, selectedId);
  if (node) selectNode(node);
  renderTree();
  updateStats(treeData);
}

// ── Tree rendering ────────────────────────────────────────────────────────────
function renderTree() {
  const container = document.getElementById('tree-container');
  const W = container.clientWidth;
  const H = container.clientHeight;
  svg.attr('width', W).attr('height', H);
  root.selectAll('*').remove();

  const visible = visibleSubtree(treeData);
  const hier    = d3.hierarchy(visible, d => d._kids.length ? d._kids : null);

  const nodeSpacing = Math.max(18, Math.min(32, 800 / (hier.leaves().length + 1)));
  d3.tree().nodeSize([nodeSpacing, 180])(hier);

  const nodes = hier.descendants();
  const minX  = d3.min(nodes, d => d.x);
  const offY  = H / 2 - minX;
  const offX  = 80;

  // Links
  root.selectAll('.link')
    .data(hier.links())
    .join('path')
    .attr('class', 'link')
    .attr('d', d3.linkHorizontal()
      .x(d => d.y + offX)
      .y(d => d.x + offY));

  // Node groups
  const nodeG = root.selectAll('.node')
    .data(nodes)
    .join('g')
    .attr('class', 'node')
    .attr('transform', d => `translate(${d.y + offX},${d.x + offY})`);

  // Circles
  nodeG.append('circle')
    .attr('r', 11)
    .attr('fill', d => arityColor(d.data.arity))
    .attr('stroke', d => d.data.id === selectedId ? '#fff' : '#333')
    .attr('stroke-width', d => d.data.id === selectedId ? 2.5 : 1)
    .on('click', (event, d) => { event.stopPropagation(); selectNode(d.data); })
    .on('dblclick', (event, d) => { event.stopPropagation(); toggleCollapse(d.data); });

  // Collapse indicator inside collapsed nodes
  nodeG.filter(d => collapsedIds.has(d.data.id) && d.data.arity > 0)
    .append('text')
    .attr('dy', '0.35em')
    .attr('text-anchor', 'middle')
    .attr('font-size', '9px')
    .attr('fill', '#fff')
    .attr('pointer-events', 'none')
    .text('…');

  // Labels
  const hasChildren = d => d.data._kids.length > 0 && !collapsedIds.has(d.data.id);
  nodeG.append('text')
    .attr('class', 'node-label')
    .attr('x', d => hasChildren(d) ? -16 : 16)
    .attr('dy', '0.35em')
    .attr('text-anchor', d => hasChildren(d) ? 'end' : 'start')
    .text(d => d.data.func);
}

function visibleSubtree(node) {
  const copy = {id: node.id, func: node.func, arity: node.arity, _kids: []};
  if (node.children.length && !collapsedIds.has(node.id)) {
    copy._kids = node.children.map(visibleSubtree);
  }
  return copy;
}

function toggleCollapse(nodeData) {
  if (nodeData.arity === 0) return;
  if (collapsedIds.has(nodeData.id)) collapsedIds.delete(nodeData.id);
  else collapsedIds.add(nodeData.id);
  renderTree();
}

function selectNode(nodeData) {
  selectedId = nodeData.id;
  document.getElementById('node-panel').classList.remove('hidden');
  document.getElementById('node-id-display')  .textContent = nodeData.id;
  document.getElementById('node-func-display').textContent = nodeData.func;
  document.getElementById('node-arity-display').textContent =
    nodeData.arity === 0 ? 'leaf' : nodeData.arity === 1 ? '1 (unary)' : '2 (binary)';
  document.getElementById('func-select').value = nodeData.func;
  renderTree();
}

// ── Fit tree to viewport ──────────────────────────────────────────────────────
function fitTree() {
  const container = document.getElementById('tree-container');
  const W = container.clientWidth;
  const H = container.clientHeight;
  const bounds = root.node().getBBox();
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
  function walk(n, d) {
    total++;
    if (n.arity === 0) leaves++;
    if (d > depth) depth = d;
    (n.children || []).forEach(c => walk(c, d + 1));
  }
  walk(node, 0);
  document.getElementById('tree-stats').innerHTML =
    `<small>${total} nodes &nbsp;·&nbsp; ${leaves} leaves &nbsp;·&nbsp; depth ${depth}</small>`;
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

function jsonHeaders() { return {'Content-Type': 'application/json'}; }

init();
