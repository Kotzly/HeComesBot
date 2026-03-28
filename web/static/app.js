'use strict';

// ── State ────────────────────────────────────────────────────────────────────
let treeId           = null;
let treeData         = null;
let selectedId       = null;
let referenceId      = null;
let collapsedIds     = new Set();
let functionsData    = null;
let funcParamsData   = null;
let sensitivityData  = null;

// ── D3 setup ─────────────────────────────────────────────────────────────────
const svg   = d3.select('#tree-svg');
const gRoot = svg.append('g').attr('class', 'root-g');
const zoom  = d3.zoom().scaleExtent([0.05, 4]).on('zoom', e => gRoot.attr('transform', e.transform));
svg.call(zoom);

// ── Symbol definitions ────────────────────────────────────────────────────────
const SYM_SIZE = 280;

function symbolType(node) {
  if (node.arity === 1) return d3.symbolDiamond;
  if (node.arity === 2) return d3.symbolSquare;
  switch (node.func) {
    case 'rand_color': return d3.symbolCircle;
    case 'cone':       return d3.symbolTriangle;
    case 'circle':     return d3.symbolStar;
    case 'x_var':      return d3.symbolCross;
    case 'y_var':      return d3.symbolWye;
    default:           return d3.symbolCircle;
  }
}

function symbolPath(node) {
  return d3.symbol().type(symbolType(node)).size(SYM_SIZE)();
}

// ── Fill / stroke ─────────────────────────────────────────────────────────────
function fillFor(node) {
  if (node.arity > 0) return node.arity === 1 ? '#569cd6' : '#ce9178';

  switch (node.func) {
    case 'rand_color': {
      const c = (node.params || {}).color || [0.5, 0.5, 0.5];
      return colorParamToHex(c);
    }
    case 'circle':
    case 'cone':
    case 'x_var':
    case 'y_var': {
      const c = (node.params || {}).color;
      return c ? colorParamToHex(c) : '#c792ea';
    }
    default:
      return '#4ec9b0';
  }
}

// Returns {color, width} for the symbol stroke
function strokeFor(node, isSelected) {
  if (isSelected) return {color: '#ffffff', width: 2.5};

  if (node.arity === 0 && (node.func === 'circle' || node.func === 'cone')) {
    const p        = node.params || {};
    const areaFrac = Math.min(Math.PI * (p.rx || 0.5) * (p.ry || 0.5) / 4, 1);
    return {color: '#aaa', width: 1 + 5 * areaFrac};
  }
  return {color: '#333', width: 1};
}

// ── Node preview ─────────────────────────────────────────────────────────────
let _previewTimer = null;

function scheduleNodePreview(nodeId) {
  clearTimeout(_previewTimer);
  _previewTimer = setTimeout(() => fetchNodePreview(nodeId), 120);
}

async function fetchNodePreview(nodeId) {
  if (!treeId || nodeId !== selectedId) return;
  const previewNodeId = referenceId || nodeId;
  const loading = document.getElementById('node-preview-loading');
  const img     = document.getElementById('node-preview-img');
  loading.style.display = '';
  img.style.display = 'none';
  try {
    const res  = await fetch('/api/node/preview', {
      method: 'POST', headers: jsonHdr(),
      body: JSON.stringify({tree_id: treeId, node_id: previewNodeId}),
    });
    const data = await res.json();
    if (data.error || nodeId !== selectedId) return;
    img.src = data.image;
    img.style.display = '';
    img.onclick = () => {
      document.getElementById('modal-img').src = data.image;
      openModal();
    };
  } finally {
    loading.style.display = 'none';
  }
}

// ── Pruning ───────────────────────────────────────────────────────────────────

function _setUndoEnabled(enabled) {
  document.getElementById('undo-btn').disabled = !enabled;
}

async function onUndo() {
  if (!treeId) return;
  const btn = document.getElementById('undo-btn');
  btn.disabled = true; btn.textContent = 'Undoing…';
  try {
    const res = await fetch('/api/undo', {
      method: 'POST', headers: jsonHdr(),
      body: JSON.stringify({tree_id: treeId}),
    });
    const data = await res.json();
    if (data.error) { _setUndoEnabled(false); return; }
    treeData = data.tree;
    sensitivityData = null;
    const node = selectedId ? findNode(treeData, selectedId) : null;
    if (node) selectNode(node); else renderTree();
    updateStats(treeData);
    _setUndoEnabled(true);
  } finally {
    btn.textContent = 'Undo';
  }
}

async function onFlatten() {
  if (!selectedId) return;
  const btn = document.getElementById('flatten-btn');
  btn.disabled = true; btn.textContent = 'Flattening…';
  try {
    const res = await fetch('/api/node/flatten', {
      method: 'POST', headers: jsonHdr(),
      body: JSON.stringify({tree_id: treeId, node_id: selectedId}),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    treeData   = data.tree;
    selectedId = data.new_node_id;
    const node = findNode(treeData, selectedId);
    if (node) selectNode(node); else renderTree();
    updateStats(treeData);
    _setUndoEnabled(true);
    if (sensitivityData) await fetchSensitivity();
  } finally {
    btn.disabled = false; btn.textContent = 'Flatten to Color';
  }
}

function _updateReferenceBar() {
  const bar = document.getElementById('reference-toolbar');
  if (referenceId) {
    document.getElementById('reference-id-display').textContent = referenceId;
    bar.classList.remove('hidden');
  } else {
    bar.classList.add('hidden');
  }
}

function onSetReference() {
  if (!selectedId) return;
  referenceId     = selectedId;
  sensitivityData = null;
  _updateReferenceBar();
  renderTree();
  scheduleNodePreview(selectedId);
}

function onClearReference() {
  referenceId     = null;
  sensitivityData = null;
  _updateReferenceBar();
  renderTree();
  if (selectedId) scheduleNodePreview(selectedId);
}

async function fetchSensitivity() {
  const res = await fetch('/api/sensitivity', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({
      tree_id:           treeId,
      delta:             parseFloat(document.getElementById('prune-delta').value),
      reference_node_id: referenceId || undefined,
    }),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  sensitivityData = data;
  renderTree();
}

async function onSensitivity() {
  if (!selectedId) return;
  const btn = document.getElementById('sensitivity-btn');
  btn.disabled = true; btn.textContent = 'Calculating…';
  try {
    await fetchSensitivity();
  } finally {
    btn.disabled = false; btn.textContent = 'Show Sensitivity';
  }
}

async function onPrune() {
  if (!selectedId) return;
  const btn = document.getElementById('prune-btn');
  const result = document.getElementById('prune-result');
  btn.disabled = true; btn.textContent = 'Pruning…';
  result.textContent = '';
  try {
    const res = await fetch('/api/prune', {
      method: 'POST', headers: jsonHdr(),
      body: JSON.stringify({
        tree_id:   treeId,
        node_id:   selectedId,
        method:    document.getElementById('prune-method-select').value,
        delta:     parseFloat(document.getElementById('prune-delta').value),
        threshold: parseFloat(document.getElementById('prune-threshold').value),
      }),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    treeData        = data.tree;
    sensitivityData = null;
    result.textContent = `Pruned ${data.pruned} node${data.pruned !== 1 ? 's' : ''}.`;
    _setUndoEnabled(true);
    const node = findNode(treeData, selectedId);
    if (node) selectNode(node); else renderTree();
    updateStats(treeData);
  } finally {
    btn.disabled = false; btn.textContent = 'Prune';
  }
}

// ── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  const [funcsRes, persRes, fparamsRes, pruneMethodsRes] = await Promise.all([
    fetch('/api/functions'),
    fetch('/api/personalities'),
    fetch('/api/function-params'),
    fetch('/api/prune-methods'),
  ]);
  functionsData  = await funcsRes.json();
  funcParamsData = await fparamsRes.json();
  const personalities = await persRes.json();
  const pruneMethods  = await pruneMethodsRes.json();

  const pmSel = document.getElementById('prune-method-select');
  pruneMethods.forEach(m => pmSel.appendChild(new Option(m, m)));

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

  document.getElementById('build-btn')        .addEventListener('click', onBuild);
  document.getElementById('save-btn')         .addEventListener('click', onSave);
  document.getElementById('load-btn')         .addEventListener('click', onLoad);
  document.getElementById('refresh-trees-btn').addEventListener('click', refreshTreeList);
  document.getElementById('preview-btn')      .addEventListener('click', onPreview);
  document.getElementById('preview-img')      .addEventListener('click', openModal);
  document.getElementById('modal-close-btn')  .addEventListener('click', closeModal);
  document.getElementById('preview-modal')    .addEventListener('click', e => {
    if (e.target.id === 'preview-modal') closeModal();
  });
  document.getElementById('apply-func-btn')        .addEventListener('click', onApplyFunc);
  document.getElementById('regen-btn')             .addEventListener('click', onRegen);
  document.getElementById('update-leaf-btn')       .addEventListener('click', onUpdateLeaf);
  document.getElementById('sensitivity-btn')       .addEventListener('click', onSensitivity);
  document.getElementById('clear-sensitivity-btn') .addEventListener('click', () => { sensitivityData = null; renderTree(); });
  document.getElementById('flatten-btn')           .addEventListener('click', onFlatten);
  document.getElementById('undo-btn')              .addEventListener('click', onUndo);
  document.getElementById('set-reference-btn')     .addEventListener('click', onSetReference);
  document.getElementById('clear-reference-btn')   .addEventListener('click', onClearReference);
  document.getElementById('prune-btn')             .addEventListener('click', onPrune);
  document.getElementById('random-seed-btn')  .addEventListener('click', () => {
    document.getElementById('seed-input').value = randInt();
  });
  document.getElementById('regen-random-btn') .addEventListener('click', () => {
    document.getElementById('regen-seed-input').value = randInt();
  });

  window.addEventListener('resize', () => { if (treeData) renderTree(); });

  buildLegend();
  await refreshTreeList();
}

function randInt() { return Math.floor(Math.random() * 1e9); }

// ── Legend ────────────────────────────────────────────────────────────────────
function buildLegend() {
  const entries = [
    {label: 'rand_color', node: {arity:0, func:'rand_color', params:{color:[0.75,0.35,0.6]}}},
    {label: 'circle',     node: {arity:0, func:'circle',     params:{cx:0,cy:0,rx:0.6,ry:0.6}}},
    {label: 'cone',       node: {arity:0, func:'cone',       params:{cx:0.8,cy:0.8,rx:0.4,ry:0.4}}},
    {label: 'x_var',      node: {arity:0, func:'x_var',      params:{}}},
    {label: 'y_var',      node: {arity:0, func:'y_var',      params:{}}},
    {label: 'other leaf', node: {arity:0, func:'_other',     params:{}}},
    {label: 'unary',      node: {arity:1, func:null,         params:{}}},
    {label: 'binary',     node: {arity:2, func:null,         params:{}}},
  ];

  const rowH = 22;
  const container = document.getElementById('legend');
  container.innerHTML = '';

  const lsvg = d3.select(container).append('svg')
    .attr('width', '100%').attr('height', entries.length * rowH);

  entries.forEach((e, i) => {
    const g = lsvg.append('g').attr('transform', `translate(14,${i * rowH + rowH / 2})`);
    const s = strokeFor(e.node, false);
    g.append('path')
      .attr('d', symbolPath(e.node))
      .attr('fill', fillFor(e.node))
      .attr('stroke', s.color)
      .attr('stroke-width', Math.min(s.width, 3));
    g.append('text')
      .attr('x', 18).attr('dy', '0.35em')
      .attr('font-size', '11px').attr('fill', '#999')
      .text(e.label);
  });
}

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
  treeId          = data.tree_id;
  treeData        = data.tree;
  selectedId      = null;
  referenceId     = null;
  sensitivityData = null;
  collapsedIds.clear();
  _updateReferenceBar();
  _setUndoEnabled(false);
  const meta = data.meta || {};
  if (meta.seed      != null) document.getElementById('seed-input').value      = meta.seed;
  if (meta.dx        != null) document.getElementById('width-input').value     = meta.dx;
  if (meta.dy        != null) document.getElementById('height-input').value    = meta.dy;
  if (meta.min_depth != null) document.getElementById('min-depth-input').value = meta.min_depth;
  if (meta.max_depth   != null) document.getElementById('max-depth-input').value    = meta.max_depth;
  if (meta.color_space != null) document.getElementById('color-space-select').value = meta.color_space;

  document.getElementById('tree-hint').style.display    = 'none';
  document.getElementById('preview-btn').disabled       = false;
  document.getElementById('save-btn').disabled          = false;
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
    const res = await fetch('/api/build', {
      method: 'POST', headers: jsonHdr(),
      body: JSON.stringify({
        seed:        parseInt(document.getElementById('seed-input').value),
        width:       parseInt(document.getElementById('width-input').value),
        height:      parseInt(document.getElementById('height-input').value),
        min_depth:   parseInt(document.getElementById('min-depth-input').value),
        max_depth:   parseInt(document.getElementById('max-depth-input').value),
        personality:  document.getElementById('personality-select').value,
        color_space:  document.getElementById('color-space-select').value,
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
    document.getElementById('preview-img').src       = data.image;
    document.getElementById('modal-img').src         = data.image;
    document.getElementById('preview-wrap').style.display = '';
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
  treeData        = data.tree;
  selectedId      = data.new_node_id;
  sensitivityData = null;
  collapsedIds.clear();
  const node = findNode(treeData, selectedId);
  if (node) selectNode(node); else renderTree();
  updateStats(treeData);
}

async function onUpdateLeaf() {
  if (!selectedId) return;
  const node = findNode(treeData, selectedId);
  if (!node || node.arity !== 0) return;

  const params = collectLeafParams(node.func);

  const res  = await fetch('/api/leaf/set-params', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({tree_id: treeId, node_id: selectedId, params}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  treeData = data.tree;
  _setUndoEnabled(true);
  const updated = findNode(treeData, selectedId);
  if (updated) {
    selectNode(updated);
    scheduleNodePreview(selectedId);
  }
}

function collectLeafParams(func) {
  const params = {};
  if (func === 'rand_color') {
    params.color = hexToColorParam(document.getElementById('lp-color').value);
    return params;
  }
  const specs = (funcParamsData || {})[func] || [];
  specs.forEach(spec => {
    if (spec.type === 'float') {
      params[spec.name] = parseFloat(document.getElementById(`lp-${spec.name}`).value);
    } else if (spec.type === 'color') {
      params[spec.name] = hexToColorParam(document.getElementById(`lp-${spec.name}`).value);
    }
  });
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
    .data(hier.links()).join('path').attr('class', 'link')
    .attr('d', d3.linkHorizontal().x(d => d.y + offX).y(d => d.x + offY));

  const nodeG = gRoot.selectAll('.node')
    .data(nodes).join('g').attr('class', 'node')
    .attr('transform', d => `translate(${d.y + offX},${d.x + offY})`);

  // Symbol path
  nodeG.append('path')
    .attr('d', d => symbolPath(d.data))
    .attr('fill', d => fillFor(d.data))
    .attr('stroke', d => strokeFor(d.data, d.data.id === selectedId).color)
    .attr('stroke-width', d => strokeFor(d.data, d.data.id === selectedId).width)
    .style('cursor', 'pointer')
    .on('click',   (e, d) => { e.stopPropagation(); selectNode(d.data); })
    .on('dblclick', (e, d) => { e.stopPropagation(); toggleCollapse(d.data); });

  // Collapse indicator
  nodeG.filter(d => collapsedIds.has(d.data.id) && d.data.arity > 0)
    .append('text')
    .attr('dy', '0.35em').attr('text-anchor', 'middle')
    .attr('font-size', '9px').attr('fill', '#fff')
    .attr('pointer-events', 'none').text('…');

  // Labels
  const hasKids = d => d.data._kids.length > 0 && !collapsedIds.has(d.data.id);
  nodeG.append('text').attr('class', 'node-label')
    .attr('x', d => hasKids(d) ? -16 : 16)
    .attr('dy', '0.35em')
    .attr('text-anchor', d => hasKids(d) ? 'end' : 'start')
    .text(d => d.data.func);

  if (sensitivityData) {
    nodeG.filter(d => sensitivityData[d.data.id] !== undefined)
      .append('text')
      .attr('y', -14).attr('text-anchor', 'middle')
      .attr('font-size', '8px').attr('fill', '#f5a623')
      .attr('pointer-events', 'none')
      .text(d => `R:${sensitivityData[d.data.id].root.toFixed(3)}`);

    nodeG.filter(d => sensitivityData[d.data.id] !== undefined)
      .append('text')
      .attr('y', 20).attr('text-anchor', 'middle')
      .attr('font-size', '8px').attr('fill', '#7ec8e3')
      .attr('pointer-events', 'none')
      .text(d => `L:${sensitivityData[d.data.id].leaf.toFixed(3)}`);
  }
}

function visibleSubtree(node) {
  const copy = {
    id: node.id, func: node.func, arity: node.arity,
    params: node.params, delta: node.delta, _kids: [],
  };
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

  const leafEditor       = document.getElementById('leaf-editor');
  const nodeParamsEditor = document.getElementById('node-params-editor');
  const pruneEditor      = document.getElementById('prune-editor');

  if (nodeData.arity === 0) {
    leafEditor.classList.remove('hidden');
    buildLeafEditor(nodeData);
    nodeParamsEditor.classList.add('hidden');
    pruneEditor.classList.add('hidden');
  } else {
    leafEditor.classList.add('hidden');
    pruneEditor.classList.remove('hidden');
    document.getElementById('prune-result').textContent = '';
    const specs = (funcParamsData || {})[nodeData.func];
    if (specs && specs.length) {
      nodeParamsEditor.classList.remove('hidden');
      buildNodeParamEditor(nodeData);
    } else {
      nodeParamsEditor.classList.add('hidden');
    }
  }

  // Reset preview state and kick off render
  document.getElementById('node-preview-img').style.display = 'none';
  scheduleNodePreview(nodeData.id);

  renderTree();
}

// ── Leaf editor ───────────────────────────────────────────────────────────────
function buildLeafEditor(node) {
  const controls = document.getElementById('leaf-controls');
  controls.innerHTML = '';

  const p = node.params || {};
  const specs = (funcParamsData || {})[node.func];

  if (node.func === 'rand_color') {
    controls.appendChild(makeColorRow('Color', 'lp-color', p.color || [0.5, 0.5, 0.5]));
  } else if (specs && specs.length) {
    specs.forEach(spec => {
      if (spec.type === 'float') {
        controls.appendChild(makeSliderRow(spec.label, `lp-${spec.name}`, spec.min, spec.max, 0.01, p[spec.name] ?? spec.min ?? 0));
      } else if (spec.type === 'color') {
        controls.appendChild(makeColorRow(spec.label, `lp-${spec.name}`, p[spec.name] || [0.5, 0.5, 0.5]));
      }
    });
  } else {
    const msg = document.createElement('div');
    msg.className = 'muted';
    msg.textContent = 'No editable params for this leaf type.';
    controls.appendChild(msg);
  }

  controls.addEventListener('input', () => {
    if (document.getElementById('auto-update-chk').checked) {
      onUpdateLeaf();
    }
  });
}

// ── Non-leaf param editor ─────────────────────────────────────────────────────
function buildNodeParamEditor(node) {
  const controls = document.getElementById('node-params-controls');
  controls.innerHTML = '';
  const specs = (funcParamsData || {})[node.func] || [];
  const p = node.params || {};

  specs.forEach(spec => {
    if (spec.type === 'float') {
      const val = p[spec.name] ?? spec.min ?? 0;
      controls.appendChild(makeSliderRow(spec.label, `np-${spec.name}`, spec.min, spec.max, 0.01, val));
    } else if (spec.type === 'int' && spec.choices) {
      controls.appendChild(makeSelectRow(spec.label, `np-${spec.name}`, spec.choices, p[spec.name] ?? spec.choices[0]));
    } else if (spec.type === 'angles') {
      const angles = p[spec.name] || [0, 0, 0];
      ['Z', 'Y', 'X'].forEach((axis, i) => {
        controls.appendChild(makeSliderRow(`${spec.label} ${axis}`, `np-${spec.name}-${i}`, 0, 6.2832, 0.01, angles[i]));
      });
    }
  });

  controls.addEventListener('input', () => {
    if (document.getElementById('auto-update-chk').checked) {
      onUpdateNodeParams();
    }
  });
}

function collectNodeParams(funcName) {
  const specs = (funcParamsData || {})[funcName] || [];
  const params = {};
  specs.forEach(spec => {
    if (spec.type === 'float') {
      params[spec.name] = parseFloat(document.getElementById(`np-${spec.name}`).value);
    } else if (spec.type === 'int') {
      params[spec.name] = parseInt(document.getElementById(`np-${spec.name}`).value);
    } else if (spec.type === 'angles') {
      params[spec.name] = [0, 1, 2].map(i => parseFloat(document.getElementById(`np-${spec.name}-${i}`).value));
    }
  });
  return params;
}

async function onUpdateNodeParams() {
  if (!selectedId) return;
  const node = findNode(treeData, selectedId);
  if (!node || node.arity === 0) return;

  const params = collectNodeParams(node.func);
  const res  = await fetch('/api/node/set-params', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({tree_id: treeId, node_id: selectedId, params}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  treeData = data.tree;
  const updated = findNode(treeData, selectedId);
  if (updated) {
    selectNode(updated);
    scheduleNodePreview(selectedId);
  }
}

function makeSelectRow(label, id, choices, value) {
  const wrap = document.createElement('label');
  wrap.textContent = label;
  const sel = document.createElement('select');
  sel.id = id;
  choices.forEach(c => {
    const opt = new Option(c, c);
    if (c == value) opt.selected = true;
    sel.appendChild(opt);
  });
  wrap.appendChild(sel);
  return wrap;
}

function makeSliderRow(label, id, min, max, step, value) {
  const wrap = document.createElement('label');
  wrap.textContent = label;
  const row = document.createElement('div');
  row.className = 'input-row';

  const slider = document.createElement('input');
  slider.type = 'range'; slider.id = id;
  slider.min = min; slider.max = max; slider.step = step; slider.value = value;
  slider.style.flex = '1';

  const num = document.createElement('input');
  num.type = 'number'; num.step = step;
  num.value = parseFloat(value).toFixed(2); num.style.width = '60px';

  slider.addEventListener('input', () => { num.value = parseFloat(slider.value).toFixed(2); });
  num.addEventListener('input',   () => { slider.value = num.value; });

  row.appendChild(slider); row.appendChild(num);
  wrap.appendChild(row);
  return wrap;
}

function makeColorRow(label, id, colorParam) {
  const wrap  = document.createElement('label');
  wrap.textContent = label;
  const input = document.createElement('input');
  input.type = 'color'; input.id = id;
  input.value = colorParamToHex(colorParam);
  input.style.cssText = 'width:100%;height:32px;cursor:pointer';
  wrap.appendChild(input);
  return wrap;
}

// ── Fit / Stats ───────────────────────────────────────────────────────────────
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

function updateStats(node) {
  let total = 0, leaves = 0, depth = 0;
  (function walk(n, d) {
    total++; if (n.arity === 0) leaves++; if (d > depth) depth = d;
    (n.children || []).forEach(c => walk(c, d + 1));
  })(node, 0);
  document.getElementById('tree-stats').innerHTML =
    `<small>${total} nodes · ${leaves} leaves · depth ${depth}</small>`;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
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

function hsvToRgb(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p]; case 1: return [q, v, p]; case 2: return [p, v, t];
    case 3: return [p, q, v]; case 4: return [t, p, v]; case 5: return [v, p, q];
  }
}

function rgbToHsv(r, g, b) {
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
  const s = max === 0 ? 0 : d / max, v = max;
  let h = 0;
  if (d !== 0) {
    if      (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (max === g) h = ((b - r) / d + 2) / 6;
    else                h = ((r - g) / d + 4) / 6;
  }
  return [h, s, v];
}

function getColorSpace() {
  return (document.getElementById('color-space-select') || {}).value || 'rgb';
}

// Stored color param → hex for the color picker
function colorParamToHex(c) {
  const cs = getColorSpace();
  if (cs === 'hsv') {
    const [r, g, b] = hsvToRgb(((c[0] % 1) + 1) % 1, Math.min(1, Math.max(0, c[1])), Math.min(1, Math.max(0, c[2])));
    return rgbToHex(r, g, b);
  }
  if (cs === 'cmy') return rgbToHex(1 - c[0], 1 - c[1], 1 - c[2]);
  return rgbToHex(c[0], c[1], c[2]);
}

// Hex from color picker → stored color param
function hexToColorParam(hex) {
  const [r, g, b] = hexToRgb(hex);
  const cs = getColorSpace();
  if (cs === 'hsv') return rgbToHsv(r, g, b);
  if (cs === 'cmy') return [1 - r, 1 - g, 1 - b];
  return [r, g, b];
}

init();
