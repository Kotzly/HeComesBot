import { state } from './state.js';
import { findNode, jsonHdr } from './utils.js';
import { scheduleNodePreview, openModal, closeModal } from './preview.js';
import { renderTree, fitTree, updateStats, buildLegend, toggleCollapse } from './renderer.js';
import { setRendererCallbacks } from './renderer.js';
import {
  buildLeafEditor, buildNodeParamEditor,
  onUpdateLeaf,
  setEditorCallbacks,
} from './editors.js';

// ── Callbacks ─────────────────────────────────────────────────────────────────
function _setUndoEnabled(enabled) {
  document.getElementById('undo-btn').disabled = !enabled;
}

setRendererCallbacks(selectNode, toggleCollapse);
setEditorCallbacks(findNode, selectNode, _setUndoEnabled, () => {
  if (state.sensitivityData) fetchSensitivity();
});

// ── Node selection ────────────────────────────────────────────────────────────
function selectNode(nodeData) {
  state.selectedId = nodeData.id;
  document.getElementById('node-panel').classList.remove('hidden');
  document.getElementById('node-id-display')   .textContent = nodeData.id;
  document.getElementById('node-func-display') .textContent = nodeData.func;
  document.getElementById('node-arity-display').textContent =
    nodeData.arity === 0 ? 'leaf' : nodeData.arity === 1 ? '1 (unary)' : '2 (binary)';
  document.getElementById('func-select').value = nodeData.func;

  const leafEditor       = document.getElementById('leaf-editor');
  const nodeParamsEditor = document.getElementById('node-params-editor');

  if (nodeData.arity === 0) {
    leafEditor.classList.remove('hidden');
    buildLeafEditor(nodeData);
    nodeParamsEditor.classList.add('hidden');
  } else {
    leafEditor.classList.add('hidden');
    document.getElementById('prune-result').textContent = '';
    const specs = (state.funcParamsData || {})[nodeData.func];
    if (specs && specs.length) {
      nodeParamsEditor.classList.remove('hidden');
      buildNodeParamEditor(nodeData);
    } else {
      nodeParamsEditor.classList.add('hidden');
    }
  }

  document.getElementById('node-preview-img').style.display = 'none';
  scheduleNodePreview(nodeData.id);
  renderTree();
}

// ── Undo / Flatten ────────────────────────────────────────────────────────────
async function onUndo() {
  if (!state.treeId) return;
  const btn = document.getElementById('undo-btn');
  btn.disabled = true; btn.textContent = 'Undoing…';
  try {
    const res = await fetch('/api/undo', {
      method: 'POST', headers: jsonHdr(),
      body: JSON.stringify({tree_id: state.treeId}),
    });
    const data = await res.json();
    if (data.error) { _setUndoEnabled(false); return; }
    state.treeData = data.tree;
    state.sensitivityData = null;
    const node = state.selectedId ? findNode(state.treeData, state.selectedId) : null;
    if (node) selectNode(node); else renderTree();
    updateStats(state.treeData);
    _setUndoEnabled(true);
  } finally {
    btn.textContent = 'Undo';
  }
}

async function onFlatten() {
  if (!state.selectedId) return;
  const btn = document.getElementById('flatten-btn');
  btn.disabled = true; btn.textContent = 'Flattening…';
  try {
    const res = await fetch('/api/node/flatten', {
      method: 'POST', headers: jsonHdr(),
      body: JSON.stringify({tree_id: state.treeId, node_id: state.selectedId}),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    state.treeData   = data.tree;
    state.selectedId = data.new_node_id;
    const node = findNode(state.treeData, state.selectedId);
    if (node) selectNode(node); else renderTree();
    updateStats(state.treeData);
    _setUndoEnabled(true);
    if (state.sensitivityData) await fetchSensitivity();
  } finally {
    btn.disabled = false; btn.textContent = 'Flatten to Color';
  }
}

// ── Reference node ────────────────────────────────────────────────────────────
function _updateReferenceBar() {
  const bar = document.getElementById('reference-toolbar');
  if (state.referenceId) {
    document.getElementById('reference-id-display').textContent = state.referenceId;
    bar.classList.remove('hidden');
  } else {
    bar.classList.add('hidden');
  }
}

function onSetReference() {
  if (!state.selectedId) return;
  state.referenceId = state.selectedId;
  _updateReferenceBar();
  renderTree();
  scheduleNodePreview(state.selectedId);
}

function onClearReference() {
  state.referenceId = null;
  _updateReferenceBar();
  renderTree();
  if (state.selectedId) scheduleNodePreview(state.selectedId);
}

// ── Sensitivity / Prune ───────────────────────────────────────────────────────
async function fetchSensitivity() {
  const res = await fetch('/api/sensitivity', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({
      tree_id:           state.treeId,
      delta:             parseFloat(document.getElementById('prune-delta').value),
      reference_node_id: state.referenceId || undefined,
    }),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  state.sensitivityData = data;
  renderTree();
}

async function onSensitivity() {
  if (!state.treeId) return;
  const btn = document.getElementById('sensitivity-btn');
  btn.disabled = true; btn.textContent = 'Calculating…';
  try {
    await fetchSensitivity();
  } finally {
    btn.disabled = false; btn.textContent = 'Show Sensitivity';
  }
}

async function onPrune() {
  if (!state.selectedId) return;
  const btn = document.getElementById('prune-btn');
  const result = document.getElementById('prune-result');
  btn.disabled = true; btn.textContent = 'Pruning…';
  result.textContent = '';
  try {
    const res = await fetch('/api/prune', {
      method: 'POST', headers: jsonHdr(),
      body: JSON.stringify({
        tree_id:   state.treeId,
        node_id:   state.selectedId,
        method:    document.getElementById('prune-method-select').value,
        delta:     parseFloat(document.getElementById('prune-delta').value),
        threshold: parseFloat(document.getElementById('prune-threshold').value),
      }),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    state.treeData        = data.tree;
    state.sensitivityData = null;
    result.textContent = `Pruned ${data.pruned} node${data.pruned !== 1 ? 's' : ''}.`;
    _setUndoEnabled(true);
    const node = findNode(state.treeData, state.selectedId);
    if (node) selectNode(node); else renderTree();
    updateStats(state.treeData);
  } finally {
    btn.disabled = false; btn.textContent = 'Prune';
  }
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
    body: JSON.stringify({tree_id: state.treeId, name}),
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
  state.treeId          = data.tree_id;
  state.treeData        = data.tree;
  state.selectedId      = null;
  state.referenceId     = null;
  state.sensitivityData = null;
  state.collapsedIds.clear();
  _updateReferenceBar();
  _setUndoEnabled(false);
  const meta = data.meta || {};
  if (meta.seed        != null) document.getElementById('seed-input').value        = meta.seed;
  if (meta.dx          != null) document.getElementById('width-input').value       = meta.dx;
  if (meta.dy          != null) document.getElementById('height-input').value      = meta.dy;
  if (meta.min_depth   != null) document.getElementById('min-depth-input').value   = meta.min_depth;
  if (meta.max_depth   != null) document.getElementById('max-depth-input').value   = meta.max_depth;
  if (meta.color_space != null) document.getElementById('color-space-select').value = meta.color_space;

  document.getElementById('tree-hint').style.display    = 'none';
  document.getElementById('preview-btn').disabled       = false;
  document.getElementById('save-btn').disabled          = false;
  document.getElementById('node-panel').classList.add('hidden');

  renderTree();
  fitTree();
  updateStats(state.treeData);
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
      method: 'POST', headers: jsonHdr(),
      body: JSON.stringify({tree_id: state.treeId}),
    });
    const data = await res.json();
    document.getElementById('preview-img').src            = data.image;
    document.getElementById('modal-img').src              = data.image;
    document.getElementById('preview-wrap').style.display = '';
  } finally {
    btn.disabled = false; btn.textContent = 'Render Frame';
  }
}

// ── Node editing ──────────────────────────────────────────────────────────────
async function onApplyFunc() {
  if (!state.selectedId) return;
  const funcName = document.getElementById('func-select').value;
  const res  = await fetch('/api/node/set-func', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({tree_id: state.treeId, node_id: state.selectedId, func_name: funcName}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  state.treeData = data.tree;
  const node = findNode(state.treeData, state.selectedId);
  if (node) selectNode(node); else renderTree();
  updateStats(state.treeData);
}

async function onRegen() {
  if (!state.selectedId) return;
  const seed = parseInt(document.getElementById('regen-seed-input').value);
  const res  = await fetch('/api/node/regenerate', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({tree_id: state.treeId, node_id: state.selectedId, seed}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  state.treeData        = data.tree;
  state.selectedId      = data.new_node_id;
  state.sensitivityData = null;
  state.collapsedIds.clear();
  const node = findNode(state.treeData, state.selectedId);
  if (node) selectNode(node); else renderTree();
  updateStats(state.treeData);
}

// ── Init ─────────────────────────────────────────────────────────────────────
function randInt() { return Math.floor(Math.random() * 1e9); }

async function init() {
  const [funcsRes, persRes, fparamsRes, pruneMethodsRes] = await Promise.all([
    fetch('/api/functions'),
    fetch('/api/personalities'),
    fetch('/api/function-params'),
    fetch('/api/prune-methods'),
  ]);
  state.functionsData  = await funcsRes.json();
  state.funcParamsData = await fparamsRes.json();
  const personalities  = await persRes.json();
  const pruneMethods   = await pruneMethodsRes.json();

  const pmSel = document.getElementById('prune-method-select');
  pruneMethods.forEach(m => pmSel.appendChild(new Option(m, m)));

  const pSel = document.getElementById('personality-select');
  personalities.forEach(p => pSel.appendChild(new Option(p, p)));

  const fSel = document.getElementById('func-select');
  const arityLabels = {'0':'Leaf (arity 0)','1':'Unary (arity 1)','2':'Binary (arity 2)'};
  Object.entries(state.functionsData).sort().forEach(([arity, funcs]) => {
    const group = document.createElement('optgroup');
    group.label = arityLabels[arity] || `Arity ${arity}`;
    funcs.forEach(f => group.appendChild(new Option(f, f)));
    fSel.appendChild(group);
  });

  document.getElementById('regen-seed-input').value = randInt();

  document.getElementById('build-btn')             .addEventListener('click', onBuild);
  document.getElementById('save-btn')              .addEventListener('click', onSave);
  document.getElementById('load-btn')              .addEventListener('click', onLoad);
  document.getElementById('refresh-trees-btn')     .addEventListener('click', refreshTreeList);
  document.getElementById('preview-btn')           .addEventListener('click', onPreview);
  document.getElementById('preview-img')           .addEventListener('click', openModal);
  document.getElementById('modal-close-btn')       .addEventListener('click', closeModal);
  document.getElementById('preview-modal')         .addEventListener('click', e => {
    if (e.target.id === 'preview-modal') closeModal();
  });
  document.getElementById('apply-func-btn')        .addEventListener('click', onApplyFunc);
  document.getElementById('regen-btn')             .addEventListener('click', onRegen);
  document.getElementById('update-leaf-btn')       .addEventListener('click', onUpdateLeaf);
  document.getElementById('sensitivity-btn')       .addEventListener('click', onSensitivity);
  document.getElementById('clear-sensitivity-btn') .addEventListener('click', () => {
    state.sensitivityData = null; renderTree();
  });
  document.getElementById('flatten-btn')           .addEventListener('click', onFlatten);
  document.getElementById('undo-btn')              .addEventListener('click', onUndo);
  document.getElementById('set-reference-btn')     .addEventListener('click', onSetReference);
  document.getElementById('clear-reference-btn')   .addEventListener('click', onClearReference);
  document.getElementById('prune-btn')             .addEventListener('click', onPrune);
  document.getElementById('random-seed-btn')       .addEventListener('click', () => {
    document.getElementById('seed-input').value = randInt();
  });
  document.getElementById('regen-random-btn')      .addEventListener('click', () => {
    document.getElementById('regen-seed-input').value = randInt();
  });

  window.addEventListener('resize', () => { if (state.treeData) renderTree(); });

  buildLegend();
  await refreshTreeList();
}

init();
