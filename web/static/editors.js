import { state } from './state.js';
import { jsonHdr, colorParamToHex, hexToColorParam } from './utils.js';
import { scheduleNodePreview } from './preview.js';
import { renderTree } from './renderer.js';

// ── Leaf param editor ─────────────────────────────────────────────────────────
export function buildLeafEditor(node) {
  const controls = document.getElementById('leaf-controls');
  controls.innerHTML = '';

  const p     = node.params || {};
  const specs = (state.funcParamsData || {})[node.func];

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
    if (document.getElementById('auto-update-chk').checked) onUpdateLeaf();
  });
}

export function collectLeafParams(func) {
  const params = {};
  if (func === 'rand_color') {
    params.color = hexToColorParam(document.getElementById('lp-color').value);
    return params;
  }
  const specs = (state.funcParamsData || {})[func] || [];
  specs.forEach(spec => {
    if (spec.type === 'float') {
      params[spec.name] = parseFloat(document.getElementById(`lp-${spec.name}`).value);
    } else if (spec.type === 'color') {
      params[spec.name] = hexToColorParam(document.getElementById(`lp-${spec.name}`).value);
    }
  });
  return params;
}

export async function onUpdateLeaf() {
  if (!state.selectedId) return;
  const node = _findNode(state.nodes, state.selectedId);
  if (!node || node.arity !== 0) return;

  const params = collectLeafParams(node.func);
  const res  = await fetch('/api/leaf/set-params', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({tree_id: state.treeId, node_id: state.selectedId, params}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  state.nodes = data.nodes;
  if (data.root_id) state.rootId = data.root_id;
  setUndoEnabled(true);
  const updated = _findNode(state.nodes, state.selectedId);
  if (updated) {
    selectNodeCallback(updated);
    scheduleNodePreview(state.selectedId);
  }
  afterLeafUpdate();
}

// ── Non-leaf param editor ─────────────────────────────────────────────────────
export function buildNodeParamEditor(node) {
  const controls = document.getElementById('node-params-controls');
  controls.innerHTML = '';
  const specs = (state.funcParamsData || {})[node.func] || [];
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
    if (document.getElementById('auto-update-chk').checked) onUpdateNodeParams();
  });
}

export function collectNodeParams(funcName) {
  const specs = (state.funcParamsData || {})[funcName] || [];
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

export async function onUpdateNodeParams() {
  if (!state.selectedId) return;
  const node = _findNode(state.nodes, state.selectedId);
  if (!node || node.arity === 0) return;

  const params = collectNodeParams(node.func);
  const res  = await fetch('/api/node/set-params', {
    method: 'POST', headers: jsonHdr(),
    body: JSON.stringify({tree_id: state.treeId, node_id: state.selectedId, params}),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  state.nodes = data.nodes;
  if (data.root_id) state.rootId = data.root_id;
  const updated = _findNode(state.nodes, state.selectedId);
  if (updated) {
    selectNodeCallback(updated);
    scheduleNodePreview(state.selectedId);
  }
}

// ── UI component builders ─────────────────────────────────────────────────────
export function makeSelectRow(label, id, choices, value) {
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

export function makeSliderRow(label, id, min, max, step, value) {
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

export function makeColorRow(label, id, colorParam) {
  const wrap  = document.createElement('label');
  wrap.textContent = label;
  const input = document.createElement('input');
  input.type = 'color'; input.id = id;
  input.value = colorParamToHex(colorParam);
  input.style.cssText = 'width:100%;height:32px;cursor:pointer';
  wrap.appendChild(input);
  return wrap;
}

// ── Callbacks injected by app.js to avoid circular imports ────────────────────
let _findNode            = () => null;
let selectNodeCallback   = () => {};
let setUndoEnabled       = () => {};
let afterLeafUpdate      = () => {};

export function setEditorCallbacks(findNode, selectNode, setUndo, onAfterLeafUpdate) {
  _findNode          = findNode;
  selectNodeCallback = selectNode;
  setUndoEnabled     = setUndo;
  afterLeafUpdate    = onAfterLeafUpdate || (() => {});
}
