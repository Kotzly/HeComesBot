import { state, svg, gRoot, zoom } from './state.js';
import { symbolPath, fillFor, strokeFor } from './symbols.js';
import { scheduleNodePreview } from './preview.js';

// Imported lazily to avoid circular deps — set by app.js after init
let _selectNode    = null;
let _toggleCollapse = null;

export function setRendererCallbacks(selectNode, toggleCollapse) {
  _selectNode     = selectNode;
  _toggleCollapse = toggleCollapse;
}

export function renderTree() {
  const container = document.getElementById('tree-container');
  const W = container.clientWidth;
  const H = container.clientHeight;
  svg.attr('width', W).attr('height', H);
  gRoot.selectAll('*').remove();

  const visible = visibleSubtree(state.treeData);
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

  nodeG.append('path')
    .attr('d', d => symbolPath(d.data))
    .attr('fill', d => fillFor(d.data))
    .attr('stroke', d => strokeFor(d.data, d.data.id === state.selectedId).color)
    .attr('stroke-width', d => strokeFor(d.data, d.data.id === state.selectedId).width)
    .style('cursor', 'pointer')
    .on('click',    (e, d) => { e.stopPropagation(); _selectNode && _selectNode(d.data); })
    .on('dblclick', (e, d) => { e.stopPropagation(); _toggleCollapse && _toggleCollapse(d.data); });

  nodeG.filter(d => state.collapsedIds.has(d.data.id) && d.data.arity > 0)
    .append('text')
    .attr('dy', '0.35em').attr('text-anchor', 'middle')
    .attr('font-size', '9px').attr('fill', '#fff')
    .attr('pointer-events', 'none').text('…');

  const hasKids = d => d.data._kids.length > 0 && !state.collapsedIds.has(d.data.id);
  nodeG.append('text').attr('class', 'node-label')
    .attr('x', d => hasKids(d) ? -16 : 16)
    .attr('dy', '0.35em')
    .attr('text-anchor', d => hasKids(d) ? 'end' : 'start')
    .text(d => d.data.func);

  if (state.sensitivityData) {
    nodeG.filter(d => state.sensitivityData[d.data.id] !== undefined)
      .append('text')
      .attr('y', -14).attr('text-anchor', 'middle')
      .attr('font-size', '8px').attr('fill', '#f5a623')
      .attr('pointer-events', 'none')
      .text(d => `R:${state.sensitivityData[d.data.id].root.toFixed(3)}`);

    nodeG.filter(d => state.sensitivityData[d.data.id] !== undefined)
      .append('text')
      .attr('y', 20).attr('text-anchor', 'middle')
      .attr('font-size', '8px').attr('fill', '#7ec8e3')
      .attr('pointer-events', 'none')
      .text(d => `L:${state.sensitivityData[d.data.id].leaf.toFixed(3)}`);
  }
}

export function visibleSubtree(node) {
  const copy = {
    id: node.id, func: node.func, arity: node.arity,
    params: node.params, delta: node.delta, _kids: [],
  };
  if (node.children.length && !state.collapsedIds.has(node.id))
    copy._kids = node.children.map(visibleSubtree);
  return copy;
}

export function toggleCollapse(nodeData) {
  if (nodeData.arity === 0) return;
  state.collapsedIds.has(nodeData.id)
    ? state.collapsedIds.delete(nodeData.id)
    : state.collapsedIds.add(nodeData.id);
  renderTree();
}

export function fitTree() {
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

export function updateStats(node) {
  let total = 0, leaves = 0, depth = 0;
  (function walk(n, d) {
    total++; if (n.arity === 0) leaves++; if (d > depth) depth = d;
    (n.children || []).forEach(c => walk(c, d + 1));
  })(node, 0);
  document.getElementById('tree-stats').innerHTML =
    `<small>${total} nodes · ${leaves} leaves · depth ${depth}</small>`;
}

export function buildLegend() {
  const entries = [
    {label: 'rand_color', node: {arity:0, func:'rand_color', params:{color:[0.75,0.35,0.6]}}},
    {label: 'circle',     node: {arity:0, func:'circle',     params:{cx:0,cy:0,rx:0.6,ry:0.6,color:[0.3,0.7,0.4]}}},
    {label: 'cone',       node: {arity:0, func:'cone',       params:{cx:0.8,cy:0.8,rx:0.4,ry:0.4,color:[0.8,0.5,0.2]}}},
    {label: 'sphere',     node: {arity:0, func:'sphere',     params:{cx:0,cy:0,rx:0.6,ry:0.6,color:[0.4,0.6,0.9]}}},
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
