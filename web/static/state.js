// ── Shared mutable state ──────────────────────────────────────────────────────
export const state = {
  treeId:          null,
  treeData:        null,
  selectedId:      null,
  referenceId:     null,
  collapsedIds:    new Set(),
  functionsData:   null,
  funcParamsData:  null,
  sensitivityData: null,
};

// ── D3 setup ──────────────────────────────────────────────────────────────────
export const svg   = d3.select('#tree-svg');
export const gRoot = svg.append('g').attr('class', 'root-g');
export const zoom  = d3.zoom()
  .scaleExtent([0.05, 4])
  .clickDistance(8)
  .on('zoom', e => gRoot.attr('transform', e.transform));
svg.call(zoom);

export const SYM_SIZE = 280;
