import { state } from './state.js';
import { jsonHdr } from './utils.js';

let _previewTimer = null;

export function scheduleNodePreview(nodeId) {
  clearTimeout(_previewTimer);
  _previewTimer = setTimeout(() => fetchNodePreview(nodeId), 120);
}

export async function fetchNodePreview(nodeId) {
  if (!state.treeId || nodeId !== state.selectedId) return;
  const previewNodeId = state.referenceId || nodeId;
  const loading = document.getElementById('node-preview-loading');
  const img     = document.getElementById('node-preview-img');
  loading.style.display = '';
  img.style.display = 'none';
  try {
    const res  = await fetch('/api/node/preview', {
      method: 'POST', headers: jsonHdr(),
      body: JSON.stringify({tree_id: state.treeId, node_id: previewNodeId}),
    });
    const data = await res.json();
    if (data.error || nodeId !== state.selectedId) return;
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

export function openModal()  { document.getElementById('preview-modal').classList.remove('hidden'); }
export function closeModal() { document.getElementById('preview-modal').classList.add('hidden'); }
