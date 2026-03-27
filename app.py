import uuid
import numpy as np
import io
import base64
import os
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

from functions import BUILD_FUNCTIONS
from build import get_random_function
from config import load_personality_list
from video import random_delta

app = Flask(__name__, static_folder='static', static_url_path='')

# In-memory state: tree_id -> session dict
_sessions = {}

# Function lookup tables (built once at startup)
FUNC_BY_NAME = {f.__name__: (n, f) for n, f in BUILD_FUNCTIONS}
FUNCS_BY_ARITY = {}
for _n, _f in BUILD_FUNCTIONS:
    FUNCS_BY_ARITY.setdefault(_n, []).append(_f.__name__)
for _arity in FUNCS_BY_ARITY:
    FUNCS_BY_ARITY[_arity].sort()


def _new_id():
    return str(uuid.uuid4())[:8]


def _build_rich(depth, min_depth, max_depth, dx, dy, weights, alpha, leaves):
    n_args, func = get_random_function(depth, p=weights, min_depth=min_depth, max_depth=max_depth)
    nid = _new_id()
    if n_args == 0:
        base = func(dx=dx, dy=dy).astype(np.float32)
        delta = np.float32(random_delta(alpha))
        leaves[nid] = {'base': base, 'delta': delta}
        return {'id': nid, 'func': func.__name__, 'arity': 0, 'children': []}
    children = [
        _build_rich(depth + 1, min_depth, max_depth, dx, dy, weights, alpha, leaves)
        for _ in range(n_args)
    ]
    return {'id': nid, 'func': func.__name__, 'arity': n_args, 'children': children}


def _eval_rich(node, steps, leaves):
    if node['arity'] == 0:
        leaf = leaves[node['id']]
        return leaf['base'] + leaf['delta'] * steps
    _, func = FUNC_BY_NAME[node['func']]
    args = [_eval_rich(c, steps, leaves) for c in node['children']]
    return func(*args)


def _collect_leaf_ids(node):
    """Return IDs of all leaf (arity=0) nodes in the subtree."""
    if node['arity'] == 0:
        return [node['id']]
    ids = []
    for c in node['children']:
        ids.extend(_collect_leaf_ids(c))
    return ids


def _find_node(tree, node_id):
    if tree['id'] == node_id:
        return tree
    for child in tree.get('children', []):
        found = _find_node(child, node_id)
        if found is not None:
            return found
    return None


def _find_parent(tree, node_id):
    """Return (parent_dict, child_index) or None if node_id is the root."""
    for i, child in enumerate(tree.get('children', [])):
        if child['id'] == node_id:
            return tree, i
        result = _find_parent(child, node_id)
        if result is not None:
            return result
    return None


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/functions')
def get_functions():
    return jsonify({str(k): v for k, v in FUNCS_BY_ARITY.items()})


@app.route('/api/personalities')
def get_personalities():
    files = sorted(f for f in os.listdir('.') if f.endswith('.json') and 'personality' in f)
    return jsonify(files)


@app.route('/api/build', methods=['POST'])
def build():
    data = request.json
    seed = int(data.get('seed', 42))
    dx = int(data.get('width', 256))
    dy = int(data.get('height', 256))
    min_depth = int(data.get('min_depth', 6))
    max_depth = int(data.get('max_depth', 16))
    personality_path = data.get('personality', 'personality.json')
    alpha = float(data.get('alpha', 4e-3))

    weights = load_personality_list(personality_path)
    np.random.seed(seed % (2**32 - 1))
    leaves = {}
    tree = _build_rich(0, min_depth, max_depth, dx, dy, weights, alpha, leaves)

    tree_id = _new_id()
    _sessions[tree_id] = {
        'tree': tree,
        'leaves': leaves,
        'meta': {'dx': dx, 'dy': dy, 'seed': seed, 'min_depth': min_depth,
                 'max_depth': max_depth, 'alpha': alpha},
    }
    return jsonify({'tree_id': tree_id, 'tree': tree})


@app.route('/api/preview', methods=['POST'])
def preview():
    data = request.json
    tree_id = data['tree_id']
    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({'error': 'unknown tree_id'}), 404

    dx, dy = session['meta']['dx'], session['meta']['dy']
    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)

    raw = _eval_rich(session['tree'], steps, session['leaves'])
    frame = np.broadcast_to(raw[0].clip(0, 1), (dy, dx, raw.shape[-1])).copy()
    if frame.shape[-1] != 3:
        frame = frame[..., :3]

    img_8 = np.rint(frame * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_8).save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({'image': f'data:image/png;base64,{b64}'})


@app.route('/api/node/set-func', methods=['POST'])
def set_func():
    data = request.json
    tree_id = data['tree_id']
    node_id = data['node_id']
    func_name = data['func_name']

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({'error': 'unknown tree_id'}), 404
    if func_name not in FUNC_BY_NAME:
        return jsonify({'error': f'unknown function: {func_name}'}), 400

    new_arity, new_func = FUNC_BY_NAME[func_name]
    node = _find_node(session['tree'], node_id)
    if node is None:
        return jsonify({'error': 'unknown node_id'}), 404

    old_arity = node['arity']
    meta = session['meta']

    if new_arity == old_arity:
        node['func'] = func_name
        return jsonify({'tree': session['tree']})

    # Arity is changing — remove old leaf data
    for lid in _collect_leaf_ids(node):
        session['leaves'].pop(lid, None)

    node['func'] = func_name
    node['arity'] = new_arity

    if new_arity == 0:
        base = new_func(dx=meta['dx'], dy=meta['dy']).astype(np.float32)
        delta = np.float32(random_delta(meta['alpha']))
        session['leaves'][node['id']] = {'base': base, 'delta': delta}
        node['children'] = []
    else:
        weights = load_personality_list('personality.json')
        child_max = max(3, meta['max_depth'] // 2)
        child_min = max(1, meta['min_depth'] // 2)
        new_children = []
        for _ in range(new_arity):
            np.random.seed(np.random.randint(0, 2**31))
            child = _build_rich(0, child_min, child_max,
                                meta['dx'], meta['dy'], weights, meta['alpha'],
                                session['leaves'])
            new_children.append(child)
        node['children'] = new_children

    return jsonify({'tree': session['tree']})


@app.route('/api/node/regenerate', methods=['POST'])
def regenerate():
    data = request.json
    tree_id = data['tree_id']
    node_id = data['node_id']
    seed = int(data.get('seed', np.random.randint(0, 2**31)))

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({'error': 'unknown tree_id'}), 404

    node = _find_node(session['tree'], node_id)
    if node is None:
        return jsonify({'error': 'unknown node_id'}), 404

    meta = session['meta']
    for lid in _collect_leaf_ids(node):
        session['leaves'].pop(lid, None)

    weights = load_personality_list('personality.json')
    np.random.seed(seed % (2**32 - 1))
    new_subtree = _build_rich(0, meta['min_depth'], meta['max_depth'],
                              meta['dx'], meta['dy'], weights, meta['alpha'],
                              session['leaves'])

    if session['tree']['id'] == node_id:
        session['tree'] = new_subtree
    else:
        parent, idx = _find_parent(session['tree'], node_id)
        parent['children'][idx] = new_subtree

    return jsonify({'tree': session['tree'], 'new_node_id': new_subtree['id']})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
