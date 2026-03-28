import uuid
import pickle
import numpy as np
import io
import base64
import os
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

from functions import BUILD_FUNCTIONS, linear_mesh, hsv_to_rgb, generate_params, FUNC_PARAMS
from build import get_random_function
from config import load_personality_list
from video import random_delta

app = Flask(__name__, static_folder='static', static_url_path='')

_sessions = {}
SAVE_DIR = 'saved_trees'
COLOR_SPACES = ('rgb', 'hsv', 'cmy')


def _render_frame(raw_batch, color_space, dx, dy):
    """Convert a raw eval result (N, ?, ?, ?) to a displayable (dy, dx, 3) uint8 array."""
    frame = np.broadcast_to(raw_batch[0], (dy, dx, raw_batch.shape[-1])).copy()
    if color_space == 'hsv':
        hsv = np.stack([
            frame[..., 0] % 1.0,
            frame[..., 1].clip(0, 1),
            frame[..., 2].clip(0, 1),
        ], axis=-1)
        frame = hsv_to_rgb(hsv)
    elif color_space == 'cmy':
        frame = (1.0 - frame.clip(0, 1))
    else:
        frame = frame.clip(0, 1)
    if frame.shape[-1] != 3:
        frame = frame[..., :3]
    return np.rint(frame * 255).astype(np.uint8)

FUNC_BY_NAME = {f.__name__: (n, f) for n, f in BUILD_FUNCTIONS}
FUNCS_BY_ARITY = {}
for _n, _f in BUILD_FUNCTIONS:
    FUNCS_BY_ARITY.setdefault(_n, []).append(_f.__name__)
for _arity in FUNCS_BY_ARITY:
    FUNCS_BY_ARITY[_arity].sort()


# ── Leaf geometry helpers ─────────────────────────────────────────────────────

def _random_point():
    return (1 - np.random.rand(2) ** 2) * 4 - 2


def _random_radius():
    return np.maximum(1 - np.random.rand(2) ** 2, 0.01)


def _cone_array(cx, cy, rx, ry, dx, dy):
    x, y = linear_mesh(dx=dx, dy=dy)
    gradient = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2).reshape(dy, dx, 1).astype(np.float32)
    return np.broadcast_to(gradient, (dy, dx, 3)).copy()


def _circle_array(cx, cy, rx, ry, color, dx, dy):
    base = _cone_array(cx, cy, rx, ry, dx, dy)  # (dy, dx, 3), all channels equal
    circ = np.ones((dy, dx, 3), dtype=np.float32) * np.array(color, dtype=np.float32).reshape(1, 1, 3)
    circ[base > 1] = 0
    return circ


def _build_leaf(func, dx, dy, alpha):
    """Build a leaf array and capture editable params where possible."""
    fname = func.__name__
    if fname == 'rand_color':
        color = np.random.rand(3).astype(np.float32)
        base = np.broadcast_to(color.reshape(1, 1, 3), (dy, dx, 3)).copy()
        params = {'color': color.tolist()}
    elif fname == 'cone':
        cx, cy = _random_point()
        rx, ry = _random_radius()
        base = _cone_array(cx, cy, rx, ry, dx, dy)
        params = {'cx': float(cx), 'cy': float(cy), 'rx': float(rx), 'ry': float(ry)}
    elif fname == 'circle':
        cx, cy = _random_point()
        rx, ry = _random_radius()
        color = np.random.rand(3).astype(np.float32)
        base = _circle_array(cx, cy, rx, ry, color, dx, dy)
        params = {'cx': float(cx), 'cy': float(cy), 'rx': float(rx), 'ry': float(ry),
                  'color': color.tolist()}
    else:
        params = generate_params(func.__name__)
        base = func(dx=dx, dy=dy, **params).astype(np.float32)
    delta = np.float32(random_delta(alpha))
    return base, delta, params


def _recompute_leaf(func_name, params, dx, dy):
    """Recompute leaf base array from stored params."""
    if func_name == 'rand_color':
        return np.broadcast_to(
            np.array(params['color'], dtype=np.float32).reshape(1, 1, 3), (dy, dx, 3)
        ).copy()
    if func_name == 'cone':
        return _cone_array(params['cx'], params['cy'], params['rx'], params['ry'], dx, dy)
    if func_name == 'circle':
        return _circle_array(params['cx'], params['cy'], params['rx'], params['ry'],
                             params['color'], dx, dy)
    if func_name in ('x_var', 'y_var'):
        _, f = FUNC_BY_NAME[func_name]
        return f(dx=dx, dy=dy, angle=params['angle']).astype(np.float32)
    return None


# ── Tree helpers ──────────────────────────────────────────────────────────────

def _new_id():
    return str(uuid.uuid4())[:8]


def _build_rich(depth, min_depth, max_depth, dx, dy, weights, alpha, leaves):
    n_args, func = get_random_function(depth, p=weights, min_depth=min_depth, max_depth=max_depth)
    nid = _new_id()
    if n_args == 0:
        base, delta, params = _build_leaf(func, dx, dy, alpha)
        leaves[nid] = {'base': base, 'delta': delta, 'func': func.__name__, 'params': params}
        return {'id': nid, 'func': func.__name__, 'arity': 0, 'children': [],
                'delta': float(delta), 'params': params}
    params = generate_params(func.__name__)
    children = [
        _build_rich(depth + 1, min_depth, max_depth, dx, dy, weights, alpha, leaves)
        for _ in range(n_args)
    ]
    return {'id': nid, 'func': func.__name__, 'arity': n_args, 'children': children, 'params': params}


def _eval_rich(node, steps, leaves):
    if node['arity'] == 0:
        leaf = leaves[node['id']]
        return leaf['base'] + leaf['delta'] * steps
    _, func = FUNC_BY_NAME[node['func']]
    args = [_eval_rich(c, steps, leaves) for c in node['children']]
    return func(*args, **node.get('params', {}))


def _collect_leaf_ids(node):
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
    for i, child in enumerate(tree.get('children', [])):
        if child['id'] == node_id:
            return tree, i
        result = _find_parent(child, node_id)
        if result is not None:
            return result
    return None


def _node_depth(tree, node_id, d=0):
    if tree['id'] == node_id:
        return d
    for child in tree.get('children', []):
        result = _node_depth(child, node_id, d + 1)
        if result is not None:
            return result
    return None


# ── Routes ────────────────────────────────────────────────────────────────────

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


@app.route('/api/trees')
def list_trees():
    os.makedirs(SAVE_DIR, exist_ok=True)
    files = sorted(f[:-4] for f in os.listdir(SAVE_DIR) if f.endswith('.pkl'))
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
    color_space = data.get('color_space', 'rgb')
    if color_space not in COLOR_SPACES:
        color_space = 'rgb'

    weights = load_personality_list(personality_path)
    np.random.seed(seed % (2**32 - 1))
    leaves = {}
    tree = _build_rich(0, min_depth, max_depth, dx, dy, weights, alpha, leaves)

    tree_id = _new_id()
    _sessions[tree_id] = {
        'tree': tree, 'leaves': leaves,
        'meta': {'dx': dx, 'dy': dy, 'seed': seed, 'min_depth': min_depth,
                 'max_depth': max_depth, 'alpha': alpha, 'color_space': color_space},
    }
    return jsonify({'tree_id': tree_id, 'tree': tree, 'meta': _sessions[tree_id]['meta']})


@app.route('/api/preview', methods=['POST'])
def preview():
    data = request.json
    tree_id = data['tree_id']
    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({'error': 'unknown tree_id'}), 404

    dx, dy = session['meta']['dx'], session['meta']['dy']
    color_space = session['meta'].get('color_space', 'rgb')
    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    raw = _eval_rich(session['tree'], steps, session['leaves'])
    img_8 = _render_frame(raw, color_space, dx, dy)
    buf = io.BytesIO()
    Image.fromarray(img_8).save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({'image': f'data:image/png;base64,{b64}'})


@app.route('/api/save', methods=['POST'])
def save_tree():
    data = request.json
    tree_id = data['tree_id']
    name = data.get('name', tree_id).strip()
    name = ''.join(c for c in name if c.isalnum() or c in '._- ')
    if not name:
        return jsonify({'error': 'invalid name'}), 400

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({'error': 'unknown tree_id'}), 404

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(session, f)
    return jsonify({'name': name})


@app.route('/api/load', methods=['POST'])
def load_tree():
    data = request.json
    name = os.path.basename(data.get('name', ''))
    path = os.path.join(SAVE_DIR, f'{name}.pkl')
    if not os.path.exists(path):
        return jsonify({'error': 'file not found'}), 404

    with open(path, 'rb') as f:
        session = pickle.load(f)

    tree_id = _new_id()
    _sessions[tree_id] = session
    return jsonify({'tree_id': tree_id, 'tree': session['tree'], 'meta': session['meta']})


@app.route('/api/node/preview', methods=['POST'])
def node_preview():
    data = request.json
    tree_id = data['tree_id']
    node_id = data['node_id']

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({'error': 'unknown tree_id'}), 404

    node = _find_node(session['tree'], node_id)
    if node is None:
        return jsonify({'error': 'unknown node_id'}), 404

    dx, dy = session['meta']['dx'], session['meta']['dy']
    color_space = session['meta'].get('color_space', 'rgb')
    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    raw = _eval_rich(node, steps, session['leaves'])
    img_8 = _render_frame(raw, color_space, dx, dy)
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
        # If becoming a leaf with same arity (0→0), regenerate leaf data
        if new_arity == 0:
            for lid in _collect_leaf_ids(node):
                session['leaves'].pop(lid, None)
            base, delta, params = _build_leaf(new_func, meta['dx'], meta['dy'], meta['alpha'])
            session['leaves'][node_id] = {'base': base, 'delta': delta,
                                          'func': func_name, 'params': params}
            node.update({'delta': float(delta), 'params': params})
        else:
            node['params'] = generate_params(func_name)
        return jsonify({'tree': session['tree']})

    # Arity is changing
    for lid in _collect_leaf_ids(node):
        session['leaves'].pop(lid, None)

    node['func'] = func_name
    node['arity'] = new_arity

    if new_arity == 0:
        base, delta, params = _build_leaf(new_func, meta['dx'], meta['dy'], meta['alpha'])
        session['leaves'][node_id] = {'base': base, 'delta': delta,
                                      'func': func_name, 'params': params}
        node.update({'children': [], 'delta': float(delta), 'params': params})
    else:
        nd = _node_depth(session['tree'], node_id) or 0
        child_depth = nd + 1
        eff_max = max(meta['max_depth'] - child_depth, 2)
        eff_min = max(min(meta['min_depth'] - child_depth, eff_max - 1), 1)
        weights = load_personality_list('personality.json')
        new_children = []
        for _ in range(new_arity):
            np.random.seed(np.random.randint(0, 2**31))
            child = _build_rich(0, eff_min, eff_max,
                                meta['dx'], meta['dy'], weights, meta['alpha'],
                                session['leaves'])
            new_children.append(child)
        node['children'] = new_children
        node['params'] = generate_params(func_name)
        node.pop('delta', None)

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

    nd = _node_depth(session['tree'], node_id) or 0
    eff_max = max(meta['max_depth'] - nd, 2)
    eff_min = max(min(meta['min_depth'] - nd, eff_max - 1), 1)

    weights = load_personality_list('personality.json')
    np.random.seed(seed % (2**32 - 1))
    new_subtree = _build_rich(0, eff_min, eff_max,
                              meta['dx'], meta['dy'], weights, meta['alpha'],
                              session['leaves'])

    if session['tree']['id'] == node_id:
        session['tree'] = new_subtree
    else:
        parent, idx = _find_parent(session['tree'], node_id)
        parent['children'][idx] = new_subtree

    return jsonify({'tree': session['tree'], 'new_node_id': new_subtree['id']})


@app.route('/api/leaf/set-params', methods=['POST'])
def set_leaf_params():
    data = request.json
    tree_id = data['tree_id']
    node_id = data['node_id']
    new_params = data.get('params', {})
    new_delta = data.get('delta', None)

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({'error': 'unknown tree_id'}), 404

    node = _find_node(session['tree'], node_id)
    if node is None or node['arity'] != 0:
        return jsonify({'error': 'node not found or not a leaf'}), 404

    leaf = session['leaves'].get(node_id)
    if leaf is None:
        return jsonify({'error': 'leaf data missing'}), 500

    meta = session['meta']

    if new_params:
        merged = {**leaf.get('params', {}), **new_params}
        new_base = _recompute_leaf(node['func'], merged, meta['dx'], meta['dy'])
        if new_base is not None:
            leaf['base'] = new_base
            leaf['params'] = merged
            node['params'] = merged

    if new_delta is not None:
        leaf['delta'] = np.float32(new_delta)
        node['delta'] = float(new_delta)

    return jsonify({'tree': session['tree']})


@app.route('/api/function-params')
def get_function_params():
    return jsonify(FUNC_PARAMS)


@app.route('/api/node/set-params', methods=['POST'])
def set_node_params():
    data = request.json
    tree_id = data['tree_id']
    node_id = data['node_id']
    new_params = data.get('params', {})

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({'error': 'unknown tree_id'}), 404

    node = _find_node(session['tree'], node_id)
    if node is None or node['arity'] == 0:
        return jsonify({'error': 'node not found or is a leaf'}), 404

    node['params'] = {**node.get('params', {}), **new_params}
    return jsonify({'tree': session['tree']})


if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")
