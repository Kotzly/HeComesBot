# Audio Generation via the Tree/Leaf Algorithm

## 1. Fundamental differences: what changes when the output is audio

### The tensor shape problem

The entire current system is built around one abstraction: every node produces a `(H, W, C)` float32 array. Leaves generate spatial fields, operators blend/warp/transform them. This is clean and uniform — every function has the same input/output contract.

For audio, the natural representation is `(T, C)` — time samples by channels (mono = 1, stereo = 2). That's not just a different shape; it's a different *semantics*. Spatial primitives (`circle`, `x_var`, `cone`, etc.) work by computing a function over a 2D coordinate grid. The audio analog would be computing a function over a 1D time axis. Most spatial primitives don't generalize — a circle has no meaningful 1D analog.

### What "leaf" means changes completely

Current leaves are visual primitives: gradients, circles, spheres, color fields. They model *spatial structure* with *color*. Audio leaves would be:
- Oscillators: `sin(2πft + φ)`, sawtooth, square, triangle
- Noise sources: white, pink (1/f), Brownian
- Envelopes: ADSR shapes over time
- Sampled waveforms

These are categorically different from spatial fields. Some operators might carry over (addition, multiplication, blending are fine), but warping operators like `swirl`, `ripple`, `polar_warp` are purely spatial and have no time-domain equivalent.

### The frequency domain wrinkle

A huge part of what makes audio interesting isn't in the time domain at all — it's spectral. Two sine waves at slightly different frequencies produce beating. A single operator over time samples can produce rich harmonic structure or silence. The tree algorithm operating on raw samples could theoretically produce interesting audio, but most randomly sampled trees would produce either silence, clipping noise, or a single tone. The perceptual "interesting" region is much narrower in audio than in visual art, where almost any smooth combination of gradients and shapes looks at least interesting.

This is a real challenge: the current algorithm for images is nearly guaranteed to produce *visually interesting* output because even random combinations of spatial fields produce plausible images. Audio doesn't have this property — random combinations of waveforms usually produce noise or near-silence.

### Temporal structure vs spatial structure

Images are *atemporal* — every pixel is independent in the sense that order doesn't matter. Video adds a slow time axis but the tree still generates each frame from a spatial field. Audio's *entire* information is temporal. This means:
- The "chunk" metaphor still works (audio chunks ≈ video chunks), but each chunk is `(T, C)` not `(H, W, C)`
- There's no analog to "width" and "height" parameters — you have sample rate and duration
- The coordinate grid (`linear_mesh`) is replaced by a 1D time vector

### Perceptual expectations

Procedurally generated images have relatively loose perceptual constraints — arbitrary colors and shapes can look "good." Audio has much stricter constraints: humans are extremely sensitive to clipping, discontinuities between chunks, spectral imbalances, and phase issues. The chunk boundary problem (visual equivalent: a seam between rendered chunks) becomes a serious issue in audio because a phase discontinuity is instantly audible. Crossfading chunks or ensuring phase coherence across chunk boundaries requires explicit design.

---

## 2. Adapting the current code vs a separate backend: a critical analysis

### What can be reused

The *structure* of the algorithm — a tree of random depth, leaves with random params, operators combining children — is medium-agnostic. Specifically:
- `Node`, `FunctionDef`, `linearize`, `compile_plan` — pure structure, reusable
- `eval_plan` loop — reusable if the node contract is `(*inputs) → array` regardless of shape
- The personality JSON system and weight sampling — fully reusable
- The path animation system (`tree_paths.py`, `ODEPath`, etc.) — conceptually reusable; leaf params animating over time maps naturally to audio parameter modulation (FM synthesis, LFO)
- The CLI scaffolding (`_video_utils.py`, chunked processing, multiprocessing) — reusable with shape changes

### What cannot be reused

- All primitives in `primitives.py` — `circle`, `sphere`, `ripple`, `swirl`, etc. are inherently 2D spatial
- All spatial operators that depend on 2D geometry — `kaleidoscope`, `polar_warp`, `hsv_warp`
- The coordinate grid system (`linear_mesh`, `random_point`, `random_radius`) — needs a 1D time vector replacement
- Color channels — audio has amplitude channels; the 3-channel assumption is baked into many primitives and `_compute_chunk`

### The core tension: generalization vs separation

**Option A: Generalize the existing backend**

You'd need to parameterize every function on "media type" and swap out the leaf/operator registries. The `FunctionDef` dataclass already has `arity` and `params` — you'd add a `domain` field (`"spatial"`, `"temporal"`). The `eval_plan` loop is already generic enough. The `compile_plan` would dispatch to the right leaf registry.

The problems: the code would be full of `if domain == "spatial"` branches. The current `(H, W, C)` assumption is pervasive — in how leaves take `dx, dy`, in how `_compute_chunk` reshapes steps, in how GPU support works. Scrubbing all of this for dual-mode support is a large, invasive refactor with real risk of breaking the working visual backend.

**Option B: A separate audio backend**

Mirror the structure of `artgen/`: create `artgen_audio/` with its own `primitives_audio.py`, `functions_audio.py`, `tree_audio.py`. The tree builder and eval loop can be copied and adapted (they're ~100 lines each). This is more code total but:
- The visual backend is never touched
- Audio-specific constraints (chunk boundary handling, normalization, clipping) are isolated
- You can evolve the audio backend independently

This is the right call. The current codebase has accumulated significant refinement in the visual backend. A clean separation protects that investment.

### Verdict

| | Adapting existing code | Separate audio backend |
|---|---|---|
| Reused code | ~20% (tree structure, node registry, eval loop) | ~20% (same, but explicit) |
| Risk to visual backend | High | None |
| Implementation effort | High (invasive refactoring) | Medium (fresh code, clear contracts) |
| Code clarity | Degrades (spatial/temporal if-branches everywhere) | Stays clean |
| Path animation reuse | Partial (requires abstraction) | Full (just import ODEPath) |

**Recommendation: separate `artgen_audio/` backend.** The visual and audio domains share the *algorithm shape* (random tree, leaves, operators, eval loop) but have almost no shared *implementation*. The right abstraction isn't "generalize everything" — it's "reuse the pattern, not the code." The path animation system is the one genuine gem worth wiring up to both backends from day one.

---

## 3. What the architecture could look like

```
src/hecomes/
├── artgen/                 ← existing, untouched
│   ├── primitives.py       (spatial leaves)
│   ├── functions.py        (spatial registry)
│   ├── tree.py             (generic structure)
│   └── tree_paths.py       (path animation)
│
├── artgen_audio/           ← new
│   ├── oscillators.py      (audio leaves: sin, saw, noise, ADSR envelopes)
│   ├── operators.py        (audio ops: mix, ring_mod, waveshape, filter)
│   ├── functions_audio.py  (audio registry — same FunctionDef dataclass)
│   └── tree_audio.py       (thin adapter over tree.py eval loop)
│
├── cli/
│   ├── audio.py            ← new entry point: hecomes-audio
│   └── _video_utils.py
│
└── config.py               (PERSONALITIES_DIR already works)
```

The key design insight: `tree.py`'s `build_node`, `compile_plan`, `eval_plan` take a `weights` dict and a registry. If you pass in an audio registry instead of the spatial one, the same tree algorithm generates an audio tree. You wouldn't copy `tree.py` at all — `tree_audio.py` would just be a thin adapter:

```python
from hecomes.artgen.tree import build_node, compile_plan, linearize

def build_audio_node(depth, min_d, max_d, n_samples, weights, nodes, leaves):
    # same logic, just n_samples instead of dx/dy
    return build_node(depth, min_d, max_d, n_samples, 1, weights, ...)

def eval_audio_plan(plan, t_chunk):
    # same loop, output shape is (T, C)
    return eval_plan(plan, t_chunk)
```

### Audio primitives

```python
def sine(n_samples=None, freq=440.0, phase=0.0, amplitude=1.0):
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples)
    return (amplitude * np.sin(2 * np.pi * freq * t + phase)).reshape(n_samples, 1)

def noise_white(n_samples=None, amplitude=1.0):
    return (amplitude * np.random.randn(n_samples)).reshape(n_samples, 1)

def adsr_envelope(n_samples=None, attack=0.1, decay=0.1, sustain=0.7, release=0.2):
    # returns a (n_samples, 1) amplitude envelope
    ...
```

### Audio operators

```python
def mix(a, b, ratio=0.5):         # linear blend — identical semantics to visual blend
    return (1 - ratio) * a + ratio * b

def ring_mod(a, b):                # amplitude modulation — (a * b), no visual analog
    return a * b

def waveshape(x, drive=1.0):       # soft clip / distortion — 1D analog of sigmoid
    return np.tanh(drive * x)

def filter_lowpass(x, cutoff=1000.0):  # requires scipy.signal — no visual analog
    ...
```

### Chunk boundary handling

The one genuinely new engineering problem: audio output must be phase-continuous across chunks. The path system already solves this for visual parameters (params are continuous functions of time). For audio, you'd need to carry phase state across chunks, which the path animation system (`ODEPath`) could handle naturally — an oscillator whose phase is an ODE state `dφ/dt = 2πf` maintains continuity automatically.
