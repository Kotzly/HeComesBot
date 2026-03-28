# HeComesBot

Source code for the Facebook bot HeComes Bot 666.

![Example of generated image](./example_image.png)

Generates abstract images and videos by building random expression trees over pixel coordinates. The bot was inspired by an article on generating images with Python.

---

## Installation

Requires [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda create -n hecomesbot python=3.11
conda activate hecomesbot
pip install -e .
```

For bot functionality (Markov text generation):

```bash
pip install -e ".[bot]"
```

For video generation, [FFmpeg](https://www.ffmpeg.org/) must be installed and accessible in PATH.

---

## Usage

### Web UI

```bash
cd web
python app.py
```

Opens at `http://localhost:5000`. Build and explore trees interactively, preview images, edit nodes, and save/load sessions.

### Standalone image

```bash
hecomes-image [output.png]
```

Options:

| Flag | Description | Default |
|------|-------------|---------|
| `-s`, `--seed` | Random seed | random |
| `-W`, `--width` | Image width | 512 |
| `-H`, `--height` | Image height | 512 |
| `--min-depth` | Minimum tree depth | 6 |
| `--max-depth` | Maximum tree depth | 16 |
| `-c`, `--color-space` | Color space: `rgb`, `hsv`, `cmy` | `rgb` |
| `--personality` | Personality JSON filename in `data/` | `personality.json` |

### Video

```bash
hecomes-video [options]
```

Options:

| Flag | Description | Default |
|------|-------------|---------|
| `-n`, `--n_videos` | Number of videos | 1 |
| `-f`, `--fps` | Frames per second | 30 |
| `-W`, `--width` | Video width | 256 |
| `-H`, `--height` | Video height | 256 |
| `-d`, `--duration` | Duration in seconds | 10 |
| `-s`, `--step` | Alpha step between frames | 3e-3 |
| `-S`, `--seed` | Fixed seed (one video) | random |
| `-e`, `--extension` | Output format: `webm`, `mp4`, `avi`, `gif`, `flv`, `ogg`, `mpeg` | `webm` |
| `-b`, `--bitrate` | Constant bitrate | 6M |
| `-C`, `--codec` | Video codec | auto |
| `-p`, `--processes` | Parallel workers | 3 |
| `-c`, `--chunk_size` | Frames per batch | 10 |
| `--min-depth` | Minimum tree depth | 6 |
| `--max-depth` | Maximum tree depth | 16 |
| `--color-space` | Color space: `rgb`, `hsv`, `cmy` | `rgb` |
| `--independent-channels` | One tree per channel (H uses `personality_h.json`) | off |
| `--k` | Generate K channel from a tree (CMY only) | off |
| `--alpha` | Generate alpha channel from a tree | off |
| `--personality` | Personality JSON filename in `data/` | `personality.json` |

### Bot (legacy)

```bash
python -m hecomes.bot.main [options]
```

Options: `-s` seed, `-d` dimensions, `-o` output path, `-f` fontsize, `-D` use `config.json`.

---

## Color spaces

- **rgb** — default, channels clipped to [0, 1].
- **hsv** — H is circular (wraps), S/V are clipped. `--independent-channels` builds a separate tree per channel; H uses `personality_h.json` which enables circular hue functions.
- **cmy** — inverted RGB. `--k` adds a K (black) channel tree.

---

## Personality files

The personality files control the probability of each function being selected when building the expression tree. Each function is assigned a weight — higher means more likely, zero means never used.

- `personality.json` — used for RGB/CMY generation and for S/V channels in HSV mode.
- `personality_h.json` — used for the H channel in `--independent-channels` HSV mode. Enables circular hue functions.

The functions are grouped by arity:
- **Arity 0 (leaves):** produce a base image — `rand_color`, `x_var`, `y_var`, `circle`, `cone`.
- **Arity 1 (unary):** transform one image — `blur`, `sharpen`, `sigmoid`, `mirrored_sigmoid`, `absolute_value`, `color_rotate`, `kaleidoscope`.
- **Arity 2 (binary):** combine two images — `add`, `subtract`, `multiply`, `safe_divide`, `safe_modulus`, `saddle`, `swap_phase_amplitude`, `circular_mean`, `circular_mean_far`, `hue_diff`, `hue_rotate`.

Notable functions:

- **cone** — elliptical cone function; produces concentric circles/ellipses.
- **saddle** — hyperbolic paraboloid (`x² - y²`). Costly; use low probability.
- **swap_phase_amplitude** — swaps FFT phase/magnitude between two images. Very costly.
- **kaleidoscope** — not recommended above 400×400.
- **color_rotate** — rotation in color space; helps diversify away from pure R/G/B outputs.
- **circular_mean / circular_mean_far** — mean of two hue values along the shorter/longer arc.
- **hue_diff** — angular distance between two hue values, in [0, 0.5].
- **hue_rotate** — circular addition of hue values.
