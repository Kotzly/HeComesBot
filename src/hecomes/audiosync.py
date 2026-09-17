"""Video-synchronised audio: drive the tree synth with per-frame video features.

The timbre still comes from a random :mod:`hecomes.audiogen` tree (so every
seed sounds different); the video decides how it moves over time. Features
come from :class:`hecomes.video_features.FrameFeatureExtractor`:

==============  ==============================================================
feature         audio effect
==============  ==============================================================
``hue``         transposes every oscillator, snapped to a pentatonic scale
``saturation``  soft-clip drive (grey = clean, vivid = gritty)
``motion``      loudness envelope, plus percussive hits on sudden motion onsets
``brightness``  cutoff of a 12 dB/oct low-pass (dark = muffled, bright = open)
``pan``         stereo position (the sound follows where the image moves)
==============  ==============================================================

Each curve is rescaled against its own 5th-95th percentile range, with a
minimum span so near-static clips don't amplify pixel jitter into big swings.
"""

import numpy as np
from scipy.signal import lfilter

from hecomes.audiogen import SAMPLE_RATE, TWO_PI, normalize, synthesize, write_wav

# Some random trees cancel out to silence (or pure DC); rebuild with seed + 1.
MAX_TREE_ATTEMPTS = 8
SILENCE_RMS = 1e-4

# Hue → pitch. Degrees sit at hue 0 (red), 0.2, 0.4, 0.6, 0.8, so the red
# wrap-around is inside one degree rather than on a boundary.
PENTATONIC_SEMITONES = np.array([-4, -2, 0, 3, 5])
HUE_MIN_CONFIDENCE = 0.05   # saturation-weighted resultant below this holds the note
HUE_HYSTERESIS = 0.1        # in degrees; stops the note flickering on a boundary

SATURATION_MIN_RANGE = 0.1
DRIVE_MAX = 6.0

MOTION_MIN_RANGE = 0.005
GAIN_FLOOR = 0.25

ONSET_REL = 6.0             # a hit needs a rise this many times the median rise…
ONSET_MIN_RISE = 0.004      # …and at least this much absolute change per frame
HIT_MIN_GAP = 0.2           # seconds
HIT_FREQ = 220.0
HIT_LEVEL = 0.6

BRIGHTNESS_MIN_RANGE = 0.15
CUTOFF_MIN = 250.0
CUTOFF_MAX = 12_000.0
FILTER_BLOCK = 256

PAN_GAIN = 3.0
PAN_WIDTH = 0.8


# ── Control curves (one value per video frame) ───────────────────────────────


def _frames(fps, seconds):
    return max(1, int(round(fps * seconds)))


def _smooth(x, width):
    """Centered moving average over ``width`` frames, edges padded."""
    if width <= 1:
        return np.asarray(x, dtype=np.float64)
    left = width // 2
    padded = np.pad(x, (left, width - 1 - left), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _unit(x, min_range):
    """Map the 5th-95th percentile range of ``x`` onto [0, 1] (span >= ``min_range``)."""
    lo, hi = np.percentile(x, [5, 95])
    span = max(hi - lo, min_range)
    return np.clip((x - (lo + hi) / 2) / span + 0.5, 0.0, 1.0)


def _pitch_ratio(hue, saturation, fps):
    k = _frames(fps, 0.4)
    angle = TWO_PI * np.asarray(hue, dtype=np.float64)
    x = _smooth(saturation * np.cos(angle), k)
    y = _smooth(saturation * np.sin(angle), k)
    position = (np.arctan2(y, x) / TWO_PI % 1.0) * len(PENTATONIC_SEMITONES)
    confidence = np.hypot(x, y)

    n_degrees = len(PENTATONIC_SEMITONES)
    degrees = np.empty(len(position), dtype=np.int64)
    current = None
    for i, (pos, conf) in enumerate(zip(position, confidence)):
        if conf >= HUE_MIN_CONFIDENCE:
            offset = None if current is None else (pos - current + n_degrees / 2) % n_degrees - n_degrees / 2
            if offset is None or abs(offset) > 0.5 + HUE_HYSTERESIS:
                current = int(round(pos)) % n_degrees
        degrees[i] = n_degrees // 2 if current is None else current
    return 2.0 ** (PENTATONIC_SEMITONES[degrees] / 12.0)


def _drive(saturation, fps):
    s = _smooth(saturation, _frames(fps, 0.25))
    amount = 0.5 * s + 0.5 * _unit(s, SATURATION_MIN_RANGE)
    return 1.0 + (DRIVE_MAX - 1.0) * amount


def _gain(motion, fps):
    level = _unit(_smooth(motion, _frames(fps, 0.1)), MOTION_MIN_RANGE)
    return GAIN_FLOOR + (1.0 - GAIN_FLOOR) * level


def _onsets(motion, fps):
    """Return ``[(frame, strength in (0, 1]), ...]`` for sudden rises in motion."""
    rise = np.diff(motion, prepend=motion[:1])
    threshold = max(ONSET_REL * float(np.median(np.abs(rise))), ONSET_MIN_RISE)
    gap = _frames(fps, HIT_MIN_GAP)
    hits, last = [], -gap
    for i in range(1, len(rise) - 1):
        is_peak = rise[i] >= rise[i - 1] and rise[i] > rise[i + 1]
        if is_peak and rise[i] > threshold and i - last >= gap:
            hits.append((i, min(rise[i] / threshold, 3.0) / 3.0))
            last = i
    return hits


def _cutoff(brightness, fps):
    b = _smooth(brightness, _frames(fps, 0.1))
    openness = 0.5 * b + 0.5 * _unit(b, BRIGHTNESS_MIN_RANGE)
    return CUTOFF_MIN * (CUTOFF_MAX / CUTOFF_MIN) ** openness


def _pan_position(pan, motion, fps):
    """Motion-weighted average of the motion centroid, centred when nothing moves.

    Still frames don't pull the position to 0, but near-static clips (whose
    centroid is just pixel jitter) fade back to the centre.
    """
    k = _frames(fps, 0.3)
    weight = _smooth(motion, k)
    position = _smooth(pan * motion, k) / np.maximum(weight, 1e-9)
    presence = np.clip(weight / MOTION_MIN_RANGE, 0.0, 1.0)
    return np.clip(position * PAN_GAIN, -1.0, 1.0) * PAN_WIDTH * presence


# ── Audio-rate processing ────────────────────────────────────────────────────


def _dc_block(y):
    return lfilter([1.0, -1.0], [1.0, -0.9975], y)


def _hit(n, freq):
    t = np.arange(n) / SAMPLE_RATE
    body = np.sin(TWO_PI * freq * t) * np.exp(-t / 0.09)
    click = np.random.uniform(-1.0, 1.0, n) * np.exp(-t / 0.006)
    return 0.8 * body + 0.4 * click


def _add_hits(y, onsets, fps, pitch):
    length = int(0.5 * SAMPLE_RATE)
    for frame, strength in onsets:
        start = int(frame / fps * SAMPLE_RATE)
        stop = min(start + length, len(y))
        if start < stop:
            y[start:stop] += HIT_LEVEL * strength * _hit(stop - start, HIT_FREQ * pitch[start])
    return y


def _lowpass(y, cutoff):
    """Two cascaded one-pole low-passes whose cutoff is updated every block."""
    out = np.empty_like(y)
    last1 = last2 = 0.0
    for start in range(0, len(y), FILTER_BLOCK):
        stop = min(start + FILTER_BLOCK, len(y))
        a = 1.0 - np.exp(-TWO_PI * cutoff[(start + stop) // 2] / SAMPLE_RATE)
        # lfilter's state for y[n] = a*x[n] + (1-a)*y[n-1] is (1-a) * previous output.
        stage1, _ = lfilter([a], [1.0, a - 1.0], y[start:stop], zi=[(1.0 - a) * last1])
        stage2, _ = lfilter([a], [1.0, a - 1.0], stage1, zi=[(1.0 - a) * last2])
        last1, last2 = stage1[-1], stage2[-1]
        out[start:stop] = stage2
    return out


def _pan(y, position):
    """Equal-power pan of mono ``y`` by ``position`` in [-1, 1]; returns ``(N, 2)``."""
    theta = (position + 1.0) * np.pi / 4.0
    return np.stack([y * np.cos(theta), y * np.sin(theta)], axis=1)


# ── Public API ───────────────────────────────────────────────────────────────


def render(features, fps, seed=None, min_depth=6, max_depth=10, verbose=False):
    """Render stereo audio matching ``features`` (one value per frame at ``fps``).

    Returns ``(samples, root)`` where samples is float32 ``(N, 2)`` normalized
    to 0.9 peak, and ``N = int(n_frames / fps * SAMPLE_RATE)``.
    """
    motion = np.asarray(features["motion"], dtype=np.float64)
    n_frames = len(motion)
    seconds = n_frames / fps
    n = int(seconds * SAMPLE_RATE)
    sample_times = np.arange(n) / SAMPLE_RATE
    frame_times = np.arange(n_frames) / fps

    def per_sample(curve):
        return np.interp(sample_times, frame_times, curve)

    pitch = per_sample(_pitch_ratio(features["hue"], features["saturation"], fps))
    for attempt in range(MAX_TREE_ATTEMPTS):
        attempt_seed = None if seed is None else seed + attempt
        y, root = synthesize(seconds, attempt_seed, min_depth, max_depth, pitch=pitch, verbose=verbose)
        y = _dc_block(y)
        if np.sqrt(np.mean(y ** 2)) > SILENCE_RMS:
            break
        print(f"[audio] Tree for seed {attempt_seed} is silent, rebuilding")
    y = normalize(y).astype(np.float64)

    drive = per_sample(_drive(features["saturation"], fps))
    y = np.tanh(drive * y) / np.tanh(drive)

    y *= per_sample(_gain(motion, fps))
    y = _add_hits(y, _onsets(motion, fps), fps, pitch)
    y = _lowpass(y, per_sample(_cutoff(features["brightness"], fps)))
    stereo = _pan(y, per_sample(_pan_position(features["pan"], motion, fps)))
    return normalize(stereo), root


def generate_wav(path, features, fps, seed=None, min_depth=6, max_depth=10, verbose=False):
    """Render video-synchronised audio with :func:`render` and write a stereo WAV.

    Returns the root node of the audio tree.
    """
    samples, root = render(features, fps, seed, min_depth, max_depth, verbose)
    write_wav(path, samples)
    return root
