"""Live real-time video server (``hecomes-video-live``).

Streams the :mod:`hecomes.artgen.fast` backend straight to a browser as MJPEG,
generated on the fly at the target frame rate rather than rendered to a file
first.  The point is the same as ``hecomes-video-fast`` — proving the generator
beats the clock — except here nothing is ever written to disk.

Design notes:

* Frames are JPEG-encoded **inside the worker processes**.  That parallelises
  the encode (1.7 ms/frame at 540x960) and, more importantly, shrinks what
  crosses the pool's pipe from 1.5 MB of raw rgb24 to roughly 32 KB.
* The producer paces itself against a wall clock, so when generation is faster
  than real time the workers idle instead of racing ahead.  At most
  ``--processes + 1`` chunks are ever in flight, which bounds both memory and
  the latency between a frame being computed and being shown.
* Every viewer reads the same generated stream.  A consumer that falls behind
  skips frames instead of stalling the producer, so a slow browser cannot drag
  the generator below real time.
"""

import argparse
import io
import multiprocessing as mp
import os
import threading
import time
from collections import deque

import numpy as np
from flask import Flask, Response, jsonify
from PIL import Image

from hecomes.artgen.fast import restrict_weights
from hecomes.cli.video_fast import compute_chunk, draw_plan_within_budget, init_worker
from hecomes.config import PERSONALITIES_DIR, load_personality_list

_quality = 80

#: How far behind the wall clock the producer tolerates before giving up on
#: catching the missed frames up and restarting the clock from now.
_RESYNC_LAG = 1.0


def init_live_worker(plan, color_space, quality):
    """Pool initialiser: the fast backend's, plus the JPEG quality."""
    global _quality
    init_worker(plan, color_space)
    _quality = quality


def compute_chunk_jpeg(steps):
    """Evaluate one chunk of frames and return them as encoded JPEGs."""
    frames = compute_chunk(steps)
    encoded = []
    for frame in frames:
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, "JPEG", quality=_quality)
        encoded.append(buf.getvalue())
    return encoded


class FrameHub:
    """Holds the most recent frame for any number of readers.

    Readers pass back the sequence number they last saw and block until a newer
    frame exists, so a reader that cannot keep up simply misses frames rather
    than applying backpressure to the generator.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0

    def publish(self, jpeg):
        with self._cond:
            self._frame = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait(self, seen, timeout=5.0):
        """Return ``(jpeg, seq)`` once ``seq > seen``, or ``(None, seen)`` on timeout."""
        with self._cond:
            if self._seq == seen:
                self._cond.wait(timeout)
            if self._seq == seen:
                return None, seen
            return self._frame, self._seq


class Producer(threading.Thread):
    """Draws scenes and pushes their frames into a :class:`FrameHub` in real time."""

    def __init__(self, hub, weights, args):
        super().__init__(daemon=True)
        self.hub = hub
        self.weights = weights
        self.args = args
        self.status = {
            "seed": None, "tree": None, "capacity_fps": None,
            "delivered_fps": 0.0, "target_fps": args.fps, "frames": 0,
            "width": args.width, "height": args.height, "error": None,
        }
        self._reseed = threading.Event()
        self._stop = threading.Event()
        self._recent = deque(maxlen=args.fps * 2)
        # One chunk per worker: enough contention to be honest, cheap to redraw on.
        probe = (np.arange(args.processes * args.chunk_size) / args.fps).astype(np.float32)
        self._probe_chunks = [
            probe[s:s + args.chunk_size].reshape(-1, 1, 1, 1)
            for s in range(0, len(probe), args.chunk_size)
        ]

    def reseed(self):
        self._reseed.set()

    def stop(self):
        self._stop.set()
        self._reseed.set()

    def run(self):
        try:
            while not self._stop.is_set():
                self._play_scene()
        except BaseException as exc:  # SystemExit included: the budget raises it
            self.status["error"] = str(exc) or type(exc).__name__
            raise

    def _next_seed(self):
        if self.args.seed is not None and self.status["seed"] is None:
            return self.args.seed
        return int(np.random.randint(0, 2**31 - 1))

    def _play_scene(self):
        args = self.args
        self._reseed.clear()
        plan, chain, capacity, seed = draw_plan_within_budget(
            self._next_seed(), self.weights, args, self._probe_chunks
        )
        self.status.update(seed=seed, tree=" -> ".join(chain),
                           capacity_fps=None if capacity is None else round(capacity, 1))
        measured = ("capacity not measured (--no-budget)" if capacity is None else
                    f"{capacity:.1f} fps capacity ({capacity / args.fps:.2f}x real time)")
        print(f"Scene {seed}: {' -> '.join(chain)} — {measured}")

        interval = 1.0 / args.fps
        scene_end = time.monotonic() + args.rotate if args.rotate else float("inf")
        frame = 0
        deadline = None  # started on the first published frame, not here: drawing
        with mp.Pool(args.processes, initializer=init_live_worker,
                     initargs=(plan, args.color_space, args.quality)) as pool:
            inflight = deque()
            while not self._reseed.is_set() and time.monotonic() < scene_end:
                while len(inflight) <= args.processes:
                    steps = (np.arange(frame, frame + args.chunk_size) / args.fps)
                    inflight.append(pool.apply_async(
                        compute_chunk_jpeg, (steps.astype(np.float32).reshape(-1, 1, 1, 1),)
                    ))
                    frame += args.chunk_size
                for jpeg in inflight.popleft().get():
                    # the tree and starting the pool must not count as lateness
                    deadline = (time.perf_counter() if deadline is None else deadline) + interval
                    slack = deadline - time.perf_counter()
                    if slack > 0:
                        time.sleep(slack)
                    elif slack < -_RESYNC_LAG:
                        deadline = time.perf_counter()  # hopelessly behind; restart the clock
                    self.hub.publish(jpeg)
                    self._tick()
                    if self._reseed.is_set():
                        break

    def _tick(self):
        now = time.perf_counter()
        self._recent.append(now)
        self.status["frames"] += 1
        if len(self._recent) > 1:
            span = self._recent[-1] - self._recent[0]
            self.status["delivered_fps"] = round((len(self._recent) - 1) / span, 1) if span else 0.0


_BOUNDARY = "hecomesframe"

_PAGE = """<!doctype html>
<meta charset="utf-8"><title>HeComes — live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0b0b0d; color: #d8d8dc; display: flex; gap: 1.5rem;
         align-items: center; justify-content: center; min-height: 100vh; flex-wrap: wrap;
         font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; padding: 1rem; }
  img { max-height: 90vh; max-width: 100%; border-radius: 6px; background: #17171b; }
  aside { min-width: 15rem; max-width: 22rem; }
  h1 { font-size: 1rem; letter-spacing: .08em; text-transform: uppercase; color: #8a8a94; margin: 0 0 1rem; }
  dl { display: grid; grid-template-columns: auto 1fr; gap: .35rem .9rem; margin: 0 0 1.25rem; }
  dt { color: #8a8a94; } dd { margin: 0; overflow-wrap: anywhere; }
  button { font: inherit; color: inherit; background: #23232a; border: 1px solid #35353f;
           border-radius: 5px; padding: .5rem 1rem; cursor: pointer; }
  button:hover { background: #2d2d36; }
  .warn { color: #e5a663; }
</style>
<img src="/stream.mjpg" alt="live stream">
<aside>
  <h1>HeComes live</h1>
  <dl>
    <dt>frame</dt><dd id="size">—</dd>
    <dt>delivered</dt><dd id="fps">—</dd>
    <dt>capacity</dt><dd id="cap">—</dd>
    <dt>seed</dt><dd id="seed">—</dd>
    <dt>tree</dt><dd id="tree">—</dd>
  </dl>
  <button onclick="fetch('/next', {method: 'POST'})">New scene</button>
  <p id="err" class="warn"></p>
</aside>
<script>
setInterval(async () => {
  const s = await (await fetch('/stats')).json();
  size.textContent = `${s.width}x${s.height} @ ${s.target_fps} fps`;
  fps.textContent = `${s.delivered_fps} fps`;
  fps.className = s.delivered_fps < s.target_fps - 2 ? 'warn' : '';
  cap.textContent = s.capacity_fps ? `${s.capacity_fps} fps (${(s.capacity_fps / s.target_fps).toFixed(2)}x)` : '—';
  seed.textContent = s.seed ?? '—';
  tree.textContent = s.tree ?? 'drawing…';
  err.textContent = s.error || '';
}, 1000);
</script>
"""


def build_app(hub, producer):
    app = Flask(__name__)

    @app.get("/")
    def index():
        return Response(_PAGE, mimetype="text/html")

    @app.get("/stream.mjpg")
    def stream():
        def frames():
            seen = 0
            while True:
                jpeg, seen = hub.wait(seen)
                if jpeg is None:
                    continue
                yield (b"--" + _BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")

        return Response(frames(),
                        mimetype=f"multipart/x-mixed-replace; boundary={_BOUNDARY}")

    @app.get("/stats")
    def stats():
        return jsonify(producer.status)

    @app.post("/next")
    def next_scene():
        producer.reseed()
        return jsonify(ok=True)

    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="hecomes-video-live",
        description="Serve generated video to a browser, in real time.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind address. Use 0.0.0.0 to expose on the LAN.")
    p.add_argument("--port", type=int, default=5001, help="Port to listen on.")
    p.add_argument("-W", "--width", type=int, default=540, help="Frame width.")
    p.add_argument("-H", "--height", type=int, default=960, help="Frame height.")
    p.add_argument("-f", "--fps", type=int, default=30, help="Frames per second to stream at.")
    p.add_argument("-S", "--seed", type=int, default=None, help="Seed for the first scene.")
    p.add_argument("-p", "--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="Worker processes. Default: CPU count minus one.")
    p.add_argument("-c", "--chunk-size", type=int, default=10, help="Frames per batch.")
    p.add_argument("-q", "--quality", type=int, default=80, help="JPEG quality (1-95).")
    p.add_argument("-r", "--rotate", type=float, default=0.0,
                   help="Seconds before drawing a new scene. 0 (default) keeps one scene "
                        "running; the leaf drift keeps it moving. A rotation freezes the "
                        "last frame for a second or two while the next tree is drawn.")
    p.add_argument("--personality", default="personality", help="Personality name in data/personalities/.")
    p.add_argument("--color-space", choices=("rgb", "hsv", "cmy"), default="rgb")
    p.add_argument("--sampling", choices=("bilinear", "nearest"), default="bilinear",
                   help="Warp interpolation. Default: bilinear.")
    p.add_argument("--min-depth", type=int, default=6, help="Minimum tree depth.")
    p.add_argument("--max-depth", type=int, default=12, help="Maximum tree depth.")
    p.add_argument("--drift", type=float, default=4e-3, help="Leaf drift per second.")
    p.add_argument("--budget-fps", type=float, default=None,
                   help="Throughput a tree must reach to be accepted. Default: --fps.")
    p.add_argument("--max-tries", type=int, default=20,
                   help="Trees to draw before giving up on the budget.")
    p.add_argument("--no-budget", action="store_true",
                   help="Accept the first tree drawn, however slow.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    weights, dropped = restrict_weights(
        load_personality_list(PERSONALITIES_DIR / (args.personality + ".json"))
    )
    if dropped:
        print(f"Note: no fast path for {', '.join(dropped)} — dropped from the personality.")

    if args.fps < 1:
        raise SystemExit("--fps must be at least 1.")
    if not 1 <= args.quality <= 95:
        raise SystemExit("--quality must be between 1 and 95.")

    hub = FrameHub()
    producer = Producer(hub, weights, args)
    producer.start()

    print(f"Streaming {args.width}x{args.height} at {args.fps} fps on "
          f"{args.processes} workers — http://{args.host}:{args.port}/")
    try:
        build_app(hub, producer).run(host=args.host, port=args.port,
                                     threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        producer.stop()


if __name__ == "__main__":
    main()
