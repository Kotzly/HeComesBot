"""Parameter path system for the path-based animation backend.

A :class:`Path` maps time (seconds, float) to a ``dict[str, value]`` of leaf
param names → values.  The dict may cover one param (scalar paths like
:class:`Oscillate`) or several coupled params (e.g. :class:`CircularOrbit`
outputs both ``cx`` and ``cy`` together).

Typical usage::

    path = CircularOrbit(cx0=0.2, cy0=0.1, r=0.3, omega=1.5)
    params = path(t=2.5)   # {"cx": ..., "cy": ...}

Adding a new path type:

1. Subclass :class:`Path` (or :class:`GeneralODEPath` for ODE-based paths).
2. Implement ``__call__``, ``to_dict``, and ``from_dict``.
3. Register in :data:`PATH_REGISTRY`::

       PATH_REGISTRY["MyPath"] = MyPath
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from hecomes.artgen.func_utils import hsv_to_rgb


# ── Abstract base ─────────────────────────────────────────────────────────────


class Path(ABC):
    """Abstract base for all parameter paths.

    Subclasses set the class attribute ``periodic = True/False`` (or override it
    as an instance attribute in ``__init__`` for paths where periodicity depends
    on constructor arguments, like :class:`WaypointPath`).
    """

    periodic: bool = False

    @abstractmethod
    def __call__(self, t: float) -> dict[str, Any]:
        """Return param dict at time *t* (seconds)."""
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict.  Must include a ``"type"`` key."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "Path":
        """Reconstruct from the dict produced by :meth:`to_dict`."""
        ...


# ── Scalar paths ──────────────────────────────────────────────────────────────


class LinearDrift(Path):
    """``param(t) = start + rate * t``  (non-periodic, open-ended drift)."""

    periodic = False

    def __init__(self, param: str, start: float, rate: float):
        self.param = param
        self.start = float(start)
        self.rate = float(rate)

    def __call__(self, t: float) -> dict:
        return {self.param: self.start + self.rate * t}

    def to_dict(self) -> dict:
        return {"type": "LinearDrift", "param": self.param,
                "start": self.start, "rate": self.rate}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearDrift":
        return cls(d["param"], d["start"], d["rate"])


class Oscillate(Path):
    """``param(t) = base + amplitude * sin(omega * t + phase)``  (periodic)."""

    periodic = True

    def __init__(self, param: str, base: float, amplitude: float,
                 omega: float, phase: float = 0.0):
        self.param = param
        self.base = float(base)
        self.amplitude = float(amplitude)
        self.omega = float(omega)
        self.phase = float(phase)

    def __call__(self, t: float) -> dict:
        return {self.param: self.base + self.amplitude * math.sin(self.omega * t + self.phase)}

    def to_dict(self) -> dict:
        return {"type": "Oscillate", "param": self.param, "base": self.base,
                "amplitude": self.amplitude, "omega": self.omega, "phase": self.phase}

    @classmethod
    def from_dict(cls, d: dict) -> "Oscillate":
        return cls(d["param"], d["base"], d["amplitude"], d["omega"], d.get("phase", 0.0))


class AngularVelocity(Path):
    """``angle(t) = (angle0 + omega * t) % 2π``  (periodic, wraps at 2π)."""

    periodic = True

    def __init__(self, angle0: float, omega: float):
        self.angle0 = float(angle0)
        self.omega = float(omega)

    def __call__(self, t: float) -> dict:
        return {"angle": (self.angle0 + self.omega * t) % (2.0 * math.pi)}

    def to_dict(self) -> dict:
        return {"type": "AngularVelocity", "angle0": self.angle0, "omega": self.omega}

    @classmethod
    def from_dict(cls, d: dict) -> "AngularVelocity":
        return cls(d["angle0"], d["omega"])


# ── Coupled paths ─────────────────────────────────────────────────────────────


class CircularOrbit(Path):
    """Center ``(cx, cy)`` moves along a circle (periodic).

    ``cx(t) = cx0 + r * cos(omega * t + phase)``
    ``cy(t) = cy0 + r * sin(omega * t + phase)``
    """

    periodic = True

    def __init__(self, cx0: float, cy0: float, r: float,
                 omega: float, phase: float = 0.0):
        self.cx0 = float(cx0)
        self.cy0 = float(cy0)
        self.r = float(r)
        self.omega = float(omega)
        self.phase = float(phase)

    def __call__(self, t: float) -> dict:
        angle = self.omega * t + self.phase
        return {
            "cx": self.cx0 + self.r * math.cos(angle),
            "cy": self.cy0 + self.r * math.sin(angle),
        }

    def to_dict(self) -> dict:
        return {"type": "CircularOrbit", "cx0": self.cx0, "cy0": self.cy0,
                "r": self.r, "omega": self.omega, "phase": self.phase}

    @classmethod
    def from_dict(cls, d: dict) -> "CircularOrbit":
        return cls(d["cx0"], d["cy0"], d["r"], d["omega"], d.get("phase", 0.0))


class HuePath(Path):
    """Color path in HSV space — outputs ``{"color": [r, g, b]}``.

    Hue cycles at constant speed; saturation and value oscillate independently.
    All three channels are controlled by this single path object so that the
    color primitive always receives a coherent RGB triple.

    Parameters
    ----------
    h0, s0, v0:
        Initial hue (0–1), saturation (0–1), value (0–1).
    h_speed:
        Hue cycles per *period* seconds.  ``h(t) = (h0 + h_speed * t / period) % 1``.
    s_A, s_omega:
        Saturation oscillation: ``s(t) = s0 + s_A * sin(s_omega * t)``.
    v_A, v_omega:
        Value oscillation: ``v(t) = v0 + v_A * sin(v_omega * t)``.
    period:
        Hue cycle period in seconds (= video duration for seamless loops).
    """

    periodic = True

    def __init__(self, h0: float, s0: float, v0: float,
                 h_speed: float = 1.0,
                 s_A: float = 0.0, s_omega: float = 1.0,
                 v_A: float = 0.0, v_omega: float = 1.0,
                 period: float = 10.0):
        self.h0 = float(h0)
        self.s0 = float(s0)
        self.v0 = float(v0)
        self.h_speed = float(h_speed)
        self.s_A = float(s_A)
        self.s_omega = float(s_omega)
        self.v_A = float(v_A)
        self.v_omega = float(v_omega)
        self.period = float(period)

    def __call__(self, t: float) -> dict:
        h = (self.h0 + self.h_speed * t / self.period) % 1.0
        s = float(np.clip(self.s0 + self.s_A * math.sin(self.s_omega * t), 0.0, 1.0))
        v = float(np.clip(self.v0 + self.v_A * math.sin(self.v_omega * t), 0.0, 1.0))
        hsv = np.array([[[h, s, v]]], dtype=np.float32)
        rgb = hsv_to_rgb(hsv)[0, 0].tolist()
        return {"color": rgb}

    def to_dict(self) -> dict:
        return {
            "type": "HuePath",
            "h0": self.h0, "s0": self.s0, "v0": self.v0,
            "h_speed": self.h_speed,
            "s_A": self.s_A, "s_omega": self.s_omega,
            "v_A": self.v_A, "v_omega": self.v_omega,
            "period": self.period,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HuePath":
        return cls(
            d["h0"], d["s0"], d["v0"],
            d.get("h_speed", 1.0),
            d.get("s_A", 0.0), d.get("s_omega", 1.0),
            d.get("v_A", 0.0), d.get("v_omega", 1.0),
            d.get("period", 10.0),
        )


# ── Waypoint path ─────────────────────────────────────────────────────────────


class WaypointPath(Path):
    """Piecewise-interpolated path through a sequence of keyframe values.

    Parameters
    ----------
    times:
        Sorted list of keyframe times in seconds (length ≥ 2).
    waypoints:
        List of ``{param_name: value}`` dicts, one per keyframe.  A keyframe
        may omit params that don't change — missing params fall back to the
        nearest previous keyframe with that param defined.
    interpolation:
        ``"linear"`` (default), ``"step"`` (nearest-left), or ``"cubic"``
        (requires ≥ 4 keyframes, falls back to linear otherwise).
    loop:
        If ``True``, time wraps at ``times[-1]`` (periodic).

    Example::

        path = WaypointPath(
            times=[0, 2, 4, 6],
            waypoints=[
                {"cx": -0.5, "cy": 0.0},
                {"cx":  0.5, "cy": 0.3},
                {"cx":  0.0, "cy": -0.4},
                {"cx": -0.5, "cy": 0.0},  # close the loop
            ],
            loop=True,
        )
    """

    def __init__(self, times: list, waypoints: list,
                 interpolation: str = "linear", loop: bool = False):
        if len(times) < 2 or len(times) != len(waypoints):
            raise ValueError("times and waypoints must have the same length (≥ 2)")
        self.times = list(times)
        self.waypoints = list(waypoints)
        self.interpolation = interpolation
        self.periodic = loop  # instance attribute overrides class default

    def __call__(self, t: float) -> dict:
        t_arr = np.asarray(self.times, dtype=np.float64)
        if self.periodic:
            t = float(t) % float(t_arr[-1])
        t = float(np.clip(t, t_arr[0], t_arr[-1]))

        param_names: set[str] = set()
        for wp in self.waypoints:
            param_names.update(wp.keys())

        result: dict[str, Any] = {}
        for name in param_names:
            valid_times, vals = [], []
            for ti, wp in zip(self.times, self.waypoints):
                if name in wp:
                    valid_times.append(ti)
                    vals.append(wp[name])
            if len(valid_times) == 1:
                result[name] = vals[0]
                continue
            vt = np.asarray(valid_times, dtype=np.float64)
            vv = np.asarray(vals, dtype=np.float64)
            if self.interpolation == "step":
                idx = max(0, int(np.searchsorted(vt, t, side="right")) - 1)
                result[name] = float(vv[idx])
            elif self.interpolation == "cubic" and len(vv) >= 4:
                from scipy.interpolate import CubicSpline
                result[name] = float(CubicSpline(vt, vv)(t))
            else:
                result[name] = float(np.interp(t, vt, vv))
        return result

    def to_dict(self) -> dict:
        return {
            "type": "WaypointPath",
            "times": self.times,
            "waypoints": self.waypoints,
            "interpolation": self.interpolation,
            "loop": self.periodic,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WaypointPath":
        return cls(d["times"], d["waypoints"],
                   d.get("interpolation", "linear"), d.get("loop", False))


# ── Random walk path ─────────────────────────────────────────────────────────


class RandomWalkPath(Path):
    """2D random walk for ``(cx, cy)`` — smooth PCHIP-interpolated motion.

    Generates random waypoints at evenly-spaced times and interpolates with a
    PCHIP spline (monotone — never overshoots the waypoint range).

    Parameters
    ----------
    cx0, cy0:
        Starting position.
    n_steps:
        Number of random steps (total waypoints = n_steps + 1).
    step_size:
        Maximum displacement per step; each step is sampled uniformly in
        ``[0, step_size]`` at a random angle.
    duration:
        Total animation duration in seconds (time of last waypoint).
    bound:
        Position is clamped to ``[-bound, bound]`` on each axis after each step.
    seed:
        Optional integer seed for reproducibility.
    """

    periodic = False

    def __init__(
        self,
        cx0: float,
        cy0: float,
        n_steps: int,
        step_size: float,
        duration: float,
        bound: float = 1.5,
        seed: int | None = None,
    ):
        from scipy.interpolate import PchipInterpolator

        self.cx0 = float(cx0)
        self.cy0 = float(cy0)
        self.n_steps = int(n_steps)
        self.step_size = float(step_size)
        self.duration = float(duration)
        self.bound = float(bound)
        self.seed = seed

        rng = np.random.RandomState(seed)
        times = np.linspace(0.0, duration, self.n_steps + 1)
        pos = np.zeros((self.n_steps + 1, 2))
        pos[0] = [self.cx0, self.cy0]
        for i in range(1, self.n_steps + 1):
            angle = rng.uniform(0.0, 2.0 * np.pi)
            step = rng.uniform(0.0, self.step_size)
            pos[i, 0] = float(np.clip(pos[i - 1, 0] + step * math.cos(angle), -bound, bound))
            pos[i, 1] = float(np.clip(pos[i - 1, 1] + step * math.sin(angle), -bound, bound))

        self._times = times
        self._cx_spline = PchipInterpolator(times, pos[:, 0])
        self._cy_spline = PchipInterpolator(times, pos[:, 1])

    def __call__(self, t: float) -> dict:
        t_c = float(np.clip(t, self._times[0], self._times[-1]))
        return {
            "cx": float(self._cx_spline(t_c)),
            "cy": float(self._cy_spline(t_c)),
        }

    def to_dict(self) -> dict:
        return {
            "type": "RandomWalkPath",
            "cx0": self.cx0,
            "cy0": self.cy0,
            "n_steps": self.n_steps,
            "step_size": self.step_size,
            "duration": self.duration,
            "bound": self.bound,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RandomWalkPath":
        return cls(
            d["cx0"], d["cy0"], d["n_steps"], d["step_size"], d["duration"],
            d.get("bound", 1.5), d.get("seed"),
        )


# ── ODE paths ─────────────────────────────────────────────────────────────────


class ODEPath(Path, ABC):
    """Abstract base for ODE-driven paths.

    Subclasses implement :meth:`_rhs`, :meth:`y0`, :attr:`state_dim`, and
    :attr:`param_names`.  Call :meth:`precompute` once (sequentially, before
    the render pool is created) to integrate the ODE over all frames.  After
    that, ``__call__(t)`` simply indexes the stored table — no repeated
    integration during rendering.

    Subclass hierarchy::

        ODEPath (ABC)
        ├── LinearODEPath        fully serializable; dy/dt = A @ y + b
        └── GeneralODEPath (ABC) user-defined; subclass and implement _rhs
    """

    periodic = False

    def __init__(self) -> None:
        self._table: np.ndarray | None = None
        self._dt: float | None = None

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the ODE state vector."""
        ...

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Leaf param names controlled by this path (length == state_dim)."""
        ...

    @abstractmethod
    def y0(self) -> np.ndarray:
        """Initial state vector, shape ``(state_dim,)``."""
        ...

    @abstractmethod
    def _rhs(self, y: np.ndarray, t: float) -> np.ndarray:
        """Right-hand side: ``dy/dt = f(y, t)``.  ``y`` shape: ``(state_dim,)``."""
        ...

    def precompute(self, n_frames: int, dt: float, solver: str = "rk4") -> None:
        """Integrate the ODE for *n_frames* steps of size *dt* (seconds).

        Parameters
        ----------
        n_frames:
            Total number of frames in the video.
        dt:
            Time step in seconds (= 1 / fps).
        solver:
            ``"euler"`` (first-order) or ``"rk4"`` (fourth-order Runge-Kutta,
            default).
        """
        table = np.zeros((n_frames, self.state_dim), dtype=np.float64)
        y = self.y0().astype(np.float64)
        self._dt = float(dt)
        for i in range(n_frames):
            table[i] = y
            t_i = i * dt
            if solver == "euler":
                y = y + dt * self._rhs(y, t_i)
            else:  # rk4
                k1 = self._rhs(y, t_i)
                k2 = self._rhs(y + 0.5 * dt * k1, t_i + 0.5 * dt)
                k3 = self._rhs(y + 0.5 * dt * k2, t_i + 0.5 * dt)
                k4 = self._rhs(y + dt * k3, t_i + dt)
                y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self._table = table

    def __call__(self, t: float) -> dict:
        if self._table is None:
            raise RuntimeError(
                f"{type(self).__name__}.precompute() must be called before evaluation."
            )
        idx = int(round(float(t) / self._dt))
        idx = max(0, min(idx, len(self._table) - 1))
        state = self._table[idx]
        return {name: float(state[i]) for i, name in enumerate(self.param_names)}

    def to_dict(self) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} does not support to_dict. "
            "Override this method in your subclass."
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ODEPath":
        raise NotImplementedError


class LinearODEPath(ODEPath):
    """Linear time-invariant ODE: ``dy/dt = A @ y + b``.

    Fully serializable — A, b, and y0 are plain numeric arrays.

    Parameters
    ----------
    A:
        State matrix, shape ``(state_dim, state_dim)``.
    b:
        Constant input vector, shape ``(state_dim,)``.
    y0_init:
        Initial state, shape ``(state_dim,)``.
    param_names_list:
        Names of the leaf params the state vector maps to.

    Example — two-dimensional oscillator that couples ``cx`` and ``cy``::

        path = LinearODEPath(
            A=[[0, 1], [-1, -0.1]],   # damped harmonic
            b=[0, 0],
            y0_init=[0.5, 0.0],
            param_names_list=["cx", "cy"],
        )
    """

    def __init__(self, A, b, y0_init, param_names_list: list[str]):
        super().__init__()
        self._A = np.array(A, dtype=np.float64)
        self._b = np.array(b, dtype=np.float64)
        self._y0 = np.array(y0_init, dtype=np.float64)
        self._param_names = list(param_names_list)

    @property
    def state_dim(self) -> int:
        return len(self._y0)

    @property
    def param_names(self) -> list[str]:
        return self._param_names

    def y0(self) -> np.ndarray:
        return self._y0.copy()

    def _rhs(self, y: np.ndarray, t: float) -> np.ndarray:
        return self._A @ y + self._b

    def to_dict(self) -> dict:
        return {
            "type": "LinearODEPath",
            "A": self._A.tolist(),
            "b": self._b.tolist(),
            "y0": self._y0.tolist(),
            "param_names": self._param_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LinearODEPath":
        return cls(d["A"], d["b"], d["y0"], d["param_names"])


class GeneralODEPath(ODEPath, ABC):
    """Abstract base for user-defined ODE paths.

    Subclass this, implement ``_rhs``, ``state_dim``, ``param_names``, and
    ``y0``, then register in :data:`PATH_REGISTRY` to enable serialization::

        class LorenzPath(GeneralODEPath):
            @property
            def state_dim(self): return 3

            @property
            def param_names(self): return ["cx", "cy", "rx"]

            def y0(self): return np.array([1.0, 0.0, 0.5])

            def _rhs(self, y, t):
                sigma, rho, beta = 10.0, 28.0, 8/3
                x, yy, z = y
                return np.array([
                    sigma * (yy - x),
                    x * (rho - z) - yy,
                    x * yy - beta * z,
                ])

        PATH_REGISTRY["LorenzPath"] = LorenzPath
    """
    pass


# ── Built-in GeneralODEPath examples ─────────────────────────────────────────


class LorenzPath(GeneralODEPath):
    """Lorenz attractor ODE — animates ``cx``, ``cy``, and ``rx``.

    The classic chaotic attractor ``(σ, ρ, β) = (10, 28, 8/3)``.  Raw state
    values range roughly x, y ∈ [−20, 20] and z ∈ [0, 50]; scale factors map
    them into typical leaf param ranges::

        cx(t) = x(t) / cx_scale        # target range [-2, 2]  → cx_scale ≈ 10
        cy(t) = y(t) / cy_scale        # target range [-2, 2]  → cy_scale ≈ 10
        rx(t) = clip(z(t) / rx_scale,  # target range [0.05, 0.95] → rx_scale ≈ 50
                     0.05, 0.95)

    Parameters
    ----------
    sigma, rho, beta:
        System parameters.  Default ``(10, 28, 8/3)`` gives the chaotic regime.
    y0_init:
        Initial state ``[x₀, y₀, z₀]``.  Default ``[1.0, 0.0, 0.0]``.
    cx_scale, cy_scale, rx_scale:
        Divisors applied after table lookup.  Adjust to fit your canvas size.
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        y0_init: list | None = None,
        cx_scale: float = 10.0,
        cy_scale: float = 10.0,
        rx_scale: float = 50.0,
    ):
        super().__init__()
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)
        self._y0_init: list = list(y0_init or [1.0, 0.0, 0.0])
        self.cx_scale = float(cx_scale)
        self.cy_scale = float(cy_scale)
        self.rx_scale = float(rx_scale)

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def param_names(self) -> list[str]:
        return ["cx", "cy", "rx"]

    def y0(self) -> np.ndarray:
        return np.array(self._y0_init, dtype=np.float64)

    def _rhs(self, y: np.ndarray, t: float) -> np.ndarray:
        x, yy, z = y
        return np.array([
            self.sigma * (yy - x),
            x * (self.rho - z) - yy,
            x * yy - self.beta * z,
        ])

    def __call__(self, t: float) -> dict:
        if self._table is None:
            raise RuntimeError("LorenzPath.precompute() must be called before evaluation.")
        idx = max(0, min(int(round(float(t) / self._dt)), len(self._table) - 1))
        x, y, z = self._table[idx]
        return {
            "cx": x / self.cx_scale,
            "cy": y / self.cy_scale,
            "rx": float(np.clip(abs(z) / self.rx_scale, 0.05, 0.95)),
        }

    def to_dict(self) -> dict:
        return {
            "type": "LorenzPath",
            "sigma": self.sigma,
            "rho": self.rho,
            "beta": self.beta,
            "y0": self._y0_init,
            "cx_scale": self.cx_scale,
            "cy_scale": self.cy_scale,
            "rx_scale": self.rx_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LorenzPath":
        return cls(
            d.get("sigma", 10.0),
            d.get("rho", 28.0),
            d.get("beta", 8.0 / 3.0),
            d.get("y0"),
            d.get("cx_scale", 10.0),
            d.get("cy_scale", 10.0),
            d.get("rx_scale", 50.0),
        )


class VanDerPolPath(GeneralODEPath):
    """Van der Pol oscillator ODE — animates ``cx`` and ``cy``.

    A non-linear oscillator with a stable limit cycle.  Unlike purely sinusoidal
    motion, the Van der Pol cycle has asymmetric acceleration: it builds slowly
    near the extremes and relaxes quickly through the center, producing smooth
    but non-uniform orbiting motion.

    The system (rewritten as first order)::

        dx/dt = y
        dy/dt = μ(1 − x²)y − x

    With ``μ = 1`` the limit cycle has amplitude ≈ 2 for ``x`` and ≈ 2 for
    ``y``.  Larger ``μ`` produces sharper relaxation oscillations with faster
    transitions.  Scale factors map the state into leaf param ranges::

        cx(t) = x(t) / cx_scale   # x amplitude ≈ 2 → cx_scale = 1 keeps cx in [-2, 2]
        cy(t) = y(t) / cy_scale   # y amplitude ≈ 2 → cy_scale = 1 keeps cy in [-2, 2]

    Parameters
    ----------
    mu:
        Damping / non-linearity strength.  ``μ = 1`` (default) gives smooth
        oscillation; ``μ ≥ 3`` gives visible relaxation behaviour.
    y0_init:
        Initial conditions ``[x₀, dx/dt₀]``.  Default ``[2.0, 0.0]`` starts
        approximately on the limit cycle.
    cx_scale, cy_scale:
        Divisors applied after table lookup.
    """

    def __init__(
        self,
        mu: float = 1.0,
        y0_init: list | None = None,
        cx_scale: float = 1.0,
        cy_scale: float = 1.0,
    ):
        super().__init__()
        self.mu = float(mu)
        self._y0_init: list = list(y0_init or [2.0, 0.0])
        self.cx_scale = float(cx_scale)
        self.cy_scale = float(cy_scale)

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def param_names(self) -> list[str]:
        return ["cx", "cy"]

    def y0(self) -> np.ndarray:
        return np.array(self._y0_init, dtype=np.float64)

    def _rhs(self, y: np.ndarray, t: float) -> np.ndarray:
        x, dxdt = y
        return np.array([
            dxdt,
            self.mu * (1.0 - x ** 2) * dxdt - x,
        ])

    def __call__(self, t: float) -> dict:
        if self._table is None:
            raise RuntimeError("VanDerPolPath.precompute() must be called before evaluation.")
        idx = max(0, min(int(round(float(t) / self._dt)), len(self._table) - 1))
        x, y = self._table[idx]
        return {
            "cx": x / self.cx_scale,
            "cy": y / self.cy_scale,
        }

    def to_dict(self) -> dict:
        return {
            "type": "VanDerPolPath",
            "mu": self.mu,
            "y0": self._y0_init,
            "cx_scale": self.cx_scale,
            "cy_scale": self.cy_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VanDerPolPath":
        return cls(
            d.get("mu", 1.0),
            d.get("y0"),
            d.get("cx_scale", 1.0),
            d.get("cy_scale", 1.0),
        )


# ── Registry ──────────────────────────────────────────────────────────────────

PATH_REGISTRY: dict[str, type[Path]] = {
    "LinearDrift": LinearDrift,
    "Oscillate": Oscillate,
    "AngularVelocity": AngularVelocity,
    "CircularOrbit": CircularOrbit,
    "RandomWalkPath": RandomWalkPath,
    "HuePath": HuePath,
    "WaypointPath": WaypointPath,
    "LinearODEPath": LinearODEPath,
    "LorenzPath": LorenzPath,
    "VanDerPolPath": VanDerPolPath,
}


def path_from_dict(d: dict) -> Path:
    """Reconstruct a :class:`Path` from its serialized dict.

    The dict must contain a ``"type"`` key matching an entry in
    :data:`PATH_REGISTRY`.
    """
    type_name = d.get("type", "")
    cls = PATH_REGISTRY.get(type_name)
    if cls is None:
        raise KeyError(
            f"Unknown path type '{type_name}'. "
            f"Available: {sorted(PATH_REGISTRY)}"
        )
    return cls.from_dict(d)
