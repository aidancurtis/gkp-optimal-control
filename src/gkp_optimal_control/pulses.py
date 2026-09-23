r"""Compilation of optimized gate parameters into physical control pulses.

This module is the bridge between :mod:`gate_optimization`, which produces
*ideal* gate parameters, and :mod:`pulse_simulation`, which propagates a
time-dependent Hamiltonian. It implements the second stage of the two-step
optimization of Eickbusch et al., *Nat. Phys.* **18**, 1464 (2022) and its
supplementary sections S3, S4 and S9.

Two gate sets are compiled.

**ECD + qubit rotations.** Each :math:`\mathrm{ECD}(\beta)` is realized by four
Gaussian cavity displacements with a qubit :math:`\pi` pulse at the midpoint
(Fig. S3a). The four amplitude ratios are found by Nelder-Mead on the cost
function (S27) evaluated along *semiclassical* phase-space trajectories (S5),
so second-order dispersive shift, Kerr and photon loss all enter the
compilation. The wait time :math:`t_w` is then reduced until the target
:math:`|\beta|` is realized; if :math:`t_w` reaches zero first, the
intermediate radius :math:`\alpha_0` is lowered instead.

**SNAP + displacements.** Each SNAP gate is realized by two number-multiplexed
selective qubit :math:`\pi` pulses whose relative phase per Fock level sets the
imparted phase, following Heeres et al., *PRL* **115**, 137002 (2015). The
displacements are the same fast unselective Gaussians used by the ECD set.

Units
-----
Time is in microseconds throughout; all rates and drive amplitudes are angular
frequencies in rad/us. A drive quoted in Hz is converted with a factor
:math:`2\pi`, e.g. ``chi = 2 * np.pi * 32.8e-3`` for 32.8 kHz.

Conventions
-----------
The system Hamiltonian is Eq. (S1) of the supplement,

.. math::
    H/\hbar = \Delta a^\dagger a - \chi a^\dagger a\, q^\dagger q
              - \chi' a^{\dagger 2} a^2 q^\dagger q
              - K_c a^{\dagger 2} a^2 - K_q q^{\dagger 2} q^2
              + \varepsilon^*(t) a + \Omega^*(t) q + \mathrm{h.c.}

with :math:`\Delta = \chi/2` for ECD control (cavity drive at the mean of the
two qubit-state-dependent cavity frequencies) and :math:`\Delta = 0` for SNAP
control (drive at the ground-state cavity frequency).

Qubit rotations use the convention of
:func:`gate_optimization.qubit_rotation`, i.e.
:math:`R(\theta,\phi) = \exp[-i(\theta/2)(\sigma_x\cos\phi + \sigma_y\sin\phi)]`
with :math:`\langle e|R|g\rangle = -i e^{i\phi}\sin(\theta/2)`. With the drive
term written as :math:`\Omega^* q + \Omega q^\dagger` this is realized by

.. math::
    \Omega(t) = \frac{\theta}{2} e^{i\phi} \frac{g(t)}{\int g},

which is the complex conjugate of the phase convention used in the paper. Note
this when comparing waveforms to Fig. S4.

Displacements follow :math:`\partial_t\alpha = -i\varepsilon(t)`, so an
instantaneous pulse of integrated area :math:`\bar\varepsilon` realizes
:math:`D(-i\bar\varepsilon)`.

The sign convention for the realized conditional displacement,

.. math::
    \beta \equiv \alpha_g(T) - \alpha_e(T),

matches Eq. (S25), :math:`\mathrm{CD}(\beta) = D(\beta/2)|g\rangle\langle g| +
D(-\beta/2)|e\rangle\langle e|`, and therefore
:func:`gate_optimization._build_ecd_sequence`. (Section S4 B of the supplement
states the opposite sign; Eq. (S25) and the main text agree with the choice
made here, and :func:`verify_ecd_pulse` checks it numerically.)
"""

from __future__ import annotations

import json
import os
import re
import warnings
from functools import lru_cache
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "SystemParams",
    "load_config",
    "parse_quantity",
    "PulseSegment",
    "PulseSequence",
    "ECDPulse",
    "gaussian_envelope",
    "rotation_waveform",
    "frame_trajectory",
    "branch_trajectories",
    "optimize_ecd_pulse",
    "compile_ecd_sequence",
    "compile_snap_sequence",
    "compile_from_result",
    "ecd_pulse_cache",
]

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Units and configuration files
# ---------------------------------------------------------------------------

# Internal units are microseconds and rad/us. Config files may instead give a
# string with an explicit unit, which is both safer and self-documenting: the
# factor of 2 pi between a quoted frequency and an angular frequency is the
# single most common source of silent 6x errors in this kind of code.

#: multiplier from <unit> to rad/us, for quantities stored as angular frequencies
_FREQUENCY_UNITS = {
    "Hz": TWO_PI * 1e-6,
    "kHz": TWO_PI * 1e-3,
    "MHz": TWO_PI,
    "GHz": TWO_PI * 1e3,
    "rad/s": 1e-6,
    "rad/ms": 1e-3,
    "rad/us": 1.0,
    "rad/ns": 1e3,
}

#: multiplier from <unit> to us
_TIME_UNITS = {"s": 1e6, "ms": 1e3, "us": 1.0, "ns": 1e-3, "ps": 1e-6}

#: multiplier from <unit> to 1/us, for quantities stored as ordinary rates
_RATE_UNITS = {
    "Hz": 1e-6,
    "kHz": 1e-3,
    "MHz": 1.0,
    "GHz": 1e3,
    "1/s": 1e-6,
    "1/ms": 1e-3,
    "1/us": 1.0,
    "1/ns": 1e3,
}

_KIND_TABLES = {
    "frequency": _FREQUENCY_UNITS,
    "time": _TIME_UNITS,
    "rate": _RATE_UNITS,
}

_QUANTITY_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$"
)

# micro sign and Greek mu both spelled as "u" internally
_MU_TRANSLATION = {ord("\u00b5"): "u", ord("\u03bc"): "u"}


def parse_quantity(value, kind: str) -> float:
    r"""Convert a config entry to internal units (us, rad/us, 1/us).

    Parameters
    ----------
    value : float or str
        A bare number is taken to be already in internal units. A string must
        carry a unit, e.g. ``"32.8 kHz"``, ``"11 ns"``, ``"1/436 us"`` is *not*
        accepted -- invert a time with the ``t1_cavity`` alias instead.
    kind : {"frequency", "time", "rate"}
        ``"frequency"`` multiplies by :math:`2\pi`, since those quantities are
        stored as angular frequencies; ``"rate"`` does not.

    Examples
    --------
    >>> abs(parse_quantity("32.8 kHz", "frequency") - 2 * np.pi * 32.8e-3) < 1e-12
    True
    >>> parse_quantity("11 ns", "time")
    0.011
    """
    if isinstance(value, bool):
        raise TypeError(f"expected a number or a quantity string, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"expected a number or a quantity string, got {value!r}")

    table = _KIND_TABLES.get(kind)
    if table is None:
        raise ValueError(f"unknown quantity kind {kind!r}")

    text = value.translate(_MU_TRANSLATION)
    match = _QUANTITY_RE.match(text)
    if match is None:
        raise ValueError(f"cannot parse quantity {value!r}")
    number, unit = match.group(1), match.group(2)
    if not unit:
        raise ValueError(
            f"quantity {value!r} has no unit; give one of "
            f"{sorted(table)} or pass a bare number in internal units "
            f"({'rad/us' if kind == 'frequency' else 'us' if kind == 'time' else '1/us'})"
        )
    # case-insensitive match, but prefer the exact spelling
    if unit not in table:
        lowered = {u.lower(): u for u in table}
        if unit.lower() not in lowered:
            raise ValueError(
                f"unknown {kind} unit {unit!r} in {value!r}; expected one of {sorted(table)}"
            )
        unit = lowered[unit.lower()]
    return float(number) * table[unit]


#: field -> quantity kind, for fields that carry units
_FIELD_KINDS = {
    "chi": "frequency",
    "chi_prime": "frequency",
    "kerr": "frequency",
    "anharm": "frequency",
    "eps_max": "frequency",
    "omega_max": "frequency",
    "kappa": "rate",
    "dt": "time",
    "sigma_disp": "time",
    "sigma_qubit": "time",
}

#: field -> python type, for dimensionless fields
_FIELD_PLAIN = {
    "n_sigma_disp": int,
    "n_sigma_qubit": int,
    "n_selective_periods": int,
    "subtract_pedestal": bool,
}

#: alias -> (canonical field, transform applied after unit conversion)
_FIELD_ALIASES = {
    "chi_2": ("chi_prime", None),
    "chi2": ("chi_prime", None),
    "k_c": ("kerr", None),
    "kerr_c": ("kerr", None),
    "two_kerr": ("kerr", lambda v: 0.5 * v),  # Table S1 quotes 2 K_c
    "k_q": ("anharm", None),
    "anharmonicity": ("anharm", None),
    "t1_cavity": ("kappa", None),
    "cavity_t1": ("kappa", None),
    "t1": ("kappa", None),
    "sample_period": ("dt", None),
}

#: aliases whose value is a *time* to be inverted into a rate
_INVERTED_TIME_ALIASES = {"t1_cavity", "cavity_t1", "t1"}


def _canonical_field(key: str):
    """Resolve a config key to ``(field, kind, transform)``."""
    k = key.strip().lower()
    if k in _FIELD_KINDS:
        return k, _FIELD_KINDS[k], None
    if k in _FIELD_PLAIN:
        return k, "plain", None
    if k in _FIELD_ALIASES:
        field, transform = _FIELD_ALIASES[k]
        if k in _INVERTED_TIME_ALIASES:
            return field, "inverse_time", transform
        kind = _FIELD_KINDS.get(field, "plain")
        return field, kind, transform
    valid = sorted(set(_FIELD_KINDS) | set(_FIELD_PLAIN) | set(_FIELD_ALIASES))
    raise ValueError(f"unknown parameter {key!r}; expected one of {valid}")


def _load_raw(path: str) -> dict:
    """Read a JSON, YAML or TOML file into a dict, dispatching on extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as fh:
            return json.load(fh)
    if ext == ".toml":
        try:
            import tomllib
        except ImportError as exc:  # pragma: no cover
            raise ImportError("reading .toml needs Python 3.11+ or the tomli package") from exc
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("reading .yaml needs PyYAML (pip install pyyaml)") from exc
        with open(path) as fh:
            return yaml.safe_load(fh)
    raise ValueError(
        f"unsupported config extension {ext!r}; use .json, .toml, .yaml or .yml"
    )


def load_config(path: str, section: str = "system", **overrides):
    r"""Read a device config file, returning parameters and any extra sections.

    The ``[system]`` section (or the top level, if there is no such section)
    populates a :class:`SystemParams`. Every other top-level mapping is passed
    through untouched, which is where compile- and simulation-time settings
    naturally live.

    Returns
    -------
    params : SystemParams
    extras : dict
        The remaining top-level sections, e.g. ``extras["ecd"]["alpha0"]``.

    Examples
    --------
    A TOML file describing the device of Eickbusch et al.::

        [system]
        chi         = "32.8 kHz"     # quoted frequencies get the 2 pi
        chi_prime   = "1.5 Hz"
        two_kerr    = "1.0 Hz"       # Table S1 quotes 2 K_c
        t1_cavity   = "436 us"       # inverted into kappa
        anharm      = "193 MHz"
        eps_max     = "400 MHz"
        omega_max   = "20 MHz"
        dt          = "1 ns"
        sigma_disp  = "11 ns"
        sigma_qubit = "6 ns"

        [ecd]
        alpha0 = 30.0
        n_fock = 60

    then::

        params, extras = load_config("device.toml")
        seq = compile_ecd_sequence(betas, thetas, phis, params=params,
                                   alpha0=extras["ecd"]["alpha0"])
    """
    raw = _load_raw(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    if section in raw:
        sys_raw = raw[section]
        extras = {k: v for k, v in raw.items() if k != section}
    else:
        # flat file: split the known parameter names off from anything else
        sys_raw, extras = {}, {}
        for k, v in raw.items():
            if isinstance(v, dict):
                extras[k] = v
                continue
            try:
                _canonical_field(k)
            except ValueError:
                extras[k] = v
            else:
                sys_raw[k] = v
    if not isinstance(sys_raw, dict):
        raise ValueError(f"section [{section}] of {path} must be a mapping")
    return SystemParams.from_dict(sys_raw, **overrides), extras


# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemParams:
    r"""Hamiltonian, decoherence and drive-constraint parameters.

    Defaults are Table S1 of Eickbusch et al. Every field is overridable;
    use :meth:`replace` for one-off edits.

    Parameters
    ----------
    chi : float
        Dispersive shift in rad/us. Table S1: :math:`2\pi \times 32.8` kHz.
    chi_prime : float
        Second-order dispersive shift :math:`\chi'`, rad/us.
        Table S1: :math:`2\pi \times 1.5` Hz.
    kerr : float
        Cavity self-Kerr :math:`K_c` in rad/us, in the sign convention
        :math:`H \supset -K_c a^{\dagger 2} a^2`. Table S1 quotes
        :math:`2 K_c = 2\pi \times 1` Hz.
    kappa : float
        Cavity relaxation rate :math:`1/T_{1,c}`, rad/us. Enters the
        semiclassical trajectories as the deterministic re-centring force
        :math:`\kappa/2`; it is *not* a decoherence channel here (the
        simulation in :mod:`pulse_simulation` is unitary by default).
    anharm : float
        Transmon anharmonicity :math:`K = 2 K_q`, rad/us. Only used when the
        transmon is truncated above two levels.
    eps_max, omega_max : float
        Drive-amplitude ceilings in rad/us, used for warnings only.
    dt : float
        Waveform sample period in us. 1 ns matches the 1 GS/s DAC of the
        experiment; coarsen it for SNAP sequences, which are 10^3 times longer.
    sigma_disp, n_sigma_disp : float, int
        Gaussian standard deviation and truncation width of the cavity
        displacement pulses. Table: :math:`\sigma = 11` ns, total 4 sigma.
    sigma_qubit, n_sigma_qubit : float, int
        Same for qubit rotation pulses: :math:`\sigma = 6` ns, total 4 sigma.
    subtract_pedestal : bool
        Subtract the value of the Gaussian at the truncation point so the
        waveform starts and ends exactly at zero.
    n_selective_periods : int
        Duration of a SNAP selective :math:`\pi` pulse in units of
        :math:`2\pi/\chi`. One period is *not* enough for a number-multiplexed
        drive, whose components add coherently at the pulse centre and break
        the weak-drive approximation; see :func:`snap_waveform`. Must be an
        integer so that the per-level phase reference is common to both pulses
        of a SNAP gate.
    """

    chi: float = TWO_PI * 32.8e-3
    chi_prime: float = TWO_PI * 1.5e-6
    kerr: float = TWO_PI * 0.5e-6
    kappa: float = 1.0 / 436.0
    anharm: float = TWO_PI * 193.0
    eps_max: float = TWO_PI * 400.0
    omega_max: float = TWO_PI * 20.0
    dt: float = 1e-3
    sigma_disp: float = 11e-3
    n_sigma_disp: int = 4
    sigma_qubit: float = 6e-3
    n_sigma_qubit: int = 4
    subtract_pedestal: bool = True
    n_selective_periods: int = 4

    # -- convenience -----------------------------------------------------
    @property
    def t_disp(self) -> float:
        """Duration of one displacement pulse, us."""
        return self.n_sigma_disp * self.sigma_disp

    @property
    def t_qubit(self) -> float:
        """Duration of one qubit rotation pulse, us."""
        return self.n_sigma_qubit * self.sigma_qubit

    @property
    def t_snap_selective(self) -> float:
        """Duration of one number-selective pi pulse, us."""
        return self.n_selective_periods * TWO_PI / self.chi

    def replace(self, **kwargs) -> "SystemParams":
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    # -- configuration --------------------------------------------------
    @classmethod
    def from_dict(cls, mapping: dict, strict: bool = True, **overrides) -> "SystemParams":
        r"""Build parameters from a mapping of possibly unit-carrying entries.

        Keys are canonical field names or the aliases in ``_FIELD_ALIASES``
        (``two_kerr``, ``t1_cavity``, ``K_q``, ...). Values are either bare
        numbers in internal units (us, rad/us, 1/us) or strings with an
        explicit unit, e.g. ``"32.8 kHz"``. Frequencies acquire the factor
        :math:`2\pi`; rates such as :math:`\kappa` do not.

        Parameters
        ----------
        strict : bool
            Raise on unrecognized keys. Turning this off downgrades them to a
            warning, which is convenient when a shared device file carries
            fields this module does not use (qubit :math:`T_2`, readout
            settings, and so on).
        **overrides
            Applied after the mapping, in the same units-aware way, so
            ``from_dict(cfg, chi="40 kHz")`` works.
        """
        fields = {}
        for key, value in {**mapping, **overrides}.items():
            try:
                field_name, kind, transform = _canonical_field(key)
            except ValueError:
                if strict:
                    raise
                warnings.warn(f"ignoring unrecognized parameter {key!r}", stacklevel=2)
                continue
            if kind == "plain":
                converted = _FIELD_PLAIN[field_name](value)
            elif kind == "inverse_time":
                seconds = parse_quantity(value, "time")
                if seconds <= 0:
                    raise ValueError(f"{key} must be positive; got {value!r}")
                converted = 1.0 / seconds
            else:
                converted = parse_quantity(value, kind)
            if transform is not None:
                converted = transform(converted)
            fields[field_name] = converted
        return cls(**fields)

    @classmethod
    def from_config(cls, path: str, section: str = "system", **overrides) -> "SystemParams":
        """Read parameters from a JSON, YAML or TOML file.

        Ignores any other sections; use :func:`load_config` to get those too.
        """
        params, _ = load_config(path, section=section, **overrides)
        return params

    def to_config(self, path: str | None = None, section: str = "system") -> dict:
        """Render the parameters as a config mapping with explicit units.

        Round-trips through :meth:`from_dict`, so it is a convenient way to
        record the exact parameters a run used. Writes the file when ``path`` is
        given, dispatching on extension.
        """
        body = {
            "chi": f"{self.chi / TWO_PI * 1e3:.10g} kHz",
            "chi_prime": f"{self.chi_prime / TWO_PI * 1e6:.10g} Hz",
            "kerr": f"{self.kerr / TWO_PI * 1e6:.10g} Hz",
            "t1_cavity": f"{1.0 / self.kappa:.10g} us" if self.kappa > 0 else "inf us",
            "anharm": f"{self.anharm / TWO_PI:.10g} MHz",
            "eps_max": f"{self.eps_max / TWO_PI:.10g} MHz",
            "omega_max": f"{self.omega_max / TWO_PI:.10g} MHz",
            "dt": f"{self.dt * 1e3:.10g} ns",
            "sigma_disp": f"{self.sigma_disp * 1e3:.10g} ns",
            "sigma_qubit": f"{self.sigma_qubit * 1e3:.10g} ns",
            "n_sigma_disp": int(self.n_sigma_disp),
            "n_sigma_qubit": int(self.n_sigma_qubit),
            "n_selective_periods": int(self.n_selective_periods),
            "subtract_pedestal": bool(self.subtract_pedestal),
        }
        out = {section: body}
        if path is not None:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".json":
                with open(path, "w") as fh:
                    json.dump(out, fh, indent=2)
            elif ext in (".yaml", ".yml"):
                import yaml

                with open(path, "w") as fh:
                    yaml.safe_dump(out, fh, sort_keys=False)
            elif ext == ".toml":
                lines = [f"[{section}]"]
                for k, v in body.items():
                    if isinstance(v, bool):
                        lines.append(f"{k} = {str(v).lower()}")
                    elif isinstance(v, int):
                        lines.append(f"{k} = {v}")
                    else:
                        lines.append(f'{k} = "{v}"')
                with open(path, "w") as fh:
                    fh.write("\n".join(lines) + "\n")
            else:
                raise ValueError(f"unsupported config extension {ext!r}")
        return out

    def drift_kwargs(self) -> dict:
        r"""Arguments for :func:`hamiltonians.cavity_transmon_drift`.

        That function builds :math:`\chi\, n_c n_t + (k/2) a^{\dagger 2}a^2 +
        (\alpha/2) b^{\dagger 2} b^2`, whereas this module uses the paper's
        :math:`-\chi n_c n_t - K_c a^{\dagger 2} a^2`. The returned values
        carry the sign flips.
        """
        return {"chi": -self.chi, "k": -2.0 * self.kerr, "alpha": -self.anharm}


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _gauss_cached(sigma: float, n_sigma: int, dt: float, subtract_pedestal: bool):
    """Cached envelope. The returned array must not be mutated in place."""
    n = int(round(n_sigma * sigma / dt))
    if n < 2:
        raise ValueError(
            f"pulse of length {n_sigma * sigma * 1e3:.1f} ns is under-sampled at "
            f"dt = {dt * 1e3:.1f} ns; reduce dt or widen the pulse"
        )
    t = (np.arange(n) + 0.5) * dt
    t_c = 0.5 * n * dt
    g = np.exp(-0.5 * ((t - t_c) / sigma) ** 2)
    if subtract_pedestal:
        g = g - np.exp(-0.5 * (0.5 * n_sigma) ** 2)
        g = np.clip(g, 0.0, None)
    g.setflags(write=False)
    return g, float(np.sum(g) * dt)


def gaussian_envelope(sigma: float, n_sigma: int, dt: float, subtract_pedestal: bool = True):
    """Unit-peak truncated Gaussian sampled on a ``dt`` grid.

    Returns
    -------
    g : ndarray
        Real envelope of length ``round(n_sigma * sigma / dt)``, samples taken
        at bin centres. Writable copy of an internally cached array.
    area : float
        ``sum(g) * dt``, the integrated area used to normalize pulse strengths.
    """
    g, area = _gauss_cached(float(sigma), int(n_sigma), float(dt), bool(subtract_pedestal))
    return g.copy(), area


def rotation_waveform(theta: float, phi: float, params: SystemParams) -> np.ndarray:
    r"""Complex :math:`\Omega(t)` realizing :math:`R(\theta,\phi)`.

    Uses the convention of :func:`gate_optimization.qubit_rotation`; see the
    module docstring.
    """
    g, area = gaussian_envelope(
        params.sigma_qubit, params.n_sigma_qubit, params.dt, params.subtract_pedestal
    )
    return (0.5 * theta * np.exp(1j * phi) / area) * g.astype(complex)


# ---------------------------------------------------------------------------
# Semiclassical phase-space trajectories (supplement S3 A)
# ---------------------------------------------------------------------------


def _alpha_rhs(alpha, eps, q, p: SystemParams, delta: float):
    """Right-hand side of Eq. (S5); ``q`` is the transmon excitation, 0 or 1."""
    n = abs(alpha) ** 2
    return (
        -1j * delta * alpha
        + 2j * p.kerr * n * alpha
        - 0.5 * p.kappa * alpha
        - 1j * eps
        + 1j * q * (p.chi + 2.0 * p.chi_prime * n) * alpha
    )


def _integrate_alpha(eps, q_of_t, p: SystemParams, delta: float, alpha0=0.0j, stride: int = 1):
    """RK4-integrate Eq. (S5) on the waveform grid, vectorized over branches.

    Parameters
    ----------
    eps : ndarray, complex
        Cavity drive samples, length ``n``.
    q_of_t : ndarray
        Transmon excitation indicator (0.0 or 1.0). Shape ``(n,)`` for a single
        branch, or ``(n, m)`` to integrate ``m`` conditional branches at once
        (the two ECD branches are always solved together, which halves the
        Python-level overhead of this loop).
    alpha0 : complex or ndarray
        Initial condition, broadcast against the branch axis.
    stride : int
        Integrate on a grid coarsened by this factor, block-averaging the
        drive so pulse areas are preserved. Used to speed up the inner
        Nelder-Mead of the ECD compiler; final trajectories use ``stride=1``.

    Returns
    -------
    ndarray, complex
        Trajectory of shape ``(n + 1,)`` or ``(n + 1, m)``; element ``k`` is
        alpha at time ``k * dt``, so element 0 is ``alpha0`` and element ``n``
        is the value after the whole waveform.
    """
    eps = np.asarray(eps, dtype=complex)
    q_of_t = np.asarray(q_of_t, dtype=float)
    squeeze = q_of_t.ndim == 1
    if squeeze:
        q_of_t = q_of_t[:, None]

    stride = int(stride)
    if stride > 1:
        # Block-average rather than decimate, so pulse areas are preserved.
        n_blk = eps.size // stride
        keep = n_blk * stride
        eps = eps[:keep].reshape(n_blk, stride).mean(axis=1)
        q_of_t = q_of_t[:keep].reshape(n_blk, stride, -1).mean(axis=1)
    dt = p.dt * stride
    n = eps.size
    m = q_of_t.shape[1]

    # Midpoint samples by linear interpolation; endpoints held.
    eps_l = eps.tolist()
    eps_r = eps_l[1:] + eps_l[-1:]
    eps_m = [0.5 * (a + b) for a, b in zip(eps_l, eps_r)]
    q_cols = [q_of_t[:, j].tolist() for j in range(m)]

    # Hoisted scalars: the inner loop is pure Python complex arithmetic, which
    # is several times faster than numpy scalar ops at this array size.
    delta_c = 1j * delta
    kerr2 = 2j * p.kerr
    half_kappa = 0.5 * p.kappa
    chi = p.chi
    chi_p2 = 2.0 * p.chi_prime
    dt6 = dt / 6.0
    dt2 = 0.5 * dt

    out = np.empty((n + 1, m), dtype=complex)
    a0_arr = np.broadcast_to(np.asarray(alpha0, dtype=complex), (m,))

    for j in range(m):
        q_l = q_cols[j]
        q_r = q_l[1:] + q_l[-1:]
        q_m = [0.5 * (x + y) for x, y in zip(q_l, q_r)]
        a = complex(a0_arr[j])
        col = out[:, j]
        col[0] = a

        for k in range(n):
            e0 = eps_l[k]
            em = eps_m[k]
            e1 = eps_r[k]
            q0 = q_l[k]
            qm = q_m[k]
            q1 = q_r[k]

            nn = a.real * a.real + a.imag * a.imag
            k1 = -delta_c * a + kerr2 * nn * a - half_kappa * a - 1j * e0 + 1j * q0 * (chi + chi_p2 * nn) * a
            b = a + dt2 * k1
            nn = b.real * b.real + b.imag * b.imag
            k2 = -delta_c * b + kerr2 * nn * b - half_kappa * b - 1j * em + 1j * qm * (chi + chi_p2 * nn) * b
            c = a + dt2 * k2
            nn = c.real * c.real + c.imag * c.imag
            k3 = -delta_c * c + kerr2 * nn * c - half_kappa * c - 1j * em + 1j * qm * (chi + chi_p2 * nn) * c
            d = a + dt * k3
            nn = d.real * d.real + d.imag * d.imag
            k4 = -delta_c * d + kerr2 * nn * d - half_kappa * d - 1j * e1 + 1j * q1 * (chi + chi_p2 * nn) * d

            a = a + dt6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            col[k + 1] = a

    return out[:, 0] if squeeze else out


def frame_trajectory(eps, params: SystemParams, delta: float, alpha0: complex = 0.0j):
    r"""Classical response :math:`\alpha(t)` of Eq. (S3), transmon in ground.

    This is the trajectory that defines the displaced frame used by
    :mod:`pulse_simulation`: with this choice the linear-in-:math:`a` term of
    Eq. (S2) cancels for the transmon ground state, and the residual linear
    term proportional to :math:`q^\dagger q` is exactly the conditional force
    that generates the ECD gate.

    Returns an array of length ``len(eps) + 1``.
    """
    return _integrate_alpha(eps, np.zeros(len(eps)), params, delta, alpha0)


def branch_trajectories(eps, q_g, params: SystemParams, delta: float, stride: int = 1):
    r"""Conditional trajectories :math:`\alpha_g(t)`, :math:`\alpha_e(t)`.

    Parameters
    ----------
    q_g : ndarray
        Transmon excitation indicator for the branch that *starts* in
        :math:`|g\rangle`. A qubit :math:`\pi` pulse is modelled as an
        instantaneous flip, so this is a step function. The branch starting in
        :math:`|e\rangle` uses ``1 - q_g``.
    """
    q_g = np.asarray(q_g, dtype=float)
    traj = _integrate_alpha(
        eps, np.stack([q_g, 1.0 - q_g], axis=1), params, delta, stride=stride
    )
    return traj[:, 0], traj[:, 1]


def _geometric_phase(eps, a_g, a_e, params: SystemParams):
    r"""Qubit phase :math:`\theta'` and residual displacement :math:`\lambda`.

    Implements Eqs. (S21)-(S25) with :math:`\delta = (\alpha_g - \alpha_e)/2`
    and :math:`\gamma = (\alpha_g + \alpha_e)/2`, i.e. the sign convention of
    Eq. (S25). The Baker-Campbell-Hausdorff separation of the joint
    displacement contributes
    :math:`\theta' = \theta(T) + 2\,\mathrm{Im}[\gamma^*\delta]`.
    """
    eps = np.asarray(eps, dtype=complex)
    delta_t = 0.5 * (a_g - a_e)
    gamma_t = 0.5 * (a_g + a_e)
    # trapezoid on the n+1 point trajectory against the n point drive
    integrand = np.real(np.conj(eps) * delta_t[:-1])
    integrand_next = np.real(np.conj(eps) * delta_t[1:])
    theta = -2.0 * float(np.sum(0.5 * (integrand + integrand_next)) * params.dt)
    beta = a_g[-1] - a_e[-1]
    lam = gamma_t[-1]
    theta_prime = theta + 2.0 * float(np.imag(np.conj(lam) * delta_t[-1]))
    return complex(beta), complex(lam), float(theta_prime)


# ---------------------------------------------------------------------------
# ECD pulse construction (supplement S4 B)
# ---------------------------------------------------------------------------


@dataclass
class ECDPulse:
    """One compiled echoed conditional displacement.

    Attributes
    ----------
    eps, omega : ndarray, complex
        Cavity and qubit drives on the ``params.dt`` grid, equal length.
    beta : complex
        Realized conditional displacement, ``alpha_g(T) - alpha_e(T)``.
    lam : complex
        Residual net displacement (driven to zero by the cost function).
    theta_prime : float
        Accumulated qubit phase, absorbed into later rotation phases.
    alpha0 : float
        Intermediate phase-space radius actually used.
    t_wait : float
        Wait time between displacement pulses, us.
    ratios : ndarray
        ``[eps0, r2, r3, r4]`` of Fig. S3a.
    cost : float
        Final value of the cost function (S27).
    alpha_g, alpha_e : ndarray
        Semiclassical trajectories, length ``len(eps) + 1``.
    """

    eps: np.ndarray
    omega: np.ndarray
    beta: complex
    lam: complex
    theta_prime: float
    alpha0: float
    t_wait: float
    ratios: np.ndarray
    cost: float
    alpha_g: np.ndarray = field(repr=False, default=None)
    alpha_e: np.ndarray = field(repr=False, default=None)

    @property
    def duration(self) -> float:
        return len(self.eps) * 0.0 if self.eps is None else len(self.eps)

    def summary(self) -> str:
        return (
            f"beta = {self.beta:+.4f} (|beta| = {abs(self.beta):.4f})   "
            f"alpha0 = {self.alpha0:.2f}   t_w = {self.t_wait * 1e3:.1f} ns   "
            f"|lambda| = {abs(self.lam):.2e}   theta' = {self.theta_prime:+.4f} rad"
        )


def _ecd_waveforms(x, t_wait, params: SystemParams, pi_phase: float = 0.0):
    """Assemble the Fig. S3a drive from ``x = [eps0, r2, r3, r4]``.

    Layout, with ``tp`` the displacement duration and ``tq`` the pi-pulse
    duration::

        [ +eps0 g ][ wait tw ][ -r2 g ][ pi pulse tq ][ -r3 g ][ wait tw ][ +r4 g ]

    Returns ``(eps, omega, q_g)`` where ``q_g`` is the step function used by
    :func:`branch_trajectories`.
    """
    eps0, r2, r3, r4 = x
    g, _ = _gauss_cached(
        params.sigma_disp, params.n_sigma_disp, params.dt, params.subtract_pedestal
    )
    n_wait = int(round(t_wait / params.dt))
    zw = np.zeros(n_wait)
    pi_pulse = rotation_waveform(np.pi, pi_phase, params)
    n_q = pi_pulse.size

    eps = np.concatenate(
        [eps0 * g, zw, -eps0 * r2 * g, np.zeros(n_q), -eps0 * r3 * g, zw, eps0 * r4 * g]
    ).astype(complex)
    omega = np.concatenate(
        [
            np.zeros(g.size + n_wait + g.size, dtype=complex),
            pi_pulse,
            np.zeros(g.size + n_wait + g.size, dtype=complex),
        ]
    )
    # Instantaneous flip at the centre of the pi pulse.
    q_g = np.zeros(eps.size)
    i_flip = g.size + n_wait + g.size + n_q // 2
    q_g[i_flip:] = 1.0
    return eps, omega, q_g


def _ecd_markers(g_size: int, n_wait: int, n_q: int):
    """Sample indices of the four cost-function probe points."""
    i_quarter = g_size + n_wait // 2
    i_mid = g_size + n_wait + g_size + n_q // 2
    i_three_quarter = g_size + n_wait + g_size + n_q + g_size + n_wait // 2
    i_end = 2 * (g_size + n_wait + g_size) + n_q
    return i_quarter, i_mid, i_three_quarter, i_end


def _ecd_cost(x, t_wait, alpha0, params: SystemParams, delta: float, stride: int = 1):
    """Cost function (S27): null net displacement, hit radius ``alpha0``."""
    eps, _, q_g = _ecd_waveforms(x, t_wait, params)
    a_g, a_e = branch_trajectories(eps, q_g, params, delta, stride=stride)
    g, _ = _gauss_cached(
        params.sigma_disp, params.n_sigma_disp, params.dt, params.subtract_pedestal
    )
    gq, _ = _gauss_cached(
        params.sigma_qubit, params.n_sigma_qubit, params.dt, params.subtract_pedestal
    )
    n_wait = int(round(t_wait / params.dt))
    i_q, i_m, i_3q, i_e = _ecd_markers(g.size, n_wait, gq.size)
    if stride > 1:
        i_q, i_m, i_3q = i_q // stride, i_m // stride, i_3q // stride
        i_e = a_g.size - 1
    return (
        abs(a_g[i_m] + a_e[i_m]) ** 2
        + abs(a_g[i_e] + a_e[i_e]) ** 2
        + (0.5 * abs(a_g[i_q] + a_e[i_q]) - alpha0) ** 2
        + (0.5 * abs(a_g[i_3q] + a_e[i_3q]) - alpha0) ** 2
    )


def _solve_ratios(t_wait, alpha0, params: SystemParams, delta: float, x0=None, maxiter=2000,
                  xatol=1e-9, fatol=1e-13, stride: int = 1):
    """Nelder-Mead on ``[eps0, r2, r3, r4]`` at fixed ``t_wait`` and radius."""
    _, area = _gauss_cached(
        params.sigma_disp, params.n_sigma_disp, params.dt, params.subtract_pedestal
    )
    if x0 is None:
        x0 = np.array([alpha0 / area, 1.0, 1.0, 1.0])
    res = minimize(
        _ecd_cost,
        x0,
        args=(t_wait, alpha0, params, delta, stride),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": xatol, "fatol": fatol},
    )
    return res.x, float(res.fun)


def _build_ecd(x, t_wait, alpha0, cost, params, delta, pi_phase=0.0) -> ECDPulse:
    eps, omega, q_g = _ecd_waveforms(x, t_wait, params, pi_phase)
    a_g, a_e = branch_trajectories(eps, q_g, params, delta)
    beta, lam, theta_prime = _geometric_phase(eps, a_g, a_e, params)
    return ECDPulse(
        eps=eps,
        omega=omega,
        beta=beta,
        lam=lam,
        theta_prime=theta_prime,
        alpha0=float(alpha0),
        t_wait=float(t_wait),
        ratios=np.asarray(x, dtype=float),
        cost=float(cost),
        alpha_g=a_g,
        alpha_e=a_e,
    )


def optimize_ecd_pulse(
    beta_target: complex,
    alpha0: float,
    params: SystemParams,
    delta: float | None = None,
    tol: float = 1e-4,
    stride: int = 2,
    verbose: bool = False,
) -> ECDPulse:
    r"""Compile the fastest pulse realizing :math:`\mathrm{ECD}(\beta)`.

    The trajectory ODE (S5) is covariant under a global phase rotation
    :math:`\alpha \to e^{i\varphi}\alpha`, :math:`\varepsilon \to
    e^{i\varphi}\varepsilon` (Kerr and :math:`\chi'` enter only through
    :math:`|\alpha|^2`). The search therefore targets
    :math:`|\beta_\mathrm{target}|` with a real drive and rotates the finished
    waveform by :math:`\arg\beta_\mathrm{target} - \arg\beta_\mathrm{achieved}`,
    which leaves the cost and :math:`\theta'` invariant.

    Following section S4 B: for a fixed intermediate radius :math:`\alpha_0`,
    :math:`|\beta|` grows monotonically with the wait time :math:`t_w`, so
    :math:`t_w` is root-found against the target. If even :math:`t_w = 0`
    overshoots -- the drive-constrained regime of Fig. 2c, where the gate is as
    short as the constituent pulses allow -- :math:`\alpha_0` is lowered
    instead.

    Parameters
    ----------
    beta_target : complex
        Desired conditional displacement.
    alpha0 : float
        Target intermediate phase-space radius. Larger is faster but populates
        higher Fock levels; the experiment used 30.
    delta : float, optional
        Cavity drive detuning. Defaults to ``params.chi / 2``.
    tol : float
        Relative tolerance on :math:`|\beta|`.
    stride : int
        Grid coarsening used inside the root find, for speed. The ratios are
        always re-solved on the full grid before the pulse is returned, so this
        affects run time and not the reported :math:`\beta`, :math:`\lambda`
        or :math:`\theta'`.
    """
    if delta is None:
        delta = 0.5 * params.chi
    target = abs(complex(beta_target))
    if target < 1e-12:
        raise ValueError("beta_target must be nonzero")

    state = {"x": None, "radius": float(alpha0)}

    def solve(t_wait, radius, stride_, maxiter):
        x0 = state["x"]
        if x0 is not None:
            x0 = np.asarray(x0, dtype=float).copy()
            x0[0] *= radius / max(state["radius"], 1e-12)
        x, cost = _solve_ratios(
            t_wait, radius, params, delta, x0=x0, maxiter=maxiter, stride=stride_
        )
        state["x"], state["radius"] = x, radius
        return _build_ecd(x, t_wait, radius, cost, params, delta)

    # --- initial guess from the ideal relation |beta| ~ 2 alpha0 sin(chi T / 2)
    radius = float(alpha0)
    t_pulses = 4 * params.t_disp + params.t_qubit
    arg = np.clip(target / (2.0 * radius), 0.0, 0.99)
    t_wait_hi = max(0.5 * (2.0 * np.arcsin(arg) / params.chi - t_pulses), 10.0 * params.dt)

    pulse_hi = solve(t_wait_hi, radius, stride, 2000)
    for _ in range(20):
        if abs(pulse_hi.beta) >= target:
            break
        t_wait_hi *= 2.0
        pulse_hi = solve(t_wait_hi, radius, stride, 600)
    if abs(pulse_hi.beta) < target:
        raise RuntimeError(
            f"cannot reach |beta| = {target:.3f} at alpha0 = {radius:.1f}: best "
            f"{abs(pulse_hi.beta):.3f} at t_w = {t_wait_hi * 1e3:.0f} ns. "
            "Increase alpha0, or check that chi has the right units (rad/us)."
        )

    pulse_lo = solve(0.0, radius, stride, 600)

    if abs(pulse_lo.beta) > target:
        # --- drive-constrained: t_w = 0 and shrink the radius
        best = _bisect(
            lambda r: solve(0.0, r, stride, 400),
            lambda pl: pl.alpha0,
            0.0,
            radius,
            target,
            tol,
        )
        best = _refine(best, target, params, delta, state, tol, radius, stride=stride)
        if verbose:
            print(f"  drive-limited: alpha0 {radius:.2f} -> {best.alpha0:.2f}")
            print(f"  {best.summary()}")
        return _phase_rotate(best, beta_target)

    # --- shorten the wait time until |beta| lands on the target
    best = _bisect(
        lambda tw: solve(tw, radius, stride, 400),
        lambda pl: pl.t_wait,
        0.0,
        t_wait_hi,
        target,
        tol,
    )
    best = _refine(best, target, params, delta, state, tol, radius, stride=stride)
    if verbose:
        print(f"  {best.summary()}")
    return _phase_rotate(best, beta_target)


def _bisect(evaluate, knob_of, lo, hi, target, tol, max_iter=20):
    """Bisect a monotone knob (wait time or radius) so that ``|beta| = target``."""
    best = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cand = evaluate(mid)
        best = cand
        if abs(abs(cand.beta) - target) <= tol * target:
            break
        if abs(cand.beta) > target:
            hi = mid
        else:
            lo = mid
    return best


def _refine(
    pulse: ECDPulse,
    target: float,
    params: SystemParams,
    delta: float,
    state: dict,
    tol: float,
    alpha0_max: float,
    stride: int = 2,
    max_iter: int = 8,
) -> ECDPulse:
    r"""Land exactly on the target :math:`|\beta|` using two knobs.

    The wait time is an integer number of DAC samples, and one sample is worth
    roughly a percent of :math:`|\beta|`, so it cannot hit an arbitrary target
    on its own. The fix is the one the experiment uses (see the paragraph above
    Fig. S4): quantize :math:`t_w` to the shortest sample count that *reaches*
    the target at the requested radius, then trim the radius down until
    :math:`|\beta|` matches. Gates therefore run at
    :math:`\alpha_0' \le \alpha_0`.
    """
    dt = params.dt

    def solve(t_wait, radius, stride_, maxiter=600):
        x0 = np.asarray(state["x"], dtype=float).copy()
        x0[0] *= radius / max(state["radius"], 1e-12)
        x, cost = _solve_ratios(
            t_wait, radius, params, delta, x0=x0, maxiter=maxiter, stride=stride_
        )
        state["x"], state["radius"] = x, radius
        return _build_ecd(x, t_wait, radius, cost, params, delta)

    # --- 1. smallest integer sample count whose |beta| reaches the target
    n_w = int(round(pulse.t_wait / dt))
    cand = solve(n_w * dt, alpha0_max, stride)
    guard = 0
    while abs(cand.beta) < target and guard < 200:
        n_w += 1
        cand = solve(n_w * dt, alpha0_max, stride)
        guard += 1
    while n_w > 0 and guard < 200:
        trial = solve((n_w - 1) * dt, alpha0_max, stride)
        if abs(trial.beta) < target:
            break
        n_w -= 1
        cand = trial
        guard += 1

    t_wait = n_w * dt

    # --- 2. secant on the radius at fixed wait time, on the full grid
    r_hi = float(alpha0_max)
    p_hi = solve(t_wait, r_hi, 1)
    if abs(abs(p_hi.beta) - target) <= tol * target or abs(p_hi.beta) < target:
        return p_hi

    best = p_hi
    r_prev, f_prev = r_hi, abs(p_hi.beta) - target
    r_cur = r_hi * max(target / abs(p_hi.beta), 0.05)
    for _ in range(max_iter):
        cand = solve(t_wait, r_cur, 1)
        f_cur = abs(cand.beta) - target
        if abs(f_cur) < abs(abs(best.beta) - target):
            best = cand
        if abs(f_cur) <= tol * target:
            break
        if abs(f_cur - f_prev) < 1e-14:
            break
        r_new = r_cur - f_cur * (r_cur - r_prev) / (f_cur - f_prev)
        r_prev, f_prev = r_cur, f_cur
        r_cur = float(np.clip(r_new, 1e-3, alpha0_max))
    return best


def _phase_rotate(pulse: ECDPulse, beta_target: complex) -> ECDPulse:
    """Rotate a compiled pulse so that ``arg(beta)`` matches the target."""
    phi = np.angle(complex(beta_target)) - np.angle(pulse.beta)
    rot = np.exp(1j * phi)
    return ECDPulse(
        eps=pulse.eps * rot,
        omega=pulse.omega,
        beta=pulse.beta * rot,
        lam=pulse.lam * rot,
        theta_prime=pulse.theta_prime,
        alpha0=pulse.alpha0,
        t_wait=pulse.t_wait,
        ratios=pulse.ratios,
        cost=pulse.cost,
        alpha_g=pulse.alpha_g * rot,
        alpha_e=pulse.alpha_e * rot,
    )


def ecd_pulse_cache(params: SystemParams, alpha0: float, delta: float | None = None, **kwargs):
    r"""Memoized ``|beta| -> ECDPulse`` factory.

    Compiling an ECD gate costs a few hundred ODE solves, and a depth-9
    sequence often contains several :math:`\beta` of nearly equal magnitude.
    Since the pulse for any :math:`\beta` follows from the one for
    :math:`|\beta|` by a global phase rotation, magnitudes are cached to a
    relative tolerance.
    """
    store: list[tuple[float, ECDPulse]] = []
    rtol = kwargs.pop("cache_rtol", 1e-6)

    def get(beta: complex) -> ECDPulse:
        mag = abs(complex(beta))
        for m, pulse in store:
            if abs(m - mag) <= rtol * max(m, 1e-12):
                return _phase_rotate(pulse, beta)
        pulse = optimize_ecd_pulse(mag, alpha0, params, delta=delta, **kwargs)
        store.append((mag, pulse))
        return _phase_rotate(pulse, beta)

    return get


# ---------------------------------------------------------------------------
# Assembled sequences
# ---------------------------------------------------------------------------


@dataclass
class PulseSegment:
    """Provenance of one slice of a compiled waveform."""

    kind: str  # "rotation" | "ecd" | "displacement" | "snap" | "idle"
    start: int
    stop: int
    label: str = ""
    meta: dict = field(default_factory=dict)

    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass
class PulseSequence:
    r"""A compiled control sequence: :math:`\varepsilon(t)` and :math:`\Omega(t)`.

    Attributes
    ----------
    eps, omega : ndarray, complex
        Cavity and transmon drives in rad/us on a uniform ``dt`` grid.
    params : SystemParams
    delta : float
        Cavity drive detuning used during compilation. The simulation must use
        the same value.
    segments : list of PulseSegment
        Gate boundaries, for slicing the trajectory and for plotting.
    gate_indices : ndarray
        Sample index at the *end* of every logical gate, i.e. the times at
        which the simulated state should be compared with the ideal circuit's
        intermediate states.
    meta : dict
    """

    eps: np.ndarray
    omega: np.ndarray
    params: SystemParams
    delta: float
    segments: list = field(default_factory=list)
    gate_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    meta: dict = field(default_factory=dict)

    # -- basics ----------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.eps.size)

    @property
    def dt(self) -> float:
        return self.params.dt

    @property
    def duration(self) -> float:
        """Total sequence duration in us."""
        return self.n_samples * self.params.dt

    @property
    def t(self) -> np.ndarray:
        """Sample times (bin centres) in us."""
        return (np.arange(self.n_samples) + 0.5) * self.params.dt

    def frame_trajectory(self, alpha_init: complex = 0.0j) -> np.ndarray:
        r"""The displaced-frame trajectory :math:`\alpha(t)`, length ``n+1``."""
        return frame_trajectory(self.eps, self.params, self.delta, alpha_init)

    def peak_amplitudes(self) -> tuple[float, float]:
        return float(np.abs(self.eps).max()), float(np.abs(self.omega).max())

    def check_drive_limits(self, warn: bool = True) -> dict:
        eps_pk, om_pk = self.peak_amplitudes()
        info = {
            "eps_peak": eps_pk,
            "omega_peak": om_pk,
            "eps_limit": self.params.eps_max,
            "omega_limit": self.params.omega_max,
            "eps_ok": eps_pk <= self.params.eps_max,
            "omega_ok": om_pk <= self.params.omega_max,
        }
        if warn and not info["eps_ok"]:
            warnings.warn(
                f"peak cavity drive {eps_pk / TWO_PI:.1f} MHz exceeds the "
                f"{self.params.eps_max / TWO_PI:.1f} MHz limit; reduce alpha0",
                stacklevel=2,
            )
        if warn and not info["omega_ok"]:
            warnings.warn(
                f"peak transmon drive {om_pk / TWO_PI:.1f} MHz exceeds the "
                f"{self.params.omega_max / TWO_PI:.1f} MHz limit",
                stacklevel=2,
            )
        return info

    def summary(self) -> str:
        eps_pk, om_pk = self.peak_amplitudes()
        alpha = self.frame_trajectory()
        n_gates = sum(1 for s in self.segments if s.kind in ("ecd", "snap"))
        lines = [
            f"gate set        : {self.meta.get('gate_set', '?')}",
            f"duration        : {self.duration:.4f} us  ({self.n_samples} samples "
            f"at {self.dt * 1e3:.1f} ns)",
            f"2 pi / chi       : {TWO_PI / self.params.chi:.4f} us  "
            f"(speedup {TWO_PI / self.params.chi / self.duration:.1f}x)",
            f"gates           : {n_gates}  ({len(self.segments)} segments)",
            f"peak |eps|      : {eps_pk / TWO_PI:.2f} MHz",
            f"peak |Omega|    : {om_pk / TWO_PI:.2f} MHz",
            f"max |alpha|     : {np.abs(alpha).max():.2f}  "
            f"(max n = {np.abs(alpha).max() ** 2:.0f})",
            f"delta           : {self.delta / TWO_PI * 1e3:.2f} kHz",
        ]
        return "\n".join(lines)

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        """Write to ``.npz``, matching the checkpointing convention elsewhere."""
        seg = np.array(
            [(s.kind, s.start, s.stop, s.label) for s in self.segments],
            dtype=object,
        )
        np.savez(
            path,
            eps=self.eps,
            omega=self.omega,
            delta=self.delta,
            gate_indices=self.gate_indices,
            segments=seg,
            params=np.array([self.params.to_dict()], dtype=object),
            meta=np.array([self.meta], dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "PulseSequence":
        d = np.load(path, allow_pickle=True)
        params = SystemParams(**d["params"][0])
        segments = [
            PulseSegment(kind=str(k), start=int(a), stop=int(b), label=str(lbl))
            for k, a, b, lbl in d["segments"]
        ]
        return cls(
            eps=d["eps"],
            omega=d["omega"],
            params=params,
            delta=float(d["delta"]),
            segments=segments,
            gate_indices=d["gate_indices"],
            meta=dict(d["meta"][0]),
        )


class _Builder:
    """Accumulator that keeps ``eps``, ``omega`` and segment bookkeeping aligned."""

    def __init__(self):
        self.eps: list[np.ndarray] = []
        self.omega: list[np.ndarray] = []
        self.segments: list[PulseSegment] = []
        self.gate_indices: list[int] = []
        self.n = 0

    def add(self, eps, omega, kind: str, label: str = "", meta=None, mark_gate=True):
        eps = np.asarray(eps, dtype=complex)
        omega = np.asarray(omega, dtype=complex)
        if eps.size != omega.size:
            raise ValueError("eps and omega segments must have equal length")
        self.eps.append(eps)
        self.omega.append(omega)
        self.segments.append(
            PulseSegment(kind=kind, start=self.n, stop=self.n + eps.size, label=label, meta=meta or {})
        )
        self.n += eps.size
        if mark_gate:
            self.gate_indices.append(self.n)

    def idle(self, n_samples: int, label: str = "idle"):
        z = np.zeros(int(n_samples), dtype=complex)
        self.add(z, z, "idle", label, mark_gate=False)

    def finish(self, params, delta, meta) -> PulseSequence:
        return PulseSequence(
            eps=np.concatenate(self.eps) if self.eps else np.zeros(0, dtype=complex),
            omega=np.concatenate(self.omega) if self.omega else np.zeros(0, dtype=complex),
            params=params,
            delta=delta,
            segments=self.segments,
            gate_indices=np.asarray(self.gate_indices, dtype=int),
            meta=meta,
        )


# ---------------------------------------------------------------------------
# ECD sequence
# ---------------------------------------------------------------------------


def compile_ecd_sequence(
    betas,
    thetas,
    phis,
    alpha0: float = 30.0,
    params: SystemParams | None = None,
    delta: float | None = None,
    frame_sign: float = 1.0,
    final_displacement: bool = False,
    verbose: bool = False,
    **kwargs,
) -> PulseSequence:
    r"""Compile ``R(theta_1,phi_1) -> ECD(beta_1) -> ... -> R(theta_{N+1},phi_{N+1})``.

    Matches the gate ordering of :func:`gate_optimization._build_ecd_sequence`:
    the first rotation acts first, then ECD 1, and so on, with a trailing
    rotation. Time order in the returned waveform is therefore
    ``R_1, ECD_1, R_2, ECD_2, ..., ECD_N, R_{N+1}``.

    Parameters
    ----------
    betas : array_like, complex, shape (N,)
        From ``result.params["betas"]``.
    thetas, phis : array_like, shape (N+1,)
        From ``result.params["thetas"]`` and ``["phis"]``.
    alpha0 : float
        Intermediate phase-space radius for every ECD gate. The compiler
        lowers it per gate when the drive constraint binds.
    frame_sign : float
        Sign of the accumulated :math:`\theta'` in the virtual-Z bookkeeping.
        ``+1`` is the convention verified in :mod:`validate_pulses`; flip it
        only if you change the qubit basis ordering.
    final_displacement : bool
        Whether to append :math:`D(\beta_{N+1}/2)` when ``betas`` has ``N+1``
        entries (the optimizer's trailing displacement, usually zero).

    Notes
    -----
    Each ECD gate imparts an extra qubit phase :math:`\theta'` (Eq. S21),
    so the realized gate is :math:`\mathrm{ECD}(\beta) Z(\theta')` with
    :math:`Z(a) = \mathrm{diag}(e^{ia/2}, e^{-ia/2})`. It is removed by
    *virtual Z*: the phase of each rotation pulse is shifted by a running frame
    phase :math:`F`, costing no pulse time, exactly as described in the
    paragraph above Fig. S4.

    The frame does not simply accumulate. Because
    :math:`Z(a)\sigma_x = \sigma_x Z(-a)` and every ECD contains a
    :math:`\pi` pulse, commuting the leftover Z past a gate flips its sign, so
    the update is

    .. math::
        \varphi_i \to \varphi_i - F, \qquad F \to -F + \theta'_i .

    With this rule the compiled sequence reproduces the ideal circuit's
    :math:`|g\rangle`-projected cavity state exactly, and its
    :math:`P(|g\rangle)`, for arbitrary :math:`\theta'` (verified to machine
    precision in :func:`validate_pulses.test_frame_rule`). A leftover Z remains
    on the qubit at the end, which is a global phase once the qubit is
    projected onto :math:`|g\rangle` -- the postselection the experiment
    performs anyway.
    """
    params = params or SystemParams()
    if delta is None:
        delta = 0.5 * params.chi
    betas = np.atleast_1d(np.asarray(betas, dtype=complex))
    thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
    phis = np.atleast_1d(np.asarray(phis, dtype=float))

    n_ecd = betas.size
    trailing_beta = None
    if thetas.size == betas.size:  # no trailing displacement provided
        pass
    elif thetas.size == betas.size + 1:
        pass
    else:
        raise ValueError(
            f"expected len(thetas) == len(betas) or len(betas) + 1; "
            f"got {thetas.size} and {betas.size}"
        )
    if final_displacement and betas.size == thetas.size:
        trailing_beta = betas[-1]
        betas = betas[:-1]
        n_ecd = betas.size

    if phis.size != thetas.size:
        raise ValueError("thetas and phis must have equal length")

    make_pulse = ecd_pulse_cache(params, alpha0, delta=delta, **kwargs)
    b = _Builder()
    frame = 0.0
    pulses: list[ECDPulse] = []

    def add_rotation(i):
        theta, phi = float(thetas[i]), float(phis[i]) - frame
        om = rotation_waveform(theta, phi, params)
        b.add(
            np.zeros_like(om),
            om,
            "rotation",
            f"R({theta / np.pi:+.3f}pi, {phi / np.pi:+.3f}pi)",
            meta={"theta": theta, "phi": phi, "phi_ideal": float(phis[i]), "frame": frame},
        )

    for i in range(n_ecd):
        if i < thetas.size:
            add_rotation(i)
        if verbose:
            print(f"ECD {i + 1}/{n_ecd}: beta = {betas[i]:+.4f}")
        p = make_pulse(betas[i])
        pulses.append(p)
        b.add(
            p.eps,
            p.omega,
            "ecd",
            f"ECD({p.beta:+.3f})",
            meta={
                "beta_target": betas[i],
                "beta": p.beta,
                "lam": p.lam,
                "theta_prime": p.theta_prime,
                "alpha0": p.alpha0,
                "t_wait": p.t_wait,
            },
        )
        frame = -frame + frame_sign * p.theta_prime

    for i in range(n_ecd, thetas.size):
        add_rotation(i)

    if trailing_beta is not None and abs(trailing_beta) > 0:
        # Unconditional displacement D(beta/2): a single Gaussian.
        g, area = gaussian_envelope(
            params.sigma_disp, params.n_sigma_disp, params.dt, params.subtract_pedestal
        )
        eps = (1j * 0.5 * trailing_beta / area) * g  # alpha = -i * integral(eps)
        b.add(eps, np.zeros_like(eps), "displacement", f"D({0.5 * trailing_beta:+.3f})")

    seq = b.finish(
        params,
        delta,
        {
            "gate_set": "ecd",
            "n_ecd": n_ecd,
            "alpha0_request": alpha0,
            "frame_sign": frame_sign,
            "betas_target": betas,
            "betas_realized": np.array([p.beta for p in pulses]),
            "theta_primes": np.array([p.theta_prime for p in pulses]),
            "alpha0_used": np.array([p.alpha0 for p in pulses]),
            "t_waits": np.array([p.t_wait for p in pulses]),
            "residual_lambda": np.array([p.lam for p in pulses]),
            "thetas": thetas,
            "phis": phis,
        },
    )
    seq.check_drive_limits()
    return seq


# ---------------------------------------------------------------------------
# SNAP sequence
# ---------------------------------------------------------------------------


def snap_waveform(
    phases,
    params: SystemParams,
    t_selective: float | None = None,
    n_drive: int | None = None,
    detuning_sign: float = 1.0,
    match_padded_identity: bool = True,
):
    r"""Two multiplexed selective :math:`\pi` pulses realizing :math:`S(\vec\theta)`.

    The drive is a sum of components, one per Fock level, each detuned onto that
    level's qubit transition:

    .. math::
        \Omega(t) = \frac{\pi}{2}\frac{g(t)}{\int g}
                    \sum_n e^{i\varphi_n} e^{+i\chi n (t - t_c)} .

    With the Hamiltonian convention :math:`-\chi a^\dagger a\, q^\dagger q` the
    :math:`|g,n\rangle \to |e,n\rangle` transition sits at detuning
    :math:`-\chi n`, and a drive resonant with detuning :math:`\delta_n` must
    carry :math:`e^{-i\delta_n t}`. Hence the **positive** carrier sign; the
    opposite sign leaves only :math:`n = 0` resonant, which shows up as high
    Fock levels passing through the gate untouched.

    Two :math:`\pi` pulses with per-level phases :math:`\varphi_n^{(1)} = 0` and
    :math:`\varphi_n^{(2)} = -\theta_n` return the qubit to :math:`|g\rangle`
    having imparted :math:`-e^{i\theta_n}` on level :math:`n`.

    That minus sign is the single most common way to get a wrong answer here.
    It is a *global* phase only if every populated level is driven; a level left
    undriven acquires :math:`+1` instead, so it sits :math:`\pi` out of phase
    with the driven block. Because
    :func:`gate_optimization._build_snap_sequence` pads unoptimized levels with
    :math:`\theta_n = 0`, i.e. with :math:`+1`, the ideal gate the optimizer
    solved for and the pulse as naively built differ by :math:`\pi` on the
    entire tail above ``n_drive``. With 20% of the population up there the
    fidelity collapses to about 0.34.

    ``match_padded_identity=True`` (the default) fixes this by adding
    :math:`\pi` to every second-pulse phase, so driven levels acquire
    :math:`+e^{i\theta_n}` and undriven ones :math:`+1`: exactly the padded
    diagonal the optimizer assumed, with no global sign left over.

    *Still drive every occupied level.* The correction above removes the
    :math:`\pi`, but undriven levels are not perfectly untouched -- the
    off-resonant tails of the multiplexed drive rotate the levels just above
    ``n_drive`` by a few tenths of a radian. Set ``n_drive`` to cover the
    occupied support, padding :math:`\theta_n = 0`, rather than relying on the
    correction alone.

    *Make the pulses long enough.* The components add coherently at
    :math:`t_c`, so the peak amplitude grows with ``n_drive`` and the pulse
    stops being a weak, selective perturbation. Measured against a full
    two-level-plus-cavity simulation, with ``n_drive = 6``:

    ==================  ============  =====================
    :math:`t_\pi`       min :math:`P(g)`  max relative phase error
    ==================  ============  =====================
    :math:`2\pi/\chi`   0.44          0.73 rad
    :math:`4\pi/\chi \cdot 2`  0.94   0.081 rad
    :math:`6 \cdot 2\pi/\chi`  0.979  0.033 rad
    :math:`12 \cdot 2\pi/\chi` 0.995  0.009 rad
    ==================  ============  =====================

    Both errors fall roughly as :math:`1/t_\pi`. The default
    ``params.n_selective_periods = 4`` is a compromise; lengthen it, or drive
    fewer levels, when the residual matters. Numerically optimized selective
    pulses (as in Heeres et al.) do much better at equal duration.

    The duration must stay an integer multiple of :math:`2\pi/\chi` so that the
    level-:math:`n` phase reference is shared by both pulses: between the two
    pulse centres, level :math:`n` precesses by :math:`\chi n t_\pi`, which is
    a multiple of :math:`2\pi` only then.

    Returns ``(eps, omega)``; ``eps`` is zero since SNAP touches the qubit only.
    """
    phases = np.asarray(phases, dtype=float)
    n_drive = int(phases.size if n_drive is None else n_drive)
    if n_drive > phases.size:
        phases = np.concatenate([phases, np.zeros(n_drive - phases.size)])
    phases = phases[:n_drive]

    t_sel = params.t_snap_selective if t_selective is None else float(t_selective)
    sigma = t_sel / 4.0
    g, area = gaussian_envelope(sigma, 4, params.dt, params.subtract_pedestal)
    t = (np.arange(g.size) + 0.5) * params.dt
    t_c = 0.5 * g.size * params.dt
    n_vec = np.arange(n_drive)
    # (n_samples, n_drive) carrier grid
    carrier = np.exp(detuning_sign * 1j * params.chi * np.outer(t - t_c, n_vec))
    amp = (0.5 * np.pi / area) * g

    offset = np.pi if match_padded_identity else 0.0
    om1 = amp * (carrier @ np.ones(n_drive, dtype=complex))
    om2 = amp * (carrier @ np.exp(-1j * (phases + offset)))
    omega = np.concatenate([om1, om2])
    return np.zeros_like(omega), omega


def compile_snap_sequence(
    snap_phases,
    alphas,
    params: SystemParams | None = None,
    delta: float = 0.0,
    t_selective: float | None = None,
    n_drive: int | None = None,
    dt: float | None = None,
    min_disp_samples: int = 8,
    verbose: bool = False,
) -> PulseSequence:
    r"""Compile ``D(alpha_1) -> S(theta_1) -> ... -> S(theta_N) -> D(alpha_{N+1})``.

    Matches :func:`gate_optimization._build_snap_sequence`, whose ``propagate``
    applies ``alphas[0]`` first.

    Parameters
    ----------
    snap_phases : array_like, shape (N, n_snap)
        From ``result.params["snap_phases"]``.
    alphas : array_like, complex, shape (N+1,)
        From ``result.params["alphas"]``.
    delta : float
        Cavity drive detuning. Zero (drive at the ground-state cavity
        frequency) is the natural frame for SNAP, so the ideal gate is exactly
        the optimizer's diagonal phase gate.
    t_selective : float, optional
        Duration of each selective :math:`\pi` pulse. Defaults to
        ``params.n_selective_periods * 2 * pi / chi``. Keep it an integer
        multiple of :math:`2\pi/\chi`; see :func:`snap_waveform`.
    n_drive : int, optional
        Number of Fock levels driven, padded with :math:`\theta_n = 0`.
        Defaults to ``n_snap``, which is almost never what you want: set it to
        the cavity truncation, or at least to the occupied support of the
        intermediate states. A warning is emitted otherwise. See
        :func:`snap_waveform`.
    dt : float, optional
        Override the sample period. SNAP sequences are ~10^3 longer than ECD
        ones, so 1 ns sampling is wasteful; 10-50 ns is ample given that the
        widest carrier in the selective pulses is only
        :math:`\chi \times` ``n_drive``.
    min_disp_samples : int
        Minimum number of samples per displacement pulse. A coarse ``dt``
        chosen for the 100-us selective pulses would leave the 44-ns
        displacements with one or two samples, so their Gaussian is widened
        until this many samples fit. The realized displacement is unaffected
        either way, since the amplitude is normalized by the sampled area, but
        a resolved envelope keeps the drive physical and the trajectory
        integration accurate. Stretching costs nothing here: the displacements
        are shorter than the selective pulses by three orders of magnitude.
    """
    params = params or SystemParams()
    if dt is not None:
        params = params.replace(dt=float(dt))
    snap_phases = np.atleast_2d(np.asarray(snap_phases, dtype=float))
    alphas = np.atleast_1d(np.asarray(alphas, dtype=complex))
    n_snap_gates = snap_phases.shape[0]
    if alphas.size != n_snap_gates + 1:
        raise ValueError(
            f"expected len(alphas) == n_snap + 1 = {n_snap_gates + 1}, got {alphas.size}"
        )

    sigma_disp = max(
        params.sigma_disp, min_disp_samples * params.dt / params.n_sigma_disp
    )
    if sigma_disp > params.sigma_disp and verbose:
        print(
            f"widening displacement pulses: sigma {params.sigma_disp * 1e3:.1f} -> "
            f"{sigma_disp * 1e3:.1f} ns to keep {min_disp_samples} samples at "
            f"dt = {params.dt * 1e3:.1f} ns"
        )
    if n_drive is None:
        warnings.warn(
            f"n_drive defaulted to n_snap = {snap_phases.shape[1]}. Any population above "
            "that level is driven only by the off-resonant tails of the selective pulses, "
            "so pass n_drive explicitly: large enough to cover the occupied support of the "
            "intermediate states, but no larger, since selectivity degrades as more "
            "components are multiplexed.",
            stacklevel=2,
        )
    g, area = gaussian_envelope(
        sigma_disp, params.n_sigma_disp, params.dt, params.subtract_pedestal
    )
    b = _Builder()

    def add_disp(alpha):
        eps = (1j * alpha / area) * g  # alpha = -i * integral(eps) dt
        b.add(eps.astype(complex), np.zeros(g.size, dtype=complex), "displacement", f"D({alpha:+.3f})")

    add_disp(alphas[0])
    for i in range(n_snap_gates):
        if verbose:
            print(f"SNAP {i + 1}/{n_snap_gates}")
        eps, om = snap_waveform(
            snap_phases[i], params, t_selective=t_selective, n_drive=n_drive
        )
        b.add(eps, om, "snap", f"S(theta_{i + 1})", meta={"phases": snap_phases[i]})
        add_disp(alphas[i + 1])

    seq = b.finish(
        params,
        delta,
        {
            "gate_set": "snap",
            "n_snap": n_snap_gates,
            "n_drive": int(snap_phases.shape[1] if n_drive is None else n_drive),
            "t_selective": params.t_snap_selective if t_selective is None else t_selective,
            "sigma_disp": sigma_disp,
            "snap_phases": snap_phases,
            "alphas": alphas,
        },
    )
    seq.check_drive_limits()
    return seq


# ---------------------------------------------------------------------------
# Entry point from a GateOptResult
# ---------------------------------------------------------------------------


def compile_from_result(result, params: SystemParams | None = None, **kwargs) -> PulseSequence:
    """Compile a :class:`gate_optimization.GateOptResult` into a pulse sequence.

    Dispatches on ``result.gate_set``. Extra keyword arguments are forwarded to
    :func:`compile_ecd_sequence` or :func:`compile_snap_sequence`.
    """
    p = result.params
    if result.gate_set == "ecd":
        if not result.config.get("echoed", True):
            raise ValueError(
                "this compiler realizes the echoed gate; re-optimize with echoed=True "
                "or drop the pi pulse from _ecd_waveforms by hand"
            )
        return compile_ecd_sequence(
            p["betas"], p["thetas"], p["phis"], params=params, **kwargs
        )
    if result.gate_set == "snap":
        return compile_snap_sequence(p["snap_phases"], p["alphas"], params=params, **kwargs)
    raise ValueError(f"unknown gate set {result.gate_set!r}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = SystemParams()
    print(f"chi/2pi = {p.chi / TWO_PI * 1e3:.1f} kHz, 2pi/chi = {TWO_PI / p.chi:.2f} us")

    pulse = optimize_ecd_pulse(1.0 + 0.5j, alpha0=30.0, params=p, verbose=True)
    print(pulse.summary())
    print(f"gate time: {pulse.eps.size * p.dt * 1e3:.0f} ns")

    seq = compile_ecd_sequence(
        betas=np.array([1.0 + 0.5j, -0.8 + 0.2j, 2.0 - 0.4j]),
        thetas=np.array([np.pi / 2, np.pi / 3, np.pi / 4, np.pi / 2]),
        phis=np.array([0.0, np.pi / 2, -np.pi / 2, 0.0]),
        alpha0=30.0,
        params=p,
        verbose=True,
    )
    print()
    print(seq.summary())
