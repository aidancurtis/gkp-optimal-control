r"""Device parameters for pulse compilation, with unit-aware config files.

:class:`SystemParams` holds the Hamiltonian, decoherence and drive-constraint
parameters used by :mod:`pulses` and :mod:`pulse_simulation`.
:func:`load_config` reads them from a JSON or TOML file whose entries may carry
explicit units (``"32.8 kHz"``, ``"11 ns"``).

Units
-----
Time is in microseconds throughout; all rates and drive amplitudes are angular
frequencies in rad/us. A frequency quoted in Hz is converted with a factor
:math:`2\pi`, e.g. ``chi = 2 * np.pi * 32.8e-3`` for 32.8 kHz.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
import warnings
from dataclasses import asdict, dataclass, replace

import numpy as np

__all__ = ["SystemParams", "load_config", "parse_quantity", "TWO_PI"]

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
    """Read a JSON or TOML file into a dict, dispatching on extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as fh:
            return json.load(fh)
    if ext == ".toml":
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    raise ValueError(
        f"unsupported config extension {ext!r}; use .json or .toml"
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
        """Read parameters from a JSON or TOML file.

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
