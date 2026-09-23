r"""Write an optimized ECD gate sequence as an explicit sum of displacements.

:mod:`gate_optimization` produces gate *parameters*. This module turns those
numbers into a closed-form operator statement: the joint unitary of the whole
circuit, expanded as a finite sum with one term per path through the sequence.

Conventions are inherited verbatim from :mod:`gate_optimization`:

.. math::
    U = R(\theta_{N+1},\phi_{N+1}) \prod_{i=N}^{1}
        \mathrm{ECD}(\beta_i)\, R(\theta_i,\phi_i),

.. math::
    \mathrm{ECD}(\beta) = D(\beta/2)\,|e\rangle\langle g|
                        + D(-\beta/2)\,|g\rangle\langle e|
                        = \sigma_x \exp\!\big[\tfrac{\sigma_z}{2}
                          (\beta a^\dagger - \bar\beta a)\big],

.. math::
    R(\theta,\phi) = \exp\!\big[-\tfrac{i\theta}{2}
        (\sigma_x\cos\phi + \sigma_y\sin\phi)\big].

How the sum arises
------------------

Pushing every :math:`\sigma_x` left into the rotation before it groups the
circuit into alternating qubit factors and *bare* conditional displacements,

.. math::
    U = Q_N\,\mathrm{CD}(\beta_N)\,Q_{N-1}\cdots Q_1\,\mathrm{CD}(\beta_1)\,Q_0,
    \qquad Q_0 = R_1,\quad Q_i = R_{i+1}\sigma_x,

see :func:`qubit_factors`. Each :math:`\mathrm{CD}` is diagonal in the qubit,
:math:`\mathrm{CD}(\beta) = \sum_{s=\pm1} P_s\, D(s\beta/2)`, so expanding all
:math:`N` of them gives :math:`2^N` terms indexed by a sign path
:math:`s \in \{\pm1\}^N`. Within one path the oscillator factor is an ordered
product of displacements, and since
:math:`[\alpha a^\dagger - \bar\alpha a,\, \alpha' a^\dagger - \bar\alpha' a]`
is a c-number, BCH terminates at second order and that product collapses
*exactly* to a single displacement times a phase
(:func:`displacement_product`). The result is

.. math::
    U = \sum_{s\in\{\pm1\}^N} c_s\, e^{i\Phi_s}\,
        |u_s\rangle\langle v_s| \otimes D(A_s),

.. math::
    A_s = \tfrac{1}{2}\sum_i s_i \beta_i, \qquad
    \Phi_s = \tfrac{1}{4}\sum_{i<j} s_i s_j\,\mathrm{Im}(\beta_j\bar\beta_i),

with each qubit block rank one because it is a chain of projectors
(:func:`branch_expansion`). Projecting onto the qubit returning to
:math:`|g\rangle` leaves the state-preparation operator
:math:`\langle g|U|g\rangle = \sum_s w_s D(A_s)`, whose Fock amplitudes have a
closed form (:func:`fock_coefficients`). This is exact -- no truncation in
:math:`\beta`, no series.

The symbolic engine is Roy Leibov's SymBosonKit. Upstream provides the
*similarity* transform :math:`e^A B e^{-A}` but not the *product* formula
:math:`\log(e^A e^B \cdots)`, so :func:`bch_product` builds the latter on its
``_normal`` primitive. When SymBosonKit is not installed an API-compatible
fallback is used; ``HAVE_SYMBOSONKIT`` reports which is active.

Nothing here imports JAX or ``jaxquantum``: this module sits at the raw
NumPy/SymPy layer, so it cross-checks the differentiable pipeline
independently.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import sympy as sp
from sympy import Add, Mul, cos, cosh, exp, factorial, sin, sinh
from sympy.physics.quantum import Commutator, Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.operatorordering import normal_ordered_form

__all__ = [
    "HAVE_SYMBOSONKIT",
    "similarity_transform",
    "normal",
    "adjoint_action",
    "Algebra",
    "bch_product",
    "displacement_generator",
    "merge_displacements_symbolic",
    "annihilation",
    "num_op",
    "num_mat",
    "fast_displacement",
    "qubit_rotation",
    "qubit_factors",
    "ecd_propagate",
    "ecd_full_unitary",
    "displacement_product",
    "branch_expansion",
    "reconstruct_unitary",
    "reconstruct_projected",
    "fock_coefficients",
    "unitary_sum_latex",
    "SIGMA_X",
    "EcdFixedThetaResult",
    "expand_fixed_theta",
    "optimize_ecd_fixed_theta",
    "rotation_symbolic",
    "sequence_generators",
    "effective_hamiltonian",
    "ecd_sequence_spec",
    "bloch_axis",
    "EffectiveHamiltonian",
    "SIGMA_SYM",
]


# ---------------------------------------------------------------------------
# SymBosonKit: upstream when importable, else an API-compatible fallback
# ---------------------------------------------------------------------------

HAVE_SYMBOSONKIT = False
for _name in ("symbosonkit", "SymBosonKit", "symbosonkit.symbosonkit"):
    try:
        _m = __import__(_name, fromlist=["similarity_transform"])
        _normal = _m._normal
        _ad = _m._ad
        _is_antihermitian = _m._is_antihermitian
        similarity_transform = _m.similarity_transform
        HAVE_SYMBOSONKIT = True
        break
    except Exception:
        continue

if not HAVE_SYMBOSONKIT:  # pragma: no cover - active only without upstream

    def _normal(expr):
        """Normal-order *and* separate commutative factors."""
        expr = sp.sympify(expr)
        if expr.is_Add:
            return Add(*[_normal(arg) for arg in expr.args])
        if expr.is_Mul:
            comm = Mul(*[f for f in expr.args if f.is_commutative])
            noncomm = Mul(*[f for f in expr.args if not f.is_commutative],
                          evaluate=False)
            if noncomm is sp.S.One:
                return comm
            return comm * normal_ordered_form(noncomm.expand(), independent=True)
        if expr.is_commutative:
            return expr
        return normal_ordered_form(expr.expand(), independent=True)

    def _ad(A, X):
        """ad_A(X) = [A, X], linear in A."""
        A, X = sp.sympify(A), sp.sympify(X)
        if A.is_Add:
            return _normal(Add(*[_ad(term, X) for term in A.args]))
        return _normal(Commutator(A, X).doit())

    def _is_antihermitian(op):
        return sp.simplify(_normal(Dagger(op) + op)) == 0

    def similarity_transform(A, B, *, max_order=6, use_bch_only=False):
        r"""``e^A B e^{-A}``, mirroring the upstream shortcuts."""
        if not _is_antihermitian(A):
            raise ValueError("Generator A must be anti-Hermitian.")

        def _series():
            result = term = _normal(B)
            for n in range(1, max_order + 1):
                term = _ad(A, term)
                if term == 0:
                    break
                result += term / factorial(n)
            return _normal(result)

        if use_bch_only:
            return _series()
        comm1 = _ad(A, B)
        if comm1 == 0:
            return _normal(B)
        if comm1.is_commutative:
            return _normal(B + comm1)
        k = sp.Wild("k", commutative=True)
        m = comm1.match(k * B)
        if m:
            return _normal(exp(_normal(m[k])) * B)
        op = sp.Wild("Op", commutative=False)
        m1 = comm1.match(k * op)
        if m1:
            k_val, c_op = _normal(m1[k]), _normal(m1[op])
            q = sp.Wild("q", commutative=True)
            m2 = _ad(A, c_op).match(q * B)
            if m2:
                q_val = _normal(m2[q])
                if sp.simplify(q_val + k_val) == 0:
                    return _normal(cos(k_val) * B + sin(k_val) / k_val * comm1)
                if sp.simplify(q_val - k_val) == 0:
                    return _normal(cosh(k_val) * B - sinh(k_val) / k_val * comm1)
        return _series()


def normal(expr):
    """Normal-order and fully collect a Weyl-algebra expression.

    ``_normal`` has to be applied twice around ``expand`` to be idempotent: one
    pass leaves e.g. ``-(a^dag a^2 + a) + a^dag a^2`` uncollected, because it
    does not distribute a scalar over an ``Add`` it has just normal-ordered.
    """
    return sp.expand(_normal(sp.expand(_normal(sp.expand(expr)))))


def adjoint_action(A, X):
    """``[A, X]``, normal-ordered. Thin wrapper over SymBosonKit ``_ad``."""
    return normal(_ad(A, X))


# ---------------------------------------------------------------------------
# BCH product formula, graded by displacement order
# ---------------------------------------------------------------------------


class Algebra:
    """Zero, identity and normalizer for one element type.

    ``"scalar"``
        Plain SymPy expressions in ``BosonOp`` -- the oscillator alone.
    ``"matrix"``
        2x2 SymPy matrices over that algebra, in the ``(|g>, |e>)`` basis --
        qubit tensor oscillator, which is what a whole ECD sequence lives in.
    """

    def __init__(self, kind: str = "scalar"):
        if kind == "scalar":
            self.zero, self.one, self.norm = sp.S.Zero, sp.S.One, normal
        elif kind == "matrix":
            self.zero, self.one = sp.zeros(2, 2), sp.eye(2)
            self.norm = lambda x: sp.Matrix(2, 2, lambda i, j: normal(x[i, j]))
        else:
            raise ValueError(f"unknown kind {kind!r}")
        self.kind = kind

    def is_zero(self, x) -> bool:
        if self.kind == "matrix":
            return bool(x.is_zero_matrix)
        return x == 0


def _g_zero(p, alg):
    return [alg.zero for _ in range(p + 1)]


def _g_one(p, alg):
    T = _g_zero(p, alg)
    T[0] = alg.one
    return T


def _g_mul(A, B, p, alg):
    out = _g_zero(p, alg)
    for i, ai in enumerate(A):
        if alg.is_zero(ai):
            continue
        for j, bj in enumerate(B):
            if i + j > p or alg.is_zero(bj):
                continue
            out[i + j] = out[i + j] + ai * bj
    return [alg.norm(t) for t in out]


def _g_exp(G, p, alg):
    """``exp(G)`` for ``G`` with vanishing ``eps**0`` part."""
    res, term = _g_one(p, alg), _g_one(p, alg)
    for m in range(1, p + 1):
        term = [alg.norm(t / m) for t in _g_mul(term, G, p, alg)]
        res = [alg.norm(r + t) for r, t in zip(res, term)]
    return res


def _g_log(P, p, alg):
    """``log(P)`` for ``P`` whose ``eps**0`` part is the identity."""
    X = list(P)
    X[0] = alg.norm(P[0] - alg.one)
    res = _g_zero(p, alg)
    Xk = _g_one(p, alg)
    for k in range(1, p + 1):
        Xk = _g_mul(Xk, X, p, alg)
        coef = sp.Rational((-1) ** (k + 1), k)
        res = [alg.norm(r + coef * t) for r, t in zip(res, Xk)]
    return res


def bch_product(generators: Sequence, max_order: int = 4,
                alg: "Algebra | None" = None) -> list:
    r"""Merge a product of exponentials into one, graded by displacement order.

    Returns the truncated series for
    :math:`\log\!\big(e^{G_{\rm last}}\cdots e^{G_{\rm first}}\big)`.

    Rather than transcribing Dynkin's BCH coefficients, this multiplies the
    truncated exponential series and takes the truncated logarithm. Each
    generator counts as :math:`O(\varepsilon)` with :math:`\varepsilon` marking
    powers of the displacement amplitude, so a term with :math:`n` nested
    commutators is :math:`O(\varepsilon^{n+1})` and the grading is consistent
    order by order. This is exact at each order and much less error-prone than
    hand-coded high-order BCH terms.

    Parameters
    ----------
    generators : sequence
        Ordered *first-applied-first*, matching the circuit; the operator
        product is formed right-to-left.
    max_order : int
        Highest retained power of the displacement amplitude.
    alg : Algebra, optional
        Element type. Defaults to ``Algebra("scalar")``; pass
        ``Algebra("matrix")`` for qubit-coupled generators.

    Returns
    -------
    list
        ``T`` with ``T[k]`` the :math:`\varepsilon^k` coefficient; ``T[0] = 0``
        and the merged generator is ``sum(T)``.

    Notes
    -----
    For displacements alone the commutator is central, so the series
    *terminates* at ``k = 2`` and the result is exact for any amplitude. Once
    qubit operators along non-parallel axes appear the algebra no longer
    closes: the series is asymptotic, and its radius must be checked -- see
    :meth:`EffectiveHamiltonian.convergence`.
    """
    alg = alg or Algebra("scalar")
    P = _g_one(max_order, alg)
    for gen in generators:
        G = _g_zero(max_order, alg)
        if max_order >= 1:
            G[1] = gen
        P = _g_mul(_g_exp(G, max_order, alg), P, max_order, alg)
    return _g_log(P, max_order, alg)


def displacement_generator(alpha, a: BosonOp | None = None):
    r"""The anti-Hermitian generator :math:`\alpha a^\dagger - \bar\alpha a`."""
    a = a or BosonOp("a")
    return alpha * Dagger(a) - sp.conjugate(alpha) * a


def merge_displacements_symbolic(n_gates: int, max_order: int = 5):
    r"""Symbolically collapse a product of ``n_gates`` displacements.

    Runs :func:`bch_product` on generators for :math:`D(\alpha_n)\cdots
    D(\alpha_1)` and returns the pieces of

    .. math::
        \prod_{i=n}^{1} D(\alpha_i) = e^{i\Phi}\,D\Big(\textstyle\sum_i
        \alpha_i\Big), \qquad \Phi = \sum_{i<j}\mathrm{Im}(\alpha_j\bar\alpha_i).

    Returns
    -------
    alphas : list of Symbol
        The symbols :math:`\alpha_1 \ldots \alpha_n`.
    generator : sympy expression
        The merged first-order generator, i.e. :math:`T[1]`.
    phase_term : sympy expression
        The c-number :math:`T[2] = i\Phi`.
    terminated : bool
        Whether every order above two vanished, as it must.
    """
    alphas = list(sp.symbols(f"alpha_1:{n_gates + 1}", complex=True))
    series = bch_product([displacement_generator(al) for al in alphas], max_order)
    terminated = all(sp.simplify(t) == 0 for t in series[3:])
    return alphas, sp.expand(series[1]), sp.simplify(series[2]), terminated


# ---------------------------------------------------------------------------
# Numeric evaluation
# ---------------------------------------------------------------------------


def annihilation(n_fock: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, n_fock)), 1).astype(complex)


def num_op(expr, n_fock: int, subs: dict | None = None) -> np.ndarray:
    """Evaluate a Weyl-algebra expression as an ``(n_fock, n_fock)`` array."""
    expr = sp.sympify(expr)
    if subs:
        expr = expr.subs({k: sp.sympify(v) for k, v in subs.items()})
    expr = sp.expand(expr)
    eye = np.eye(n_fock, dtype=complex)
    a = annihilation(n_fock)

    def walk(e):
        if e.is_Add:
            return sum(walk(t) for t in e.args)
        if e.is_Mul:
            out = eye
            for f in e.args:
                out = out @ walk(f)
            return out
        if e.is_Pow:
            if not (e.exp.is_Integer and e.exp >= 0):
                raise TypeError(f"non-integer operator power: {e!r}")
            out = eye
            for _ in range(int(e.exp)):
                out = out @ walk(e.base)
            return out
        if isinstance(e, BosonOp):
            return a if e.is_annihilation else a.conj().T
        if e.is_commutative:
            return complex(e) * eye
        raise TypeError(f"cannot evaluate {e!r} ({type(e).__name__})")

    return walk(expr)


def num_mat(mat, n_fock: int, subs: dict | None = None) -> np.ndarray:
    """2x2 operator matrix -> ``(2 n_fock, 2 n_fock)``, ``kron(qubit, cavity)``."""
    return np.block([[num_op(mat[i, j], n_fock, subs) for j in range(2)]
                     for i in range(2)])


class fast_displacement:
    r"""Cached exact truncated displacement operator.

    Uses :math:`D(\alpha) = e^{i\theta n} D(r) e^{-i\theta n}` for
    :math:`\alpha = r e^{i\theta}`, so only one eigendecomposition of the fixed
    matrix :math:`a^\dagger - a` is ever needed. Conjugating by the diagonal
    :math:`e^{i\theta n}` commutes with Fock truncation, so this is *identical*
    to ``expm`` of the truncated generator (agreement ~1e-15) while being
    several times faster.
    """

    def __init__(self, n_fock: int):
        self.n_fock = n_fock
        a = annihilation(n_fock)
        self._lam, self._vec = np.linalg.eigh(1j * (a.conj().T - a))
        self._n = np.arange(n_fock)

    def __call__(self, alpha) -> np.ndarray:
        r = abs(alpha)
        if r == 0.0:
            return np.eye(self.n_fock, dtype=complex)
        d_r = self._vec @ (np.exp(-1j * r * self._lam)[:, None] * self._vec.conj().T)
        ph = np.exp(1j * np.angle(alpha) * self._n)
        return ph[:, None] * d_r * ph.conj()[None, :]


# ---------------------------------------------------------------------------
# Circuit structure (NumPy mirror of the ECD gate set)
# ---------------------------------------------------------------------------

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
_KET = {1: np.array([1, 0], dtype=complex), -1: np.array([0, 1], dtype=complex)}


def qubit_rotation(theta: float, phi: float) -> np.ndarray:
    r"""``R(theta, phi)`` as a 2x2 array, matching :mod:`gate_optimization`."""
    c, s = np.cos(theta / 2) + 0j, np.sin(theta / 2) + 0j
    return np.array([[c, -1j * np.exp(-1j * phi) * s],
                     [-1j * np.exp(1j * phi) * s, c]])


def qubit_factors(thetas, phis) -> list[np.ndarray]:
    r"""The ``N + 1`` qubit factors :math:`Q_0 \ldots Q_N` of the circuit.

    :math:`Q_0 = R_1` and :math:`Q_i = R_{i+1}\sigma_x`, the echo flip of each
    ECD having been absorbed into the rotation that follows it.
    """
    rot = [qubit_rotation(t, p) for t, p in zip(np.asarray(thetas), np.asarray(phis))]
    return [rot[0]] + [r @ SIGMA_X for r in rot[1:]]


def ecd_propagate(thetas, phis, betas, n_fock: int, psi0=None,
                  return_history: bool = False):
    """Run the ECD circuit on ``|g>`` times a cavity ket.

    The state is carried as a ``(2, n_fock)`` block pair with index 0 = ``|g>``,
    the same layout :mod:`gate_optimization` uses internally.
    """
    disp = fast_displacement(n_fock)
    if psi0 is None:
        psi0 = np.zeros(n_fock, dtype=complex)
        psi0[0] = 1.0
    psi = np.stack([np.asarray(psi0, dtype=complex),
                    np.zeros(n_fock, dtype=complex)])
    hist = [psi.copy()]
    psi = qubit_rotation(thetas[0], phis[0]) @ psi
    hist.append(psi.copy())
    for i, beta in enumerate(np.asarray(betas)):
        d_op = disp(beta / 2)
        psi = np.stack([d_op.conj().T @ psi[1], d_op @ psi[0]])
        hist.append(psi.copy())
        psi = qubit_rotation(thetas[i + 1], phis[i + 1]) @ psi
        hist.append(psi.copy())
    return (psi, np.stack(hist)) if return_history else psi


def ecd_full_unitary(thetas, phis, betas, n_fock: int) -> np.ndarray:
    """The joint ``(2 n_fock, 2 n_fock)`` unitary, ``kron(qubit, cavity)``."""
    disp = fast_displacement(n_fock)
    eye = np.eye(n_fock, dtype=complex)
    out = np.kron(qubit_rotation(thetas[0], phis[0]), eye)
    for i, beta in enumerate(np.asarray(betas)):
        d_op = disp(beta / 2)
        ecd = np.zeros((2 * n_fock, 2 * n_fock), dtype=complex)
        ecd[:n_fock, n_fock:] = d_op.conj().T
        ecd[n_fock:, :n_fock] = d_op
        out = np.kron(qubit_rotation(thetas[i + 1], phis[i + 1]), eye) @ ecd @ out
    return out


# ---------------------------------------------------------------------------
# The summation
# ---------------------------------------------------------------------------


def displacement_product(alphas) -> tuple[complex, float]:
    r"""Collapse :math:`D(\alpha_n)\cdots D(\alpha_1)` into one displacement.

    Because :math:`[\alpha a^\dagger - \bar\alpha a,\;\alpha' a^\dagger -
    \bar\alpha' a] = 2i\,\mathrm{Im}(\alpha\bar\alpha')` is central, BCH stops
    at second order and

    .. math::
        \prod_{i=n}^{1} D(\alpha_i) = e^{i\Phi}\,
        D\Big(\textstyle\sum_i \alpha_i\Big), \qquad
        \Phi = \sum_{i<j}\mathrm{Im}(\alpha_j\bar\alpha_i),

    exactly, at any magnitude.

    Returns
    -------
    total : complex
        :math:`\sum_i \alpha_i`.
    phase : float
        :math:`\Phi` in radians.
    """
    alphas = np.asarray(alphas, dtype=complex)
    phase = sum(float(np.imag(alphas[j] * np.conj(alphas[i])))
                for i in range(len(alphas)) for j in range(i + 1, len(alphas)))
    return complex(alphas.sum()), phase


def branch_expansion(thetas, phis, betas) -> list[dict]:
    r"""Expand the joint unitary over the :math:`2^N` sign paths.

    Substituting :math:`\mathrm{CD}(\beta) = \sum_{s=\pm1} P_s D(s\beta/2)` for
    every ECD and collapsing each path's displacement product gives the exact
    finite sum

    .. math::
        U = \sum_{s\in\{\pm1\}^N} c_s\,e^{i\Phi_s}\,
            |u_s\rangle\langle v_s| \otimes D(A_s).

    Each qubit block is rank one, being a chain of projectors: with
    :math:`|a_{+1}\rangle = |g\rangle`, :math:`|a_{-1}\rangle = |e\rangle`,

    .. math::
        c_s = \prod_{i=1}^{N-1}\langle a_{s_{i+1}}|Q_i|a_{s_i}\rangle, \quad
        |u_s\rangle = Q_N|a_{s_N}\rangle, \quad
        \langle v_s| = \langle a_{s_1}|Q_0 .

    Returns
    -------
    list of dict
        One entry per path, with keys ``signs``, ``alpha`` (:math:`A_s`),
        ``phase`` (:math:`\Phi_s`), ``qubit`` (the full 2x2 block including
        :math:`c_s` and the phase), ``coeff`` (:math:`c_s`), ``u``, ``v``, and
        ``weight`` (:math:`w_s`, the scalar left after projecting the qubit
        onto :math:`|g\rangle \to |g\rangle`).
    """
    betas = np.asarray(betas, dtype=complex)
    n_gates = len(betas)
    q = qubit_factors(thetas, phis)
    ground = _KET[1]
    out = []
    for signs in itertools.product((1, -1), repeat=n_gates):
        coeff = 1.0 + 0.0j
        for i in range(n_gates - 1):
            coeff *= np.vdot(_KET[signs[i + 1]], q[i + 1] @ _KET[signs[i]])
        u = q[n_gates] @ _KET[signs[n_gates - 1]]
        v = q[0].conj().T @ _KET[signs[0]]          # <v| = <a_s1| Q_0
        alpha, phase = displacement_product(
            [s * b / 2 for s, b in zip(signs, betas)])
        block = coeff * np.exp(1j * phase) * np.outer(u, v.conj())
        out.append(dict(signs=signs, alpha=alpha, phase=phase,
                        qubit=block, coeff=complex(coeff), u=u, v=v,
                        weight=complex(block[0, 0])))
    return out


def reconstruct_unitary(branches: Sequence[dict], n_fock: int) -> np.ndarray:
    """Rebuild the joint unitary from the branch sum, for verification."""
    disp = fast_displacement(n_fock)
    out = np.zeros((2 * n_fock, 2 * n_fock), dtype=complex)
    for term in branches:
        out = out + np.kron(term["qubit"], disp(term["alpha"]))
    return out


def reconstruct_projected(branches: Sequence[dict], n_fock: int) -> np.ndarray:
    r"""Rebuild :math:`\langle g|U|g\rangle = \sum_s w_s D(A_s)`."""
    disp = fast_displacement(n_fock)
    out = np.zeros((n_fock, n_fock), dtype=complex)
    for term in branches:
        out = out + term["weight"] * disp(term["alpha"])
    return out


def fock_coefficients(branches: Sequence[dict], max_n: int = 8) -> np.ndarray:
    r"""Fock amplitudes of :math:`\langle g|U|g\rangle|0\rangle`.

    From :math:`D(A)|0\rangle = e^{-|A|^2/2}\sum_n A^n|n\rangle/\sqrt{n!}`,

    .. math::
        \kappa_n = \frac{1}{\sqrt{n!}}
            \sum_s w_s\,e^{-|A_s|^2/2} A_s^{\,n}.

    Preparing :math:`|m\rangle` drives every :math:`\kappa_n` with
    :math:`n \neq m` to zero; the fidelity achieved is :math:`|\kappa_m|^2`.
    """
    return np.array([
        sum(t["weight"] * np.exp(-abs(t["alpha"]) ** 2 / 2) * t["alpha"] ** n
            for t in branches) / math.sqrt(math.factorial(n))
        for n in range(max_n)
    ])


def _fmt(z: complex, digits: int = 3) -> str:
    r, i = np.real(z), np.imag(z)
    if abs(i) < 10 ** (-digits):
        return f"{r:.{digits}f}"
    if abs(r) < 10 ** (-digits):
        return f"{i:+.{digits}f}i"
    return f"({r:.{digits}f}{i:+.{digits}f}i)"


def unitary_sum_latex(branches: Sequence[dict], digits: int = 3,
                      projected: bool = False, max_terms: int = 16) -> str:
    r"""Render the branch sum as a LaTeX ``aligned`` environment.

    With ``projected=False`` each term carries its 2x2 qubit block; with
    ``projected=True`` the qubit is projected onto :math:`|g\rangle \to
    |g\rangle` and every term is a scalar times a displacement.
    """
    lhs = (r"\langle g|\hat U|g\rangle" if projected else r"\hat U")
    rows = []
    for k, term in enumerate(branches[:max_terms]):
        sgn = ",".join("+" if s > 0 else "-" for s in term["signs"])
        disp = rf"\hat D\!\left({_fmt(term['alpha'], digits)}\right)"
        if projected:
            body = rf"{_fmt(term['weight'], digits)}\,{disp}"
        else:
            b = term["qubit"]
            mat = (r"\begin{pmatrix}"
                   + _fmt(b[0, 0], digits) + " & " + _fmt(b[0, 1], digits)
                   + r"\\" + _fmt(b[1, 0], digits) + " & "
                   + _fmt(b[1, 1], digits) + r"\end{pmatrix}\otimes " + disp)
            body = mat
        lead = r"=\;&" if k == 0 else r"+\;&"
        rows.append(rf"{lead}{body} &&\;s=({sgn})")
    tail = ""
    if len(branches) > max_terms:
        tail = rf"\\+\;&\cdots && (\text{{{len(branches) - max_terms} more}})"
    return (r"\begin{aligned}" + lhs + r"\;" + r"\\".join(rows) + tail
            + r"\end{aligned}")


# ---------------------------------------------------------------------------
# Optimization with the rotation angles frozen
# ---------------------------------------------------------------------------


@dataclass
class EcdFixedThetaResult:
    r"""Outcome of a fixed-:math:`\theta` ECD optimization.

    Deliberately quacks like :class:`gate_optimization.GateOptResult`:
    ``params``, ``flat_params``, ``fidelity``, ``leakage`` and ``summary()``
    all carry the same meaning and layout, so downstream code that consumes an
    optimizer result needs no changes. ``flat_params`` is the *full*
    :math:`4N+2` vector in the project's interleaved
    ``[theta_1, phi_1, ..., Re beta_1, Im beta_1, ...]`` order, with every
    ``theta`` set to the frozen value, so it can be handed straight to
    :func:`gate_optimization.sequence_history`.
    """

    gate_set: str
    n_gates: int
    n_fock: int
    theta: float
    fidelity: float
    loss: float
    leakage: float
    params: dict
    flat_params: np.ndarray
    reduced_params: np.ndarray
    final_blocks: np.ndarray
    per_seed_fidelity: np.ndarray
    best_seed: int
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        spread = (f"{self.per_seed_fidelity.min():.4f} .. "
                  f"{self.per_seed_fidelity.max():.4f}")
        n_free = self.reduced_params.size
        return "\n".join([
            f"gate set        : {self.gate_set} (theta frozen)",
            f"n_gates         : {self.n_gates}",
            f"theta_i         : {self.theta / np.pi:.6f} pi  (all {self.n_gates + 1})",
            f"n_params        : {n_free} free of {self.flat_params.size} "
            f"({self.n_gates + 1} frozen)",
            f"fidelity        : {self.fidelity:.6f}",
            f"infidelity      : {1 - self.fidelity:.3e}",
            f"loss            : {self.loss:+.6e}",
            f"leakage         : {self.leakage:.3e}",
            f"peak |beta|     : {np.abs(self.params['betas']).max():.3f}",
            f"best seed       : {self.best_seed} of {self.per_seed_fidelity.size}",
            f"seed F spread   : {spread}",
        ])

    def final_ket(self) -> np.ndarray:
        r"""The ``|g>`` cavity block, unnormalized.

        Pass through ``utils.to_qarray`` to plot it -- this module stays on the
        raw-array side of the ``Qarray`` boundary on purpose.
        """
        return self.final_blocks[0]


def expand_fixed_theta(reduced, n_gates: int, theta: float) -> np.ndarray:
    r"""Reduced vector -> full flat parameter vector with every angle frozen.

    The reduced vector is
    :math:`[\phi_1 \ldots \phi_{N+1},\, \mathrm{Re}\beta_1, \mathrm{Im}\beta_1,
    \ldots]`, of length :math:`3N+1`; the output is the :math:`4N+2` vector
    :mod:`gate_optimization` expects.
    """
    reduced = np.asarray(reduced, dtype=float)
    n_rot = n_gates + 1
    phis, betas_ri = reduced[:n_rot], reduced[n_rot:]
    rots = np.stack([np.full(n_rot, float(theta)), phis], axis=-1)
    return np.concatenate([rots.ravel(), betas_ri])


def _unpack_flat(flat, n_gates: int):
    flat = np.asarray(flat, dtype=float)
    n_rot = n_gates + 1
    rots = flat[:2 * n_rot].reshape(n_rot, 2)
    betas_ri = flat[2 * n_rot:].reshape(n_gates, 2)
    return rots[:, 0], rots[:, 1], betas_ri[:, 0] + 1j * betas_ri[:, 1]


def optimize_ecd_fixed_theta(
    n_gates: int,
    n_fock: int,
    fock_target: int | None = None,
    *,
    psi_target=None,
    theta: float | None = None,
    n_seeds: int = 64,
    seed: int = 0,
    init_disp_scale: float = 0.8,
    max_disp: float | None = 4.0,
    max_disp_weight: float = 1.0,
    n_leak: int = 6,
    leakage_weight: float = 1.0,
    maxiter: int = 600,
    params0=None,
    verbose: bool = True,
) -> EcdFixedThetaResult:
    r"""Optimize an ECD sequence for Fock preparation with all angles frozen.

    Holds every :math:`\theta_i` at ``theta`` (default :math:`\pi/2`) and
    optimizes only the rotation phases :math:`\phi_i` and the displacements
    :math:`\beta_i`, cutting the parameter count from :math:`4N+2` to
    :math:`3N+1`.

    :math:`\theta = \pi/2` is the natural choice rather than an arbitrary one:
    :math:`R(\pi/2,\phi)` maps :math:`\sigma_z` onto the equator, so each
    conditional displacement acts along an equatorial Bloch axis and successive
    axes can be set purely by the phases :math:`\phi_i`. Left free, the
    optimizer picks :math:`\theta_i = \pi/2` on its own at the optima found
    here, so the constraint costs no fidelity while removing a flat direction.

    This runs on the NumPy layer with multi-start L-BFGS-B rather than through
    :func:`gate_optimization.optimize_gate_sequence`, because the JAX ansatz has
    no hook for freezing a subset of parameters (``params0`` only seeds them).
    At :math:`3N+1 \le 13` parameters that is not a real cost. The penalties
    mirror :class:`gate_optimization.GateBounds` term for term, so results are
    comparable.

    Parameters
    ----------
    n_gates : int
        Circuit depth :math:`N`, counted in ECDs.
    n_fock : int
        Truncation used during optimization.
    fock_target : int, optional
        Target Fock level :math:`m`. Exactly one of this and ``psi_target``
        must be given.
    psi_target : array_like, optional
        Arbitrary normalized target ket of shape ``(n_fock,)`` -- a GKP logical
        state, a cat state, anything. Use ``utils.to_ket`` to get the raw
        vector out of a ``Qarray``.
    theta : float, optional
        Frozen rotation angle. Defaults to :math:`\pi/2`.
    n_seeds : int
        Random restarts. Each gets an L-BFGS-B run on the infidelity followed
        by a ``log10`` infidelity polish, which matters above :math:`F = 0.99`
        where :math:`1 - F` is nearly flat.
    init_disp_scale : float
        Centre of the ladder of initial displacement scales the restarts walk.
        The spread matters more than the raw seed count: restarts launched from
        a single scale collapse into one basin and deeper circuits stall short.
    params0 : array_like, optional
        Explicit warm start(s), tried *in addition* to the ``n_seeds`` random
        restarts. Each row may be a reduced vector of length :math:`3N+1`
        (:math:`\phi` then :math:`\mathrm{Re}\beta, \mathrm{Im}\beta`) or a
        full :math:`4N+2` vector in the project's interleaved layout, whose
        :math:`\theta` entries are simply discarded. Pass
        ``result_free.flat_params`` to snap a free-angle solution onto
        :math:`\theta = \pi/2` and re-polish from the same basin, which is far
        cheaper than cold-starting at GKP depth -- though note it then measures
        the cost of the *constraint* within that basin, not an independent
        search.
    max_disp, max_disp_weight, n_leak, leakage_weight
        Soft displacement cap and Fock-leakage penalty, matching
        :class:`gate_optimization.GateBounds`. Leaving these off invites
        solutions that exploit the truncation: displacements at
        :math:`|\beta| \sim 20` can report a high fidelity that evaporates when
        ``n_fock`` grows.

    Returns
    -------
    EcdFixedThetaResult
    """
    theta = float(np.pi / 2 if theta is None else theta)
    n_rot = n_gates + 1
    if (fock_target is None) == (psi_target is None):
        raise ValueError("give exactly one of fock_target or psi_target")
    if fock_target is not None:
        target_vec = np.zeros(n_fock, dtype=complex)
        target_vec[int(fock_target)] = 1.0
    else:
        target_vec = np.asarray(psi_target, dtype=complex).ravel()
        if target_vec.size != n_fock:
            raise ValueError(
                f"psi_target has {target_vec.size} entries, expected n_fock={n_fock}"
            )
        target_vec = target_vec / np.linalg.norm(target_vec)

    def _score(reduced):
        flat = expand_fixed_theta(reduced, n_gates, theta)
        thetas, phis, betas = _unpack_flat(flat, n_gates)
        _, hist = ecd_propagate(thetas, phis, betas, n_fock, return_history=True)
        fid = float(abs(np.vdot(target_vec, hist[-1][0])) ** 2)
        leak = 0.0
        if n_leak > 0:
            for k in range(2, 2 * n_gates + 2, 2):      # states right after each ECD
                leak += float(np.sum(np.abs(hist[k][:, -n_leak:]) ** 2))
        cap = 0.0
        if max_disp is not None and max_disp_weight != 0.0:
            cap = float(np.sum(np.maximum(np.abs(betas) - max_disp, 0.0) ** 2))
        return fid, leak, cap

    def _penalty(reduced):
        _, leak, cap = _score(reduced)
        return leakage_weight * leak + max_disp_weight * cap

    def _loss_infid(reduced):
        fid, leak, cap = _score(reduced)
        return (1.0 - fid) + leakage_weight * leak + max_disp_weight * cap

    def _loss_log(reduced):
        fid, leak, cap = _score(reduced)
        return (np.log10(max(1.0 - fid, 1e-16))
                + leakage_weight * leak + max_disp_weight * cap)

    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    best = (-np.inf, None, -1)
    seed_fids = []
    # The initial displacement scale matters more than the seed count: at a single
    # scale the restarts pile into one basin and deeper circuits stall. Walk a
    # ladder around init_disp_scale instead.
    ladder = np.array([0.4, 0.7, 1.0, 1.5, 2.2]) * init_disp_scale

    warm = []
    if params0 is not None:
        for row in np.atleast_2d(np.asarray(params0, dtype=float)):
            if row.size == 3 * n_gates + 1:
                warm.append(row.copy())
            elif row.size == 4 * n_gates + 2:
                rots = row[:2 * n_rot].reshape(n_rot, 2)
                warm.append(np.concatenate([rots[:, 1], row[2 * n_rot:]]))
            else:
                raise ValueError(
                    f"params0 row has {row.size} entries; expected "
                    f"{3 * n_gates + 1} (reduced) or {4 * n_gates + 2} (full)"
                )

    for k in range(len(warm) + n_seeds):
        if k < len(warm):
            x0 = warm[k]
        else:
            j = k - len(warm)
            x0 = np.concatenate([
                rng.uniform(-np.pi, np.pi, n_rot),
                ladder[j % ladder.size] * rng.normal(size=2 * n_gates),
            ])
        res = minimize(_loss_infid, x0, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, ftol=1e-15))
        res = minimize(_loss_log, res.x, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, ftol=1e-16, gtol=1e-14))
        fid, _, _ = _score(res.x)
        seed_fids.append(fid)
        # reject seeds that bought fidelity with leakage or an oversized beta
        if fid > best[0] and _penalty(res.x) < 1e-6:
            best = (fid, res.x, k)

    fid_best, reduced_best, best_seed = best
    if reduced_best is None:
        raise RuntimeError(
            f"no seed converged to a leakage-free solution at N = {n_gates}; "
            "raise n_seeds or n_fock, or relax max_disp / n_leak"
        )

    flat = expand_fixed_theta(reduced_best, n_gates, theta)
    thetas, phis, betas = _unpack_flat(flat, n_gates)
    blocks = ecd_propagate(thetas, phis, betas, n_fock)
    fid, leak, cap = _score(reduced_best)

    result = EcdFixedThetaResult(
        gate_set="ecd",
        n_gates=n_gates,
        n_fock=n_fock,
        theta=theta,
        fidelity=fid,
        loss=float(_loss_infid(reduced_best)),
        leakage=leak,
        params={"thetas": thetas, "phis": phis, "betas": betas,
                "disp_magnitudes": np.abs(betas)},
        flat_params=flat,
        reduced_params=np.asarray(reduced_best),
        final_blocks=blocks,
        per_seed_fidelity=np.asarray(seed_fids),
        best_seed=best_seed,
        # the keys gate_optimization.sequence_history looks up; "expm" because
        # fast_displacement is exactly equal to expm of the truncated generator
        config={
            "disp_method": "expm",
            "echoed": True,
            "qubit_target": "ground",
            "n_snap": None,
            "bounds": SimpleNamespace(
                max_disp=max_disp, max_disp_weight=max_disp_weight,
                n_leak=n_leak, leakage_weight=leakage_weight),
            "theta_fixed": theta,
        },
    )
    if verbose:
        print(result.summary())
    return result


# ---------------------------------------------------------------------------
# log U and the effective Hamiltonian of an arbitrary ECD sequence
# ---------------------------------------------------------------------------

SIGMA_SYM = {
    "i": sp.eye(2),
    "x": sp.Matrix([[0, 1], [1, 0]]),
    "y": sp.Matrix([[0, -sp.I], [sp.I, 0]]),
    "z": sp.Matrix([[1, 0], [0, -1]]),
}


def rotation_symbolic(axis, angle) -> sp.Matrix:
    r"""Exact :math:`\exp[-i\,\vartheta\,(\hat n\cdot\vec\sigma)/2]` as a 2x2 matrix.

    ``axis`` is ``"x"``, ``"y"``, ``"z"`` or a 3-vector (normalized here).
    ``angle`` may be symbolic; pass SymPy values such as ``sp.pi / 2`` to keep
    the result exact rather than floating point.
    """
    if isinstance(axis, str):
        vec = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis.lower()]
    else:
        vec = tuple(sp.sympify(c) for c in axis)
    norm = sp.sqrt(sum(sp.Abs(c) ** 2 for c in vec))
    vec = [sp.simplify(c / norm) for c in vec]
    n_dot_sigma = sum((v * SIGMA_SYM[c] for v, c in zip(vec, "xyz")),
                      sp.zeros(2, 2))
    angle = sp.sympify(angle)
    return sp.simplify(sp.cos(angle / 2) * sp.eye(2)
                       - sp.I * sp.sin(angle / 2) * n_dot_sigma)


def _op_from_spec(item):
    """Normalize one sequence entry into ``("qubit", M)`` or ``("cd", amp)``."""
    if isinstance(item, sp.MatrixBase):
        return ("qubit", item)
    kind = str(item[0]).lower()
    if kind in ("ecd", "cd"):
        return (kind, sp.sympify(item[1]))
    if kind in ("rx", "ry", "rz"):
        return ("qubit", rotation_symbolic(kind[1], item[1]))
    if kind == "rot":                       # project's equatorial R(theta, phi)
        theta, phi = sp.sympify(item[1]), sp.sympify(item[2])
        return ("qubit", rotation_symbolic((sp.cos(phi), sp.sin(phi), 0), theta))
    if kind == "axis":
        return ("qubit", rotation_symbolic(item[1], item[2]))
    if kind == "qubit":
        return ("qubit", sp.Matrix(item[1]))
    raise ValueError(f"unknown sequence entry {item!r}")


@dataclass
class EffectiveHamiltonian:
    r"""Result of merging an ECD sequence into a single exponential.

    Holds the graded expansion of :math:`\log U`, factored as

    .. math:: U = \mathcal{Q}\,\exp(G) = \mathcal{Q}\,e^{-i H_{\rm eff}},

    where :math:`\mathcal{Q}` is the leftover pure-qubit unitary and
    :math:`G = -i H_{\rm eff}` is graded by powers of the ECD amplitudes.
    """

    order: int
    terms: list                  # terms[k] = 2x2 matrix, coefficient of eps**k
    q_total: sp.Matrix
    amplitudes: list
    axes: list                   # (n_k . sigma) for each ECD, as 2x2 matrices
    algebra: Algebra

    @property
    def generator(self) -> sp.Matrix:
        r""":math:`G = \log(\mathcal{Q}^\dagger U)`, summed over retained orders."""
        return self.algebra.norm(sum(self.terms[1:], sp.zeros(2, 2)))

    @property
    def h_eff(self) -> sp.Matrix:
        r""":math:`H_{\rm eff} = i G`, Hermitian, with :math:`U = \mathcal{Q}e^{-iH_{\rm eff}}`."""
        return self.algebra.norm(sp.I * self.generator)

    def at_order(self, k: int) -> sp.Matrix:
        r""":math:`H_{\rm eff}` contribution at exactly order :math:`k`."""
        return self.algebra.norm(sp.I * self.terms[k])

    def pauli(self, k: int | None = None):
        r"""Pauli decomposition of :math:`H_{\rm eff}` (or of order ``k`` alone).

        Returns ``(h_0, h_x, h_y, h_z)`` with
        :math:`H_{\rm eff} = h_0 + h_x\sigma_x + h_y\sigma_y + h_z\sigma_z`,
        each :math:`h_i` an operator on the oscillator.
        """
        mat = self.h_eff if k is None else self.at_order(k)
        out = []
        for key in ("i", "x", "y", "z"):
            prod = SIGMA_SYM[key] * mat
            out.append(normal((prod[0, 0] + prod[1, 1]) / 2))
        return tuple(out)

    def monomials(self, k: int | None = None) -> dict:
        r"""Break :math:`H_{\rm eff}` into normal-ordered monomials.

        Returns ``{(pauli, m, n): coefficient}`` for terms
        :math:`\text{coeff}\,\sigma_{\rm pauli}\,a^{\dagger m} a^{n}`, with
        ``pauli`` one of ``"1"``, ``"x"``, ``"y"``, ``"z"`` and each coefficient
        factored. Far easier to read than the raw expansion, which spreads a
        single anticommutator over a dozen terms.
        """
        out: dict = {}
        for label, h in zip(("1", "x", "y", "z"), self.pauli(k)):
            h = sp.expand(h)
            if h == 0:
                continue
            for term in sp.Add.make_args(h):
                coeff, m, n = sp.S.One, 0, 0
                for fac in sp.Mul.make_args(term):
                    base, expo = (fac.base, int(fac.exp)) if fac.is_Pow else (fac, 1)
                    if isinstance(base, BosonOp):
                        if base.is_annihilation:
                            n += expo
                        else:
                            m += expo
                    else:
                        coeff *= fac
                key = (label, m, n)
                out[key] = out.get(key, sp.S.Zero) + coeff
        return {key: sp.factor(sp.simplify(val))
                for key, val in out.items() if sp.simplify(val) != 0}

    def q_is_identity(self) -> bool:
        r"""Whether :math:`\mathcal{Q} \propto \mathbb{1}`, i.e. whether
        :math:`\log U = -i H_{\rm eff}` up to a global phase."""
        q = sp.simplify(self.q_total)
        return bool(sp.simplify(q - q[0, 0] * sp.eye(2)).is_zero_matrix)

    def latex(self, k: int | None = None, terms_only: bool = False) -> str:
        r"""LaTeX for :math:`H_{\rm eff}` in Pauli form."""
        h0, hx, hy, hz = self.pauli(k)
        pieces = []
        for label, h in ((r"", h0), (r"\sigma_x", hx),
                         (r"\sigma_y", hy), (r"\sigma_z", hz)):
            if h == 0:
                continue
            body = sp.latex(sp.nsimplify(h))
            pieces.append(rf"\left({body}\right)" + (rf"\,{label}" if label else ""))
        if not pieces:
            return "0"
        body = r"\;+\;".join(pieces)
        if terms_only:
            return body
        lhs = (r"H_{\rm eff}" if k is None
               else rf"H_{{\rm eff}}^{{({k})}}")
        return lhs + r" \;=\; " + body

    def numeric(self, subs: dict, n_fock: int, order: int | None = None):
        r""":math:`H_{\rm eff}` as a ``(2 n_fock, 2 n_fock)`` array."""
        upto = self.order if order is None else order
        out = np.zeros((2 * n_fock, 2 * n_fock), dtype=complex)
        for k in range(1, upto + 1):
            if self.algebra.is_zero(self.terms[k]):
                continue
            out = out + num_mat(sp.I * self.terms[k], n_fock, subs)
        return out

    def convergence(self, subs: dict, n_fock: int, psi0=None) -> dict:
        r"""State-space error of :math:`e^{-iH_{\rm eff}}` against the exact product.

        Operator-norm error is the wrong diagnostic: :math:`H_{\rm eff}` carries
        :math:`a^{\dagger m}a^n` terms whose matrix elements grow with the
        truncation, so ``max|exp(-iH) - U|`` is dominated by Fock levels the
        state never reaches. This compares
        :math:`\lVert e^{-iH_{\rm eff}}\ket{\psi_0} - \mathcal{Q}^\dagger U\ket{\psi_0}\rVert`
        at each truncation order.

        Returns ``{order: error}``.
        """
        from scipy.linalg import expm

        if psi0 is None:
            psi0 = np.zeros(2 * n_fock, dtype=complex)
            psi0[0] = 1.0
        exact = np.eye(2 * n_fock, dtype=complex)
        eye = np.eye(n_fock, dtype=complex)
        for axis, amp in zip(self.axes, self.amplitudes):
            gen = num_mat(axis * (displacement_generator(amp) / 2), n_fock, subs)
            exact = expm(gen) @ exact
        ref = exact @ psi0
        return {k: float(np.linalg.norm(
                    expm(-1j * self.numeric(subs, n_fock, order=k)) @ psi0 - ref))
                for k in range(1, self.order + 1)}


def sequence_generators(spec, *, order: str = "product"):
    r"""Strip the rotations out of an ECD sequence.

    Every qubit rotation can be conjugated away, leaving each conditional
    displacement as the exponential of a *single* generator along a rotated
    Bloch axis. Walking the sequence in application order and accumulating the
    qubit-only factors :math:`W`,

    .. math::
        U = \mathcal{Q}\prod_{k=N}^{1}
            \exp\!\Big[\tfrac{\hat n_k\cdot\vec\sigma}{2}
            (\alpha_k a^\dagger - \bar\alpha_k a)\Big],
        \qquad \hat n_k\cdot\vec\sigma = W_k^\dagger \sigma_z W_k ,

    with :math:`W_k` the accumulated qubit unitary *before* the ``k``-th ECD
    and :math:`\mathcal{Q} = W_{N+1}` whatever is left at the end. Each ECD
    also advances :math:`W \to \sigma_x W`, since
    :math:`\mathrm{ECD}(\alpha) = \sigma_x\,\mathrm{CD}(\alpha)`.

    Parameters
    ----------
    spec : sequence
        Entries are ``("ecd", amp)``, ``("cd", amp)``, ``("rx"|"ry"|"rz", angle)``,
        ``("rot", theta, phi)``, ``("axis", (nx, ny, nz), angle)``,
        ``("qubit", M)``, or a bare 2x2 SymPy matrix.
    order : {"product", "applied"}
        ``"product"`` (default) reads the list left to right **as the operator
        product is written**, so the last entry acts on the state first --
        copy the formula straight across. ``"applied"`` reads it in the order
        the gates are applied.

    Returns
    -------
    generators : list of 2x2 matrices
        Ordered first-applied-first, ready for :func:`bch_product`.
    q_total : 2x2 matrix
        The leftover pure-qubit unitary :math:`\mathcal{Q}`.
    amplitudes : list
        The ECD amplitudes, in the same order as ``generators``.
    axes : list of 2x2 matrices
        Each :math:`\hat n_k\cdot\vec\sigma`.
    """
    ops = [_op_from_spec(item) for item in spec]
    if order == "product":
        ops = ops[::-1]
    elif order != "applied":
        raise ValueError(f"order must be 'product' or 'applied', got {order!r}")

    w = sp.eye(2)
    gens, amps, axes = [], [], []
    for kind, payload in ops:
        if kind == "qubit":
            w = sp.simplify(payload * w)
            continue
        axis = sp.simplify(w.conjugate().T * SIGMA_SYM["z"] * w)
        axes.append(axis)
        amps.append(payload)
        gens.append(axis * (displacement_generator(payload) / 2))
        if kind == "ecd":
            w = sp.simplify(SIGMA_SYM["x"] * w)
    return gens, sp.simplify(w), amps, axes


def effective_hamiltonian(spec, order: int = 3, *,
                          sequence_order: str = "product") -> EffectiveHamiltonian:
    r"""Compute :math:`\log U` of an ECD sequence to a given order.

    Factors the sequence as :math:`U = \mathcal{Q}\,e^{-i H_{\rm eff}}` and
    expands :math:`H_{\rm eff}` to ``order`` in the ECD displacement amplitudes.

    Example
    -------
    >>> import sympy as sp
    >>> a1, a2 = sp.symbols('alpha_1 alpha_2', complex=True)
    >>> res = effective_hamiltonian(
    ...     [("ecd", a2), ("rx", sp.pi), ("ecd", a1)], order=2)
    >>> res.order
    2

    Notes
    -----
    :math:`\mathcal{Q}` is reported separately rather than folded in, and this
    is not a cosmetic choice. The grading needs every merged generator to be
    :math:`O(\alpha)`; a leftover qubit rotation is :math:`O(1)`, and nesting
    it into the BCH series would put infinitely many terms at each order in
    :math:`\alpha`. So :math:`H_{\rm eff}` is the generator in the frame
    :math:`\mathcal{Q}` defines -- the usual toggling-frame effective
    Hamiltonian. When :meth:`EffectiveHamiltonian.q_is_identity` is true,
    :math:`\log U = -i H_{\rm eff}` up to a global phase.

    Because the qubit axes of successive ECDs are generally non-parallel, the
    algebra does not close and this series is *asymptotic*, not convergent for
    arbitrary amplitude. Call :meth:`EffectiveHamiltonian.convergence` with the
    amplitudes you care about before trusting a truncation: empirically it
    behaves well for :math:`|\alpha| \lesssim 1` and fails by
    :math:`|\alpha| \approx 2`.
    """
    alg = Algebra("matrix")
    gens, q_total, amps, axes = sequence_generators(
        spec, order=sequence_order)
    if not gens:
        raise ValueError("sequence contains no ECD or CD entries")
    terms = bch_product(gens, order, alg)
    return EffectiveHamiltonian(order=order, terms=terms, q_total=q_total,
                                amplitudes=amps, axes=axes, algebra=alg)


def ecd_sequence_spec(thetas, phis, betas, *, theta_exact=None) -> list:
    r"""Build an :func:`effective_hamiltonian` spec from optimized parameters.

    Lays the gates out in *product* order, matching how the circuit is written,

    .. math::
        U = R(\theta_{N+1},\phi_{N+1})\,\mathrm{ECD}(\beta_N)\,
            R(\theta_N,\phi_N)\cdots \mathrm{ECD}(\beta_1)\,R(\theta_1,\phi_1),

    so it can be handed straight to :func:`effective_hamiltonian` (whose default
    ``sequence_order="product"`` reads left to right).

    Parameters
    ----------
    thetas, phis : array_like
        The ``N+1`` rotation angles and phases.
    betas : array_like
        The ``N`` complex ECD amplitudes.
    theta_exact : sympy expression, optional
        Substituted for every ``theta``, e.g. ``sp.pi / 2``. Use this for a
        frozen-angle sequence so the rotations stay exact instead of picking up
        floating-point noise, which keeps the resulting axes recognisable
        (``n_z`` comes out as an exact ``0`` rather than ``1e-17``).
    """
    thetas = np.asarray(thetas)
    phis = np.asarray(phis)
    betas = np.asarray(betas, dtype=complex)
    n_gates = len(betas)

    def _theta(i):
        return sp.sympify(thetas[i]) if theta_exact is None else theta_exact

    spec = [("rot", _theta(n_gates), sp.sympify(float(phis[n_gates])))]
    for i in range(n_gates - 1, -1, -1):
        spec.append(("ecd", sp.sympify(complex(betas[i]))))
        spec.append(("rot", _theta(i), sp.sympify(float(phis[i]))))
    return spec


def bloch_axis(axis_matrix) -> tuple:
    r"""Components :math:`(n_x, n_y, n_z)` of a ``2x2`` matrix
    :math:`\hat n \cdot \vec\sigma`."""
    out = []
    for comp in "xyz":
        prod = SIGMA_SYM[comp] * axis_matrix
        out.append(sp.simplify((prod[0, 0] + prod[1, 1]) / 2))
    return tuple(out)
