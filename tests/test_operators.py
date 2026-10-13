"""The shared operator module and its re-exports from the modules that used to own them."""

import numpy as np
import pytest

import gkp_optimal_control  # noqa: F401  (enables 64-bit JAX)
from gkp_optimal_control import ecd_symbolic as es
from gkp_optimal_control import gate_optimization as go
from gkp_optimal_control import hamiltonians as hm
from gkp_optimal_control import operators as op
from gkp_optimal_control import pulse_simulation as ps

N_FOCK = 12


def test_ladder_and_quadrature_operators():
    a, adag, n = op.cavity_operators(N_FOCK)
    np.testing.assert_allclose(a, op.annihilation(N_FOCK))
    np.testing.assert_allclose(np.diag(n), np.arange(N_FOCK), atol=1e-12)
    # [a, a^dag] = 1 away from the truncation edge.
    comm = np.asarray(a @ adag - adag @ a)
    np.testing.assert_allclose(comm[:-1, :-1], np.eye(N_FOCK - 1), atol=1e-12)

    q, p = op.quadrature_operators(N_FOCK)
    np.testing.assert_allclose(q, np.asarray(q).conj().T)
    np.testing.assert_allclose(p, np.asarray(p).conj().T)
    comm = np.asarray(q @ p - p @ q)
    np.testing.assert_allclose(comm[:-1, :-1], 1j * np.eye(N_FOCK - 1), atol=1e-12)


def test_joint_operators_are_cavity_first():
    n_tr = 3
    o = op.joint_operators(N_FOCK, n_tr)
    a_c, b_t = np.asarray(op.annihilation(N_FOCK)), np.asarray(op.annihilation(n_tr))
    np.testing.assert_allclose(o["a"], np.kron(a_c, np.eye(n_tr)))
    np.testing.assert_allclose(o["b"], np.kron(np.eye(N_FOCK), b_t))
    np.testing.assert_allclose(o["adag"], np.asarray(o["a"]).conj().T)
    np.testing.assert_allclose(o["bdag"], np.asarray(o["b"]).conj().T)
    np.testing.assert_allclose(o["a"] @ o["b"], o["b"] @ o["a"], atol=1e-12)


def test_pauli_matrices():
    x, y, z = (np.asarray(m) for m in (op.SIGMA_X, op.SIGMA_Y, op.SIGMA_Z))
    np.testing.assert_allclose(x @ y, 1j * z)
    np.testing.assert_allclose(y @ z, 1j * x)
    np.testing.assert_allclose(z @ x, 1j * y)
    np.testing.assert_allclose(op.qubit_rotation(np.pi, 0.0), -1j * x, atol=1e-15)
    np.testing.assert_allclose(op.qubit_rotation(np.pi, np.pi / 2), -1j * y, atol=1e-15)


def test_displacement_is_the_expm_builder():
    alpha = 0.4 - 0.3j
    d_op = op.displacement(N_FOCK, alpha)
    np.testing.assert_allclose(d_op, op.make_displacement(N_FOCK, "expm")(alpha))
    np.testing.assert_allclose(ps._displace_np(N_FOCK, alpha), d_op)


def test_old_import_paths_still_resolve():
    assert hm.cavity_operators is op.cavity_operators
    assert hm.quadrature_operators is op.quadrature_operators
    assert ps.joint_operators is op.joint_operators
    for name in ("make_displacement", "make_displacement_apply", "make_squeeze",
                 "make_squeeze_apply", "qubit_rotation"):  # fmt: skip
        assert getattr(go, name) is getattr(op, name)


def test_numpy_wrappers_return_numpy():
    assert isinstance(es.annihilation(N_FOCK), np.ndarray)
    assert isinstance(es.qubit_rotation(1.1, -2.0), np.ndarray)
    assert isinstance(es.SIGMA_X, np.ndarray)
    assert isinstance(ps._displace_np(N_FOCK, 0.2), np.ndarray)
    np.testing.assert_allclose(es.qubit_rotation(1.1, -2.0), op.qubit_rotation(1.1, -2.0))


@pytest.mark.parametrize("chi, k, alpha", [(0.7, 0.0, 0.0), (-0.4, 0.05, -1.3)])
def test_cavity_transmon_drift_matches_explicit_form(chi, k, alpha):
    n_tr = 3
    o = op.joint_operators(N_FOCK, n_tr)
    a, adag, b, bdag = (np.asarray(o[key]) for key in ("a", "adag", "b", "bdag"))
    expected = (
        chi * (adag @ a) @ (bdag @ b)
        + 0.5 * k * adag @ adag @ a @ a
        + 0.5 * alpha * bdag @ bdag @ b @ b
    )
    np.testing.assert_allclose(hm.cavity_transmon_drift(N_FOCK, n_tr, chi, k, alpha), expected)
    controls = np.asarray(hm.cavity_transmon_iq_controls(N_FOCK, n_tr))
    np.testing.assert_allclose(controls[0], a + adag)
    np.testing.assert_allclose(controls[3], 1j * (b - bdag))
