"""The shared operator module and its re-exports from the modules that used to own them."""

import numpy as np

import gkp_optimal_control  # noqa: F401  (enables 64-bit JAX)
from gkp_optimal_control import operators as op

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


def test_displacement_builder_is_cached_and_shared():
    assert op.make_displacement(N_FOCK) is op.make_displacement(N_FOCK)
    alpha = 0.4 - 0.3j
    np.testing.assert_allclose(op.displacement(N_FOCK, alpha), op.make_displacement(N_FOCK)(alpha))
