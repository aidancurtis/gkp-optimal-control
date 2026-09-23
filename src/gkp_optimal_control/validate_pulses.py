r"""NumPy reference simulator and convention checks for :mod:`pulses`.

The JAX simulator in :mod:`pulse_simulation` is the production path; this file
is an independent NumPy implementation of the same displaced-frame Hamiltonian,
used to pin down sign conventions and to cross-check the JAX version. Keeping
two independent implementations is the same validation pattern used elsewhere
in the package.

What is checked
---------------
1. ``D(alpha) = exp(-i * integral(eps) * ...)``: a bare Gaussian realizes the
   intended displacement.
2. A single compiled ECD pulse realizes ``sigma_x CD(beta) exp(i theta' sz/2)``
   with ``CD(beta) = D(beta/2)|g><g| + D(-beta/2)|e><e|``, i.e. exactly
   ``gate_optimization.apply_ecd`` with ``echoed=True``.
3. A compiled rotation realizes ``gate_optimization.qubit_rotation``.
4. A full multi-gate ECD sequence reproduces the ideal circuit, and the
   virtual-Z frame sign is the one used by :func:`pulses.compile_ecd_sequence`.
5. The selective SNAP waveform imparts the intended per-Fock phases.

Run with ``python validate_pulses.py``. A larger ``chi`` than the Table S1
default is used for the SNAP test so the selective pulses stay short enough to
simulate quickly.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import pulses as P

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Operators (kron(cavity, transmon), index = n_cav * n_tr + s_transmon)
# ---------------------------------------------------------------------------


def operators(n_fock: int, n_tr: int = 2):
    a_c = np.diag(np.sqrt(np.arange(1, n_fock)), k=1).astype(complex)
    b_t = np.diag(np.sqrt(np.arange(1, n_tr)), k=1).astype(complex)
    i_c = np.eye(n_fock, dtype=complex)
    i_t = np.eye(n_tr, dtype=complex)
    a = np.kron(a_c, i_t)
    b = np.kron(i_c, b_t)
    return {
        "a": a,
        "adag": a.conj().T,
        "b": b,
        "bdag": b.conj().T,
        "n_c": a.conj().T @ a,
        "n_t": b.conj().T @ b,
        "a_c": a_c,
        "i_c": i_c,
        "dim": n_fock * n_tr,
    }


def displaced_frame_terms(params: P.SystemParams, n_fock: int, n_tr: int = 2):
    """Static operator plus the operators carrying time-dependent coefficients.

    Returns ``(h_static, ops, coeff_fn)`` where ``coeff_fn(alpha, dalpha, eps,
    omega, delta, kappa_h)`` gives the coefficient of each entry of ``ops``.
    Implements Eq. (S2) of the supplement.
    """
    o = operators(n_fock, n_tr)
    a, adag, b, bdag = o["a"], o["adag"], o["b"], o["bdag"]
    n_c, n_t = o["n_c"], o["n_t"]
    a2 = a @ a
    ad2 = adag @ adag
    quart = ad2 @ a2  # a^dag^2 a^2
    cube = ad2 @ a  # a^dag^2 a
    p = params

    h_static = (
        -p.chi * (n_c @ n_t)
        - p.kerr * quart
        - p.chi_prime * (quart @ n_t)
    )
    if n_tr > 2:
        n_t2 = bdag @ bdag @ b @ b
        h_static = h_static - 0.5 * p.anharm * n_t2

    ops = {
        "n_c": n_c,  # delta - 4 K |alpha|^2
        "n_c_n_t": n_c @ n_t,  # -4 chi' |alpha|^2
        "n_t": n_t,  # -(chi |alpha|^2 + chi' |alpha|^4)
        "lin_re": None,  # filled below
    }

    def hermitian_pair(op):
        return op, op.conj().T

    def coeff_and_ops(alpha, dalpha, eps, omega, delta, kappa_h):
        """Assemble H_tilde(t) for a single time sample."""
        n = abs(alpha) ** 2
        h = h_static.copy()
        h += (delta - 4.0 * p.kerr * n) * n_c
        h += (-4.0 * p.chi_prime * n) * (n_c @ n_t)
        h += (-(p.chi * n + p.chi_prime * n * n)) * n_t

        # conditional force: -(chi + 2 chi' |alpha|^2)(alpha* a + alpha a^dag) n_t
        c_force = -(p.chi + 2.0 * p.chi_prime * n)
        force = c_force * (np.conj(alpha) * a + alpha * adag)
        h += force @ n_t

        # cubic / squeezing terms from Kerr and chi'
        cub = 2.0 * alpha * cube + alpha**2 * ad2
        cub = cub + cub.conj().T
        h += -p.kerr * cub
        h += -p.chi_prime * (cub @ n_t)

        # residual linear term (exactly zero when alpha solves the frame ODE
        # with the same kappa the Hamiltonian carries)
        amp = delta * alpha - 2.0 * p.kerr * n * alpha - 1j * dalpha - 1j * 0.5 * kappa_h * alpha + eps
        h += np.conj(amp) * a + amp * adag

        # transmon drive
        h += np.conj(omega) * b + omega * bdag
        return h

    return coeff_and_ops


def simulate(seq: P.PulseSequence, psi0, n_fock, n_tr=2, kappa_h=0.0, save_every=1):
    """Piecewise-constant propagation of the displaced-frame Hamiltonian.

    Returns ``(psi_final, saved_states, save_indices, alpha)``. ``alpha`` is the
    frame trajectory used, so the lab-frame state is ``D(alpha) psi_tilde``.
    """
    build = displaced_frame_terms(seq.params, n_fock, n_tr)
    p = seq.params
    # frame trajectory consistent with the Hamiltonian's kappa
    frame_params = p.replace(kappa=kappa_h)
    alpha = P.frame_trajectory(seq.eps, frame_params, seq.delta)
    dalpha = np.array(
        [
            P._alpha_rhs(alpha[k], seq.eps[k], 0.0, frame_params, seq.delta)
            for k in range(seq.n_samples)
        ]
    )
    psi = np.asarray(psi0, dtype=complex).reshape(-1)
    saved, idx = [psi.copy()], [0]
    for k in range(seq.n_samples):
        h = build(alpha[k], dalpha[k], seq.eps[k], seq.omega[k], seq.delta, kappa_h)
        psi = expm(-1j * p.dt * h) @ psi
        if (k + 1) % save_every == 0:
            saved.append(psi.copy())
            idx.append(k + 1)
    return psi, np.array(saved), np.array(idx), alpha


# ---------------------------------------------------------------------------
# Ideal circuit mirror of gate_optimization
# ---------------------------------------------------------------------------


def displace_op(n_fock, alpha):
    a_c = np.diag(np.sqrt(np.arange(1, n_fock)), k=1).astype(complex)
    return expm(alpha * a_c.conj().T - np.conj(alpha) * a_c)


def qubit_rotation(theta, phi):
    c, s = np.cos(theta / 2) + 0j, np.sin(theta / 2) + 0j
    return np.array([[c, -1j * np.exp(-1j * phi) * s], [-1j * np.exp(1j * phi) * s, c]])


def ideal_ecd_circuit(betas, thetas, phis, n_fock, psi_cav0):
    """Mirror of ``gate_optimization._build_ecd_sequence`` with echoed=True."""
    psi = np.stack([np.asarray(psi_cav0, dtype=complex), np.zeros(n_fock, complex)])
    out = []

    def rot(psi, theta, phi):
        return np.einsum("ij,jd->id", qubit_rotation(theta, phi), psi)

    def ecd(psi, beta):
        d = displace_op(n_fock, beta / 2)
        return np.stack([d.conj().T @ psi[1], d @ psi[0]])

    psi = rot(psi, thetas[0], phis[0])
    out.append(psi.copy())
    for i, beta in enumerate(betas):
        psi = ecd(psi, beta)
        out.append(psi.copy())
        if i + 1 < len(thetas):
            psi = rot(psi, thetas[i + 1], phis[i + 1])
            out.append(psi.copy())
    return psi, out


def blocks_to_joint(psi_blocks):
    """(2, n_fock) -> joint ket with index = n_cav * 2 + s."""
    return np.swapaxes(np.asarray(psi_blocks), -1, -2).reshape(-1)


def joint_to_blocks(psi, n_fock):
    return np.swapaxes(psi.reshape(n_fock, 2), 0, 1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def undisplace(psi_joint, alpha, n_fock, n_tr=2):
    """Displaced-frame state -> lab-frame state: ``psi = D(alpha) psi_tilde``.

    The frame trajectory does *not* return to the origin at the end of a
    sequence: the compiled amplitude ratios null the mean of the two
    *conditional* trajectories, whereas the frame follows the ground branch
    alone, which keeps rotating at ``delta``. Skipping this step is the single
    easiest way to get a wrong answer out of a displaced-frame simulation.
    """
    d = np.kron(displace_op(n_fock, alpha), np.eye(n_tr))
    return d @ psi_joint


def test_displacement():
    p = P.SystemParams(chi=0.0, chi_prime=0.0, kerr=0.0, kappa=0.0)
    _, area = P.gaussian_envelope(p.sigma_disp, p.n_sigma_disp, p.dt)
    g, _ = P.gaussian_envelope(p.sigma_disp, p.n_sigma_disp, p.dt)
    target = 0.8 - 0.3j
    eps = (1j * target / area) * g
    alpha = P.frame_trajectory(eps, p, delta=0.0)
    err = abs(alpha[-1] - target)
    print(f"[1] Gaussian displacement: alpha(T) = {alpha[-1]:+.6f}, "
          f"target {target:+.6f}, err {err:.2e}")
    assert err < 1e-9


def test_rotation():
    p = P.SystemParams()
    sm = np.array([[0, 1], [0, 0]], complex)
    for theta, phi in [(np.pi / 2, 0.0), (np.pi / 3, 1.1), (np.pi, -2.0)]:
        om = P.rotation_waveform(theta, phi, p)
        u = np.eye(2, dtype=complex)
        for w in om:
            u = expm(-1j * p.dt * (np.conj(w) * sm + w * sm.conj().T)) @ u
        err = np.abs(u - qubit_rotation(theta, phi)).max()
        print(f"[2] R({theta / np.pi:+.3f}pi, {phi / np.pi:+.3f}pi): "
              f"max|U - R| = {err:.2e}")
        assert err < 1e-8


def test_single_ecd(beta=0.9 + 0.4j, alpha0=8.0, n_fock=28, chi_khz=200.0):
    r"""One compiled pulse vs ``sigma_x CD(beta) Z(theta')``.

    ``CD(beta) = D(beta/2)|g><g| + D(-beta/2)|e><e|``, so
    ``sigma_x CD(beta)`` is exactly ``gate_optimization.apply_ecd`` with
    ``echoed=True``. Section S4 B of the supplement writes
    ``beta = alpha_e - alpha_g``; the opposite sign, Eq. (S25), is the one that
    matches, and is what :mod:`pulses` uses.
    """
    p = P.SystemParams(chi=TWO_PI * chi_khz * 1e-3)
    pulse = P.optimize_ecd_pulse(beta, alpha0, p)
    seq = P.PulseSequence(
        eps=pulse.eps, omega=pulse.omega, params=p, delta=0.5 * p.chi,
        segments=[P.PulseSegment("ecd", 0, pulse.eps.size, "ECD")],
        gate_indices=np.array([pulse.eps.size]), meta={"gate_set": "ecd"},
    )
    d = displace_op(n_fock, pulse.beta / 2)
    vac = np.eye(n_fock)[0].astype(complex)
    print(f"[3] single ECD: beta = {pulse.beta:+.4f} (target {beta:+.4f}), "
          f"alpha0' = {pulse.alpha0:.2f}, T = {pulse.eps.size * p.dt * 1e3:.0f} ns")

    ov = {}
    for lbl, blocks in [
        ("g", np.stack([vac, 0 * vac])),
        ("e", np.stack([0 * vac, vac])),
    ]:
        psi, _, _, alpha = simulate(seq, blocks_to_joint(blocks), n_fock)
        sim = joint_to_blocks(undisplace(psi, alpha[-1], n_fock), n_fock)
        cd = np.stack([d @ blocks[0], d.conj().T @ blocks[1]])
        ideal = np.stack([cd[1], cd[0]])  # sigma_x CD(beta)
        ov[lbl] = np.sum(np.conj(ideal) * sim)
        print(f"    input |0,{lbl}>: |<ideal|sim>| = {abs(ov[lbl]):.6f}   "
              f"arg = {np.angle(ov[lbl]):+.6f}")
    measured = (np.angle(ov["g"]) - np.angle(ov["e"]) + np.pi) % (2 * np.pi) - np.pi
    print(f"    theta' measured = {measured:+.6f}   Eq. (S21) = {pulse.theta_prime:+.6f}   "
          f"(ratio {measured / pulse.theta_prime:+.3f})")
    print(f"    |alpha(T)| = {abs(alpha[-1]):.4f}  <- frame does not close")
    assert abs(ov["g"]) > 0.99 and abs(ov["e"]) > 0.99


def test_frame_rule(n_fock=30, n_trials=5):
    r"""The virtual-Z rule, checked by exact matrix algebra.

    The physical gate is ``ECD(beta) Z(theta')``. Since
    ``Z(a) sigma_x = sigma_x Z(-a)``, the leftover Z changes sign each time it
    is commuted past an ECD, so the running frame obe
    ``F -> -F + theta'`` rather than ``F -> F + theta'``. Rotation phases are
    compiled as ``phi - F``.
    """
    print("[4] virtual-Z frame rule (matrix algebra, arbitrary theta')")

    def rot(psi, th, ph):
        return np.einsum("ij,jd->id", qubit_rotation(th, ph), psi)

    def ecd(psi, b):
        d = displace_op(n_fock, b / 2)
        return np.stack([d.conj().T @ psi[1], d @ psi[0]])

    def zed(psi, a):
        return np.stack([np.exp(1j * a / 2) * psi[0], np.exp(-1j * a / 2) * psi[1]])

    worst = 0.0
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 8))
        betas = rng.normal(size=n) + 1j * rng.normal(size=n)
        th = rng.uniform(0, np.pi, n + 1)
        ph = rng.uniform(-np.pi, np.pi, n + 1)
        tp = rng.uniform(-1.5, 1.5, n)
        vac = np.zeros(n_fock, complex)
        vac[0] = 1.0
        psi0 = np.stack([vac, 0 * vac])

        ideal = rot(psi0, th[0], ph[0])
        for i in range(n):
            ideal = ecd(ideal, betas[i])
            ideal = rot(ideal, th[i + 1], ph[i + 1])

        for label, update in (
            ("F -> -F + tp", lambda f, t: -f + t),
            ("F -> F + tp", lambda f, t: f + t),
        ):
            q, f = psi0.copy(), 0.0
            for i in range(n + 1):
                q = rot(q, th[i], ph[i] - f)
                if i < n:
                    q = ecd(q, betas[i])
                    q = zed(q, tp[i])
                    f = update(f, tp[i])
            fid = abs(np.sum(np.conj(ideal[0]) * q[0])) ** 2 / (
                np.linalg.norm(ideal[0]) ** 2 * np.linalg.norm(q[0]) ** 2
            )
            if label.startswith("F -> -F"):
                worst = max(worst, abs(1.0 - fid))
                mark = "  <- used by pulses.py"
            else:
                mark = ""
            print(f"    N = {n}, max|theta'| = {np.abs(tp).max():.2f}, {label:>13}: "
                  f"F_cav|g = {fid:.12f}{mark}")
    print(f"    worst infidelity of the adopted rule: {worst:.2e}")
    assert worst < 1e-10


def test_sequence(n_fock=30, chi_khz=200.0, alpha0=8.0):
    """Full compiled sequence against the ideal circuit."""
    p = P.SystemParams(chi=TWO_PI * chi_khz * 1e-3)
    betas = np.array([0.7 + 0.3j, -1.1 + 0.2j])
    thetas = np.array([np.pi / 2, 0.7 * np.pi, np.pi / 3])
    phis = np.array([0.0, 0.4 * np.pi, -0.3 * np.pi])

    psi_cav0 = np.eye(n_fock)[0].astype(complex)
    psi_ideal, _ = ideal_ecd_circuit(betas, thetas, phis, n_fock, psi_cav0)

    seq = P.compile_ecd_sequence(betas, thetas, phis, alpha0=alpha0, params=p)
    psi, _, _, alpha = simulate(
        seq, blocks_to_joint(np.stack([psi_cav0, np.zeros(n_fock, complex)])), n_fock
    )
    sim = joint_to_blocks(undisplace(psi, alpha[-1], n_fock), n_fock)
    p_g = np.linalg.norm(sim[0]) ** 2
    f_cav = abs(np.sum(np.conj(psi_ideal[0]) * sim[0])) ** 2 / (
        np.linalg.norm(psi_ideal[0]) ** 2 * p_g
    )
    print(f"[5] sequence: {betas.size} ECDs, {thetas.size} rotations, "
          f"T = {seq.duration * 1e3:.0f} ns at chi/2pi = {chi_khz} kHz")
    print(f"    F(cavity | g) = {f_cav:.6f}   P(g) = {p_g:.5f} "
          f"(ideal {np.linalg.norm(psi_ideal[0]) ** 2:.5f})")
    print(f"    |alpha(T)| = {abs(alpha[-1]):.4f}")
    return f_cav


def test_snap(n_fock=16, chi_khz=2000.0, dt_ns=2.0, n_periods=6):
    r"""Selective pulses impart the intended phases on each Fock level.

    Three failure modes this test exists to catch.

    An inverted carrier sign leaves only :math:`n = 0` resonant, so high Fock
    levels pass through untouched. Too short a pulse breaks selectivity,
    because the multiplexed components add coherently at the pulse centre.

    And the one that actually destroys sequences: two :math:`\pi` pulses impart
    :math:`-e^{i\theta_n}`, whereas ``gate_optimization`` pads unoptimized
    levels with :math:`\theta_n = 0`, i.e. :math:`+1`. Any population above
    ``n_drive`` therefore sits :math:`\pi` out of phase with the driven block.
    The second half of this test measures that directly on a coherent state
    with a fifth of its population above ``n_drive``.
    """
    p = P.SystemParams(
        chi=TWO_PI * chi_khz * 1e-3, dt=dt_ns * 1e-3, n_selective_periods=n_periods
    )
    n_snap = 6
    phases = np.array([0.0, 0.9, -1.7, 2.4, 0.5, -0.8])

    def make_seq(n_drive, correction=True):
        eps, om = P.snap_waveform(
            phases, p, n_drive=n_drive, match_padded_identity=correction
        )
        return P.PulseSequence(
            eps=eps, omega=om, params=p, delta=0.0,
            segments=[P.PulseSegment("snap", 0, om.size, "S")],
            gate_indices=np.array([om.size]), meta={"gate_set": "snap"},
        )

    seq = make_seq(n_snap)
    print(f"[6] SNAP at chi/2pi = {chi_khz} kHz, t_pi = "
          f"{p.t_snap_selective * 1e3:.0f} ns ({n_periods} x 2 pi / chi)")
    got, pops = [], []
    for n in range(n_snap):
        blocks = np.stack([np.eye(n_fock)[n], np.zeros(n_fock)]).astype(complex)
        psi, _, _, alpha = simulate(seq, blocks_to_joint(blocks), n_fock)
        sim = joint_to_blocks(undisplace(psi, alpha[-1], n_fock), n_fock)
        amp = sim[0][n]
        got.append(np.angle(amp))
        pops.append(np.linalg.norm(sim[0]) ** 2)
        print(f"    n = {n}: P(g) = {pops[-1]:.4f}  |<n|psi_g>| = {abs(amp):.4f}  "
              f"arg = {np.angle(amp):+.4f}  target {phases[n]:+.4f}")
    err = (np.array(got) - phases + np.pi) % (2 * np.pi) - np.pi
    print(f"    max absolute phase error: {np.abs(err).max():.4f} rad   "
          f"min P(g): {min(pops):.4f}")
    print("    (both fall roughly as 1 / t_pi; see snap_waveform)")
    assert np.abs(err).max() < 0.1

    # --- the padded-identity convention, on a state with a real tail
    alpha_c = 2.0
    psi0 = np.zeros(n_fock, dtype=complex)
    term = np.exp(-0.5 * alpha_c**2)
    for n in range(n_fock):
        psi0[n] = term
        term = term * alpha_c / np.sqrt(n + 1)
    psi0 /= np.linalg.norm(psi0)
    tail = float(np.sum(np.abs(psi0[n_snap:]) ** 2))
    ideal = psi0 * np.exp(1j * np.concatenate([phases, np.zeros(n_fock - n_snap)]))

    print(f"    coherent state, alpha = {alpha_c}: {tail:.1%} of the population "
          f"lies above n_drive = {n_snap}")
    fids = {}
    for n_drive in (n_snap, n_fock):
        for correction in (False, True):
            psi, _, _, alpha = simulate(
                make_seq(n_drive, correction),
                blocks_to_joint(np.stack([psi0, np.zeros(n_fock, complex)])),
                n_fock,
            )
            sim = joint_to_blocks(undisplace(psi, alpha[-1], n_fock), n_fock)
            g = sim[0] / np.linalg.norm(sim[0])
            f = float(abs(np.vdot(ideal, g)) ** 2)
            fids[(n_drive, correction)] = f
            tag = "pi-corrected" if correction else "uncorrected "
            print(f"      n_drive = {n_drive:2d}, {tag}: F = {f:.4f}")
    assert fids[(n_fock, True)] > 0.99
    assert fids[(n_snap, False)] < 0.5  # the failure this test documents
    return fids


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    for fn in (test_displacement, test_rotation, test_frame_rule,
               test_single_ecd, test_sequence, test_snap):
        fn()
        print()
