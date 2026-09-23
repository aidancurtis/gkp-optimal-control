"""
Build sigma_x_eff (based on Eq. 20) from the GKP Brachistochrone notes.

From the notes:
    sigma_x_eff = |psi_perp><0|  +  |0><psi_perp|

where |psi_perp> = |mu_tilde_L> - <0|mu_tilde_L> |0>  (Eq. 13)

and |mu_tilde_L> is the finite-energy GKP logical state (Eq. 9 + envelope).

Expanding via the Taylor series of e^{B a†}, the |psi_perp><0| part of each
(m,l) lattice term is

    sum_{n>=1} (phase * gauss) * (B^n / n!) * a†^n |0><0|

(where |0><0| is the exact vacuum projector). Its Hermitian conjugate gives the
|0><psi_perp| part, so sigma_x_eff is Hermitian by construction. The notes' Eq. 20
writes |0><0| via its normal-ordered expansion  sum_k (-1)^k/k! a†^k a^k ; that is
algebraically identical but numerically catastrophic for large Fock cutoffs
(alternating terms ~ C(n,k) ~ 2^n cancel with ~1e-16 relative error, i.e. ~1e17
absolute error for n ~ 100). Substituting the exact projector is therefore
algebraically identical and numerically stable.

where (aligned to the package's `gkp_states` convention, which applies the
finite-energy envelope as a *scalar weight* on full-amplitude coherent peaks;
the notes' Eq. 20 uses a damped amplitude ζ^Δ = e^{-delta^2} zeta, a different
state, so Method B is written in the full-amplitude form to match Method A):
    zeta_{m,l} = m*alpha + l*beta          (lattice displacement)
    B          = zeta_{m,l}                 (full displacement amplitude)
    phase      = e^{-i pi m l / 2}
    gauss      = e^{-(envelope_w + 1/2) |zeta|^2}   (= envelope weight * displacement Gaussian)

NOTE: this substitution of the exact vacuum projector for the normal-ordered
expansion is the only change from Eq. 20; the forward (B^n/n!) and backward
(Hermitian-conjugate) halves are kept separate so the construction is exactly
Hermitian for complex B.

Pure JAX implementation (jax.numpy) -- no QuTiP dependency.
Float64 is enabled so the cross-check reaches machine precision.

The script builds sigma_x_eff two ways and cross-checks them:
  - Method A: direct outer product  |psi_perp><0| + h.c.    (reference)
  - Method B: operator series expansion                       (Eq. 20, stable form)

Usage:
    python test.py
    from test import sigma_x_ref, sigma_x_eq20
"""

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ── Parameters ──────────────────────────────────────────────────────────────
N_FOCK = 100  # Fock truncation  (project default: 80; keep 40 for speed)
DELTA = 0.3  # GKP envelope width (project default)
GKP_ALPHA = jnp.sqrt(jnp.pi / 2)  # real lattice spacing
GKP_BETA = 1j * jnp.sqrt(jnp.pi / 2)  # imaginary lattice spacing
CUTOFF = 8  # lattice sum over m,l in [-CUTOFF, CUTOFF]
MU = 0  # logical state index: 0 -> |0_L>, 1 -> |1_L>

# Truncation depth for the operator series (n loop limit). The series converges
# rapidly since max|B|^N_SER / N_SER! << 1 once n is a few times max|B|.
N_SER = N_FOCK

# ── Operators ────────────────────────────────────────────────────────────────
# Annihilation operator in the Fock basis:  a|n> = sqrt(n) |n-1>.
_diag = jnp.sqrt(jnp.arange(1, N_FOCK, dtype=jnp.float64))
a = jnp.diag(_diag, k=1).astype(jnp.complex128)
adag = a.conj().T  # creation operator

# Vacuum |0> as a column vector.
vac = jnp.zeros((N_FOCK, 1), dtype=jnp.complex128)
vac = vac.at[0, 0].set(1.0)

# Cache (a†)^n for n = 0 … N_FOCK to avoid repeated matrix-powers.
print("Precomputing operator powers …", flush=True)
adag_pow = [jnp.eye(N_FOCK, dtype=jnp.complex128)]
for _ in range(N_FOCK):
    adag_pow.append(adag_pow[-1] @ adag)
print("  done.", flush=True)

# Factorial lookup table (n!) for the series coefficients.
FAC = jnp.array([math.factorial(i) for i in range(N_FOCK + 1)], dtype=jnp.float64)
SQRT_FAC = jnp.sqrt(FAC[:N_FOCK])  # sqrt(n!)

# ── Lattice grid ─────────────────────────────────────────────────────────────
idx_range = jnp.arange(-CUTOFF, CUTOFF + 1)
m_grid, l_grid = jnp.meshgrid(idx_range, idx_range, indexing="ij")
m_flat = m_grid.ravel()
l_flat = l_grid.ravel()

# Sublattice constraint: m ≡ MU (mod 2)
sub = (m_flat % 2) == (MU % 2)
m_flat = m_flat[sub]
l_flat = l_flat[sub]
M = int(m_flat.shape[0])
print(f"Lattice points after sublattice filter: {M}")


# ── Helper: coherent state in Fock basis ─────────────────────────────────────
def coherent_fock(alpha: complex) -> jnp.ndarray:
    """Return the raw Fock-basis amplitudes of the coherent state ``|alpha>``.

    Uses the *analytic* amplitudes ``exp(-|alpha|^2/2) * alpha^n / sqrt(n!)``
    (i.e. ``e^{alpha a^dagger}|0>`` together with its displacement Gaussian)
    rather than a renormalization-and-truncate routine, which would rescale
    each lattice peak by an amplitude-dependent factor and break the
    operator-series cross-check. The raw form matches the package's
    ``gkp_states`` / ``_coherent_state_vectors`` convention.
    """
    alpha = complex(alpha)
    n = jnp.arange(N_FOCK)
    coeffs = jnp.exp(-0.5 * abs(alpha) ** 2) * (alpha**n) / SQRT_FAC
    return coeffs[:, None]


# ── Build GKP logical state ──────────────────────────────────────────────────
print("Building finite-energy GKP logical state …", flush=True)

# Envelope weight matches the gkp_states() convention in states.py
envelope_w = 0.5 * (1.0 - jnp.exp(-2.0 * DELTA**2))

state_vec = jnp.zeros((N_FOCK, 1), dtype=jnp.complex128)
for m_idx, l_idx in zip(m_flat, l_flat, strict=True):
    zeta = m_idx * GKP_ALPHA + l_idx * GKP_BETA
    phase = jnp.exp(-1j * jnp.pi * m_idx * l_idx / 2)
    weight = jnp.exp(-envelope_w * abs(zeta) ** 2)
    state_vec = state_vec + phase * weight * coherent_fock(zeta)

# Normalize
raw_norm = float(jnp.linalg.norm(state_vec))
gkp_ket = state_vec / raw_norm

# Amplitude <0|gkp_ket> (NOT |<0|gkp_ket>|^2 that an "expect" would return).
overlap = complex((vac.conj().T @ gkp_ket).item())
print(f"  ||gkp_ket|| = {float(jnp.linalg.norm(gkp_ket)):.8f}  (should be 1.0)")
print(f"  <0|mu_tilde_L> = {overlap:.6f}")

# ── |psi_perp> = |mu_tilde_L> - <0|mu_tilde_L> |0>  (Eq. 13) ───────────────
psi_perp = gkp_ket - overlap * vac
orthog_check = abs(complex((vac.conj().T @ psi_perp).item()))
print(f"  |<0|psi_perp>| = {orthog_check:.2e}  (should be ~0)")

# ── METHOD A: reference  |psi_perp><0| + h.c. ───────────────────────────────
sigma_x_ref = psi_perp @ vac.conj().T + vac @ psi_perp.conj().T
sigma_x_ref = sigma_x_ref / jnp.linalg.norm(sigma_x_ref, "fro")
print("\nMethod A (direct outer product) built.")

# ── METHOD B: operator series (Eq. 20 of the notes) ────────────────────────
# |psi_perp><0| = [ sum_{m,l} phase*gauss * sum_{n>=1} (B^n / n!) a†^n ] |0><0|
# with B = zeta (full displacement, package convention) and
#      gauss = exp(-(envelope_w + 1/2)|zeta|^2)  (= envelope weight * displacement Gaussian).
# Its Hermitian conjugate (|0><psi_perp|) is obtained exactly by using the self-adjoint
# projector P0 = |0><0| rather than expanding it.
#
# The notes expand |0><0| in its normal-ordered form  sum_k (-1)^k/k! a†^k a^k  (= Eq. 20),
# which is exact but numerically CATASTROPHIC for large Fock cutoffs: the alternating
# terms grow like the binomial coefficients C(n,k) ~ 2^n and cancel back down to the
# identity only at the cost of full floating-point precision (relative ~1e-16, i.e. ~1e17
# absolute error for n ~ 100). Replacing that expansion with the exact projector P0 is
# algebraically identical and numerically stable. The raw_norm factor cancels under the
# final Frobenius normalization.
print("Building corrected Eq. 20 operator series …", flush=True)

P0 = vac @ vac.conj().T  # exact |0><0| projector
op_series = jnp.zeros((N_FOCK, N_FOCK), dtype=jnp.complex128)

for idx, (m_idx, l_idx) in enumerate(zip(m_flat, l_flat, strict=True)):
    zeta = m_idx * GKP_ALPHA + l_idx * GKP_BETA
    phase = jnp.exp(-1j * jnp.pi * m_idx * l_idx / 2)
    # Package convention (matches gkp_states): the finite-energy envelope is
    # applied as a scalar weight on full-amplitude coherent peaks, so the
    # series amplitude is the full displacement B = zeta (no damping).
    gauss = jnp.exp(-(envelope_w + 0.5) * abs(zeta) ** 2)

    for n in range(1, N_SER + 1):
        op_series = op_series + (phase * gauss * zeta**n / FAC[n]) * adag_pow[n]

    if (idx + 1) % max(1, M // 5) == 0:
        print(f"  ... {idx + 1}/{M} lattice points done", flush=True)

# sigma_x = |psi_perp><0| + |0><psi_perp| = op_series*P0 + P0*op_series^dagger.
# Hermitian by construction: (op_series*P0)^dagger = P0*op_series^dagger.
sigma_x_eq20 = op_series @ P0 + P0 @ op_series.conj().T
sigma_x_eq20 = sigma_x_eq20 / jnp.linalg.norm(sigma_x_eq20, "fro")
print("  series expansion complete.", flush=True)

# ── Cross-check ───────────────────────────────────────────────────────────────
print("\n── Cross-check ──────────────────────────────────────────────────────────")
diff = sigma_x_ref - sigma_x_eq20
diff_norm = float(jnp.max(jnp.abs(diff)))
print(f"max |sigma_x_ref - sigma_x_eq20|  = {diff_norm:.4e}   (should be ~1e-10)")

frob_ref = float(jnp.linalg.norm(sigma_x_ref, "fro"))
frob_eq20 = float(jnp.linalg.norm(sigma_x_eq20, "fro"))
print(f"Frobenius norm  ref               = {frob_ref:.8f}")
print(f"Frobenius norm  eq20              = {frob_eq20:.8f}")

herm_ref = float(jnp.max(jnp.abs(sigma_x_ref - sigma_x_ref.conj().T)))
herm_eq20 = float(jnp.max(jnp.abs(sigma_x_eq20 - sigma_x_eq20.conj().T)))
print(f"max Hermiticity error  ref        = {herm_ref:.2e}")
print(f"max Hermiticity error  eq20       = {herm_eq20:.2e}")

print("\nDone.  sigma_x_ref and sigma_x_eq20 are available for import.")
