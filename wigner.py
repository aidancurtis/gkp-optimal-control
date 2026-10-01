import jax.numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import gammaln, eval_laguerre

# ----------------------------- parameters -----------------------------
DIM    = 80
DELTA  = 0.3
CUTOFF = 30
ALPHA  = np.sqrt(np.pi / 2)
BETA   = 1j * np.sqrt(np.pi / 2)
OMEGA  = 1.0

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 11,
                     "axes.titlesize": 12, "figure.dpi": 110, "axes.grid": False,
                     "grid.alpha": 0.25, "legend.frameon": False})

# ----------------------- GKP states in Fock basis ----------------------
def lattice_moments(nmax, mu, alpha_v, beta_v, delta_v, cutoff):
    S = np.zeros(nmax + 1, complex)
    powers = np.arange(nmax + 1)
    for jj in range(-cutoff, cutoff + 1):
        for ll in range(-cutoff, cutoff + 1):
            m = 2 * jj + mu
            z = m * alpha_v + ll * beta_v
            weight = np.exp(-1j * np.pi * m * ll / 2) * np.exp(-abs(z) ** 2 / 2)
            zd = np.exp(-delta_v ** 2) * z
            if zd == 0:
                pw = np.zeros(nmax + 1, complex); pw[0] = 1.0
            else:
                pw = np.exp(powers * np.log(abs(zd)) + 1j * powers * np.angle(zd))
            S += weight * pw
    return S

def gkp_fock(dim, mu):
    S = lattice_moments(dim - 1, mu, ALPHA, BETA, DELTA, CUTOFF)
    v = S[:dim] / np.exp(0.5 * gammaln(np.arange(dim) + 1.0))
    return v / np.linalg.norm(v)

psi0 = gkp_fock(DIM, 0)   # |0_L>  (initial)
psi1 = gkp_fock(DIM, 1)   # |1_L>  (target)

# --------------- time-optimal generator  psi_i -> psi_f ----------------
def brachistochrone_H(psi_i, psi_f, omega=OMEGA):
    s   = np.vdot(psi_i, psi_f)
    print(s)
    a   = min(abs(s), 1.0)
    phi = np.angle(s)
    R   = 1.0 / np.sqrt(1.0 - a ** 2)
    perp = R * (psi_f - s * psi_i)
    X = 1j * omega * np.exp(-1j * phi) * np.outer(psi_i, perp.conj())
    H = X + X.conj().T
    info = dict(s=s, phi=float(phi), R=float(R),
                theta_B=float(np.arccos(a)), T_star=float(np.arccos(a)) / omega, perp=perp)
    return H, info

def truncate_state(psi, D):
    v = psi.copy(); v[D + 1:] = 0.0
    return v / np.linalg.norm(v)

def truncated_H(psi_i, psi_f, D, omega=OMEGA):
    """Degree-D drive: rebuild the geodesic generator from Fock-<=D states."""
    H, _ = brachistochrone_H(truncate_state(psi_i, D), truncate_state(psi_f, D), omega)
    return H

H, REF = brachistochrone_H(psi0, psi1)
T_STAR = REF["T_star"]

# ----------------------- fast evolution helpers ------------------------
def trajectory(H, times, psi_i):
    w, V = np.linalg.eigh(H); c = V.conj().T @ psi_i
    return [V @ (np.exp(-1j * w * t) * c) for t in times]

def peak_time(H, psi_i, psi_f, t_hint, n=400):
    w, V = np.linalg.eigh(H); Vd = V.conj().T
    ci, cf = Vd @ psi_i, Vd @ psi_f
    ts = np.linspace(0.0, 1.5 * t_hint, n)
    amp = (np.conj(cf)[None, :] * ci[None, :] * np.exp(-1j * np.outer(ts, w))).sum(1)
    f = np.abs(amp) ** 2; k = int(np.argmax(f))
    return ts[k], f[k]

# ---------------------- verification of exact gate ---------------------
sv   = np.linalg.svd(H, compute_uv=False)
psiT = trajectory(H, [T_STAR], psi0)[0]
print(f"|s|=|<0_L|1_L>|  = {abs(REF['s']):.2e}   (parity-orthogonal -> ~0)")
print(f"theta_B          = {REF['theta_B']:.6f}   T* = {T_STAR:.6f}")
print(f"Hermiticity      = {np.abs(H - H.conj().T).max():.2e}   (->0)")
print(f"||H||_2          = {sv[0]:.12f}   (-> OMEGA = {OMEGA})")
print(f"rank-2  s3/s1    = {sv[2] / sv[0]:.2e}   (->0)")
print(f"transfer fidelity |<1_L|U(T*)|0_L>|^2 = {abs(np.vdot(psi1, psiT))**2:.12f}  (->1)\n")

# ------------------- D-sweep: quantitative convergence -----------------
D_LIST = (2, 4, 8, 16, 32)
print("degree-D truncated drive: peak fidelity to the true |1_L>")
for D in D_LIST:
    tp, fp = peak_time(truncated_H(psi0, psi1, D), psi0, psi1, T_STAR)
    print(f"  D={D:2d}:  F_peak = {fp:.6f}   at t/t* = {tp / T_STAR:.3f}")

# ------------------------------- Wigner --------------------------------
def hermite_funcs(nmax, x):
    x = np.asarray(x, float)
    Hf = np.empty((nmax + 1,) + x.shape)
    Hf[0] = np.pi ** -0.25 * np.exp(-x ** 2 / 2)
    if nmax >= 1: Hf[1] = np.sqrt(2.0) * x * Hf[0]
    for n in range(2, nmax + 1):
        Hf[n] = np.sqrt(2.0 / n) * x * Hf[n - 1] - np.sqrt((n - 1) / n) * Hf[n - 2]
    return Hf

def wigner_numpy(psi, xg, pg, ny=301, pad=2.5):
    nmax = len(psi) - 1
    ymax = max(abs(xg[0]), abs(xg[-1])) + pad
    yg = np.linspace(-ymax, ymax, ny)
    Hp = hermite_funcs(nmax, xg[:, None] + yg[None, :])
    Hm = hermite_funcs(nmax, xg[:, None] - yg[None, :])
    M = np.conj(np.tensordot(psi, Hp, (0, 0))) * np.tensordot(psi, Hm, (0, 0))
    return np.real(M @ np.exp(2j * np.outer(pg, yg)).T).T / np.pi * (yg[1] - yg[0])

# ------------------------- snapshot grid figure ------------------------
frac   = np.array([0.0, 0.34, 0.67, 1.0])
xw = pw = np.linspace(-5.5, 5.5, 121)
panels = [("exact", H)] + [(rf"$D={D}$", truncated_H(psi0, psi1, D)) for D in D_LIST]

W_tgt = wigner_numpy(psi1, xw, pw)
vmax  = float(np.abs(W_tgt).max())
norm  = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

fig, axes = plt.subplots(len(panels), len(frac),
                         figsize=(11.0, 2.75 * len(panels)),
                         constrained_layout=True, squeeze=False)
for row, (label, Hr) in enumerate(panels):
    tp, _ = peak_time(Hr, psi0, psi1, T_STAR)
    psis  = trajectory(Hr, frac * tp, psi0)
    for col, fr in enumerate(frac):
        ax = axes[row, col]
        ax.pcolormesh(xw, pw, wigner_numpy(psis[col], xw, pw), cmap="RdBu_r",
                      norm=norm, shading="gouraud", rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([-4, 0, 4]); ax.set_yticks([-4, 0, 4])
        if row == 0: ax.set_title(rf"$t/t_\star={fr:.2f}$", fontsize=10)
        if col == 0: ax.set_ylabel(label + "\n$p$", fontsize=10)
    ax = axes[row, -1]
    ax.set_aspect("equal"); ax.set_xticks([-4, 0, 4]); ax.set_yticks([-4, 0, 4])
for col in range(len(frac)):
    axes[-1, col].set_xlabel("$x$", fontsize=10)
plt.savefig("wigner_trajectories_0L_1L.pdf")
plt.show()
