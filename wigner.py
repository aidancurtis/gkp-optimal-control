import numpy as np

from gkp_optimal_control.brachistochrone import quantum_brachistochrone_hamiltonian
from gkp_optimal_control.plotting import set_plot_style, plot_wigner_snapshots, plot_wigner
from gkp_optimal_control.states import gkp_states
import matplotlib.pyplot as plt
import jaxquantum as jqt
import jax.numpy as jnp

set_plot_style()

# gkp params
n_fock = 80
gkp_delta = 0.3
gkp_cutoff = 30
gkp_alpha = jnp.sqrt(jnp.pi / 2)
gkp_beta = jnp.sqrt(jnp.pi / 2) * 1j

# gkp states
gkp_0, gkp_1 = gkp_states(n_fock, gkp_alpha, gkp_beta, gkp_delta, gkp_cutoff)
gkp_0, gkp_1 = jqt.Qarray.create(gkp_0), jqt.Qarray.create(gkp_1)

# vacuum
vac = jqt.basis(n_fock, 0)

rho0 = gkp_0
rhof = gkp_1

H_opt, min_time = quantum_brachistochrone_hamiltonian(
        rho0.data, rhof.data, energy_bound=1.0
)

H_opt = jqt.Qarray.create(H_opt)

# solve time evolution
n_time = 200
x_bound = 5.0
y_bound = 5.0
grid_points = 200
tlist = jnp.linspace(0.0, min_time, n_time)
states = jqt.sesolve(H_opt, rho0, tlist)

plot_wigner_snapshots(states.data, rho0.data, rhof.data, n_snapshots=6, x_bound=x_bound,
                      y_bound=y_bound, grid_points=grid_points)
# plt.tight_layout()
plt.show()
