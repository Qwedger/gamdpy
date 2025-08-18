""" 2D Active OU particles  """
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import gamdpy as gp
import numpy as np


temperature = 0.01
density = 0.5
DT=0.25
DA=0.5
mu=1.0
tau=0.75
dt=0.001

# Setup configuration  
configuration = gp.Configuration(D=2)
configuration.make_lattice(unit_cell=gp.unit_cells.HEXAGONAL, cells=[32, 20], rho=density)


# Setup pair potential.
pairfunc = gp.apply_shifted_potential_cutoff(gp.make_IPL_n(3))
Gamma, cut = 1.0, 2.5
pairpot = gp.PairPotential(pairfunc, params=[Gamma, cut], max_num_nbs=1000)
interactions = [pairpot, ]

# Setup integrator
integrator = gp.integrators.ActiveOUP(DT=DT, DA=DA, mu=mu, tau=tau, dt=dt, seed=2025)
runtime_actions = [
    gp.RestartSaver(),
    gp.TrajectorySaver(),
    gp.MomentumReset(steps_between_reset=100),
    gp.ScalarSaver(16, {'W' :True, 'K' :True}),
]

sim = gp.Simulation(configuration, interactions, integrator, runtime_actions,
                    num_timeblocks=16,
                    steps_per_timeblock=512,
                    storage='memory')



print('Production:', end='\t')
for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
print(sim.summary())


dynamics = gp.tools.calc_dynamics(sim.output, first_block=2,qvalues=1.0)



# # Plot particle positions
positions = sim.configuration['r']
X, Y = configuration.simbox.get_lengths()
fig, ax = plt.subplots(figsize=(5,5))
ax.set_title("Particle Positions")
ax.set_aspect('equal')
for x, y in positions:
    c = Circle((x, y), radius=0.1, facecolor='white', edgecolor='black')
    ax.add_patch(c)
ax.set_xlim(-X/2, X/2)
ax.set_ylim(-Y/2, Y/2)
ax.set_xlabel('x')
ax.set_ylabel('y')
if __name__ == "__main__":
    plt.show()

# plot msd
plt.figure(figsize=(8, 9))
plt.plot(dynamics['times'], dynamics['msd'])
plt.xlabel('t')
plt.ylabel('msd')
plt.title('msd 2D AOUP dynamics')
plt.grid(True)
plt.tight_layout()     
plt.show()



