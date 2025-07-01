""" Minimal example of a running an simulation using gamdpy

Simulation of a Lennard-Jones crystal in the NVT ensemble.

"""

import numpy as np
import gamdpy as gp
import matplotlib.pyplot as plt

# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7)

# Setup pair potential: Single component 12-6 Lennard-Jones

labels = [['KABLJ_AA', 'KABLJ_AB'],['KABLJ_AB', 'KABLJ_BB']]

pair_pot = gp.TabulatedPairPotential('tab_LJ_LAMMPS.dat', params=labels, max_num_nbs=1000)
table = np.loadtxt('temp_SF.dat')

pfunction_LJ = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)

r_values = np.arange(0.6, 2.5, 0.001)
v_analytic = np.zeros_like(r_values)
v_table = np.zeros_like(r_values)
params_analytic = (1.0, 1.0, 2.5)
for idx in range(len(r_values)):
    v, s, v2 = pfunction_LJ(r_values[idx], params_analytic)
    v_analytic[idx] = s
    v, s, v2 = pair_pot.evaluate_potential_function(r_values[idx], (0,0))
    v_table[idx] = s


# Setup integrator: NVT
integrator = gp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup runtime actions, i.e., actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.RestartSaver(),
                   gp.MomentumReset(100)]

# Setup Simulation. 
sim = gp.Simulation(configuration, [pair_pot], integrator, runtime_actions,
                    num_timeblocks=32, steps_per_timeblock=1*1024,
                    storage='LJ_T0.70.h5')

# Run simulation
for timeblock in sim.run_timeblocks():
        print(sim.status(per_particle=True))
print(sim.summary())

# Print current status of configuration
print(configuration)

# Do analysis from the commandline (like computing MSD) with something like,
#    python -m gamdpy.tools.calc_dynamics -f 4 -o msd.pdf LJ_T*.h5
# or with a Python script. See examples.
