""" Minimal example of a running an simulation using gamdpy

Simulation of a Lennard-Jones crystal in the NVT ensemble.

"""

import gamdpy as gp
import numpy as np
from numba import cuda

cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = False

# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[3, 3, 3], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7)
print(configuration.simbox.get_lengths())

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.4
pair_pot_nsq = gp.PairPotentialNsquared(pair_func, params=[sig, eps, cut])
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = gp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup runtime actions, i.e., actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(16),
                   gp.RestartSaver(),
                   gp.MomentumReset(100)]


runtime_actions = []
num_timeblocks=16 
steps_per_timeblock=4*1024

pb = 16
print(f'{pb=}')
print(f'{configuration.N/pb=:f}')

for tp in [1, 2, 4, 8, 12, 16, 20, 24, 28, 30, 32]:
    compute_plan = gp.get_default_compute_plan(configuration)
    compute_plan['tp'] = tp
    compute_plan['pb'] = pb
    #print(f'{compute_plan=}')

    # Setup Simulation. 
    sim = gp.Simulation(configuration, [pair_pot], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    compute_plan=compute_plan,
                    storage='LJ_T0.70.h5')

    # Run simulation
    for timeblock in sim.run_timeblocks():
        ...
        #print(sim.status(per_particle=True))
    #print(sim.summary())

    time_sim = np.sum(sim.timing_numba_blocks) / 1000
    tps_sim = sim.last_num_blocks * sim.steps_per_block / time_sim
    
    sim = gp.Simulation(configuration, [pair_pot_nsq], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    compute_plan=compute_plan,
                    storage='LJ_T0.70_Nsquared.h5')

    # Run simulation
    for timeblock in sim.run_timeblocks():
        ...
        #print(sim.status(per_particle=True))
    #print(sim.summary())
    time_sim = np.sum(sim.timing_numba_blocks) / 1000
    tps_sim_nsq = sim.last_num_blocks * sim.steps_per_block / time_sim
 
    print(f'{tp=} {tps_sim=:f} {tps_sim_nsq=:f}')

# Print current status of configuration
print(configuration)

print('\nAnalyse the saved simulation with scripts found in "examples"')
print('(visualize requires that ovito is installed):')
print('   python3 analyze_structure.py LJ_T0.70')
print('   python3 analyze_dynamics.py LJ_T0.70')
print('   python3 analyze_thermodynamics.py LJ_T0.70')
print('   python3 visualize.py LJ_T0.70.h5')

