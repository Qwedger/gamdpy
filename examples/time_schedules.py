""" Example of possible uses of time scheduling classes, for a flexible TrajectorySaver.

Based on minimal.py
"""

import gamdpy as gp

# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = gp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup runtime actions, i.e., actions performed during simulation of timeblocks
runtime_actions = [gp.ScalarSaver(),
                   gp.RestartSaver(),
                   gp.MomentumReset(100)]


############################## TIME SCHEDULING ##############################

# Create an instance of TimeScheduler, with keywords appropriate to the shedule.
# The schedule will be used in each time block of the simulation.

# log2 schedule (also default option, if no option is given)
scheduler_log2 = gp.Logarithmic2()

# Logarithmic schedule with real base.
# Smaller bases give denser trajectory samplings
scheduler_log = gp.Logarithmic(base=1.5)

# Linear schedule with number of steps between saves
scheduler_lin1 = gp.Linear(steps_between_output=100)

# Linear schedule with number of saves
scheduler_lin1 = gp.Linear(npoints=10)

# Geometric schedule: logarithmic spacing with base chosen to get `npoints` saves
scheduler_geom = gp.Geometric(npoints=10)

# Pass the chosen scheduler instance to TrajectorySaver and add it to runtime_actions
trajectory_saver = gp.TrajectorySaver(scheduler=scheduler_log2)
runtime_actions.append(trajectory_saver)

"""
Of course everything can be done in one line

        ..code-block:: python
                runtime_actions.append(gp.TrajectorySaver(scheduler=gp.Logarithmic(base=1.5)))
"""

#############################################################################

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
