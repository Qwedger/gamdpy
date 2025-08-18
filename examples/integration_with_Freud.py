'''On the run integration with Freud analysis library'''
#Here is shown how is possible to integrate Freud library tools with Gamdpy to do calculations based on voronoi neighbors 

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import freud as fd
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

#setup simulation
nblocks = 16
sim = gp.Simulation(configuration, interactions, integrator, runtime_actions,
                    num_timeblocks=nblocks,
                    steps_per_timeblock=512,
                    storage='memory')

#integration with freud

#converting gamdpy box in a freud box
X, Y = configuration.simbox.get_lengths()   
fd_box = fd.box.Box(Lx=X, Ly=Y, Lz=0, xy=0, xz=0, yz=0, is2D=True)

#zero column to convert (x,y) points in (x,y,0) points (2D freud format)
N = sim.configuration.N     
zero_col = np.zeros((N, 1),dtype=np.float32)

#freud objects to compute
voro = fd.locality.Voronoi()
hex_order = fd.order.Hexatic(k=6)
bins = 100
r_max = (fd_box.Lx)/2.01
orient_corr = fd.density.CorrelationFunction(bins=100, r_max= r_max)
avg_orient_corr = np.zeros((bins,), dtype=np.complex64)

print('Production:', end='\t')
for block in sim.run_timeblocks():
    #(x,y)->(x,y,0) (freud works with 3N configurations)
    fd_points = np.hstack((configuration['r'], zero_col))         
    #neighbor query computed by freud "compute" functions
    nq = fd.locality.AABBQuery(box=fd_box, points=fd_points)    #Freud recommend do it once and pass the result as a parameter
    #compute voronoi neighbors
    voro_results = voro.compute(system=nq)
    #compute local hexatic order parameter
    hex_order_results = hex_order.compute(system=nq, neighbors=voro_results.nlist)
    #compute orientational correlation function 
    orient_corr_results = orient_corr.compute(system=nq, values=hex_order_results.particle_order)
    avg_orient_corr += orient_corr_results.correlation

    print(sim.status(per_particle=True))
print(sim.summary())

avg_orient_corr /= nblocks

#plot orientational correlation
r = np.linspace(0.0, r_max, bins) 
plt.figure(1)
plt.plot(r, avg_orient_corr)
plt.title("orientational correlation")
plt.ylabel("g_6(r)")
plt.xlabel("r")
plt.show()
