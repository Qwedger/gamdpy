"""
"""

import rumdpy as rp
import pandas as pd
import numpy as np
import pandas as pd
import pytest
from numba import config
import time
import sys



def test_NVU():
    # Setup configuration: FCC Lattice
    configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=9, rho=1.200)
    configuration.randomize_velocities(T=1.6)
    configuration.ptype[::5] = 1
    
    
    # Setup pair potential: Single component 12-6 Lennard-Jones
    pairfunc = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
    sig = [[1.00, 0.80],
           [0.80, 0.88]]
    eps = [[1.00, 1.50],
           [1.50, 0.50]]
    cut = np.array(sig)*2.5
    pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

    
    # Setup of NVT integrator and simulation, in order to find the average value
    # of the potential energy to be used by the NVU integrator.
    NVT_integrator = rp.integrators.NVT(temperature = 0.7, tau = 0.2, dt = 0.004)
    NVT_sim = rp.Simulation(configuration, pairpot, NVT_integrator, 
                            scalar_output=2, steps_between_momentum_reset = 100, 
                            num_timeblocks = 8, steps_per_timeblock = 1024,
                            storage = 'memory')
    #Running the NVT simulation
    for block in NVT_sim.timeblocks():
        pass 


    
    #Finding the average potential energy (= U_0) of the run.
    columns = ['U']
    data = np.array(rp.extract_scalars(NVT_sim.output, columns, first_block=4))
    df = pd.DataFrame(data.T, columns=columns)
    U_0 = np.mean(df['U'])/configuration.N
    
    
    #Setting up the NVU integrator and simulation. Note, that dt = dl.
    NVU_integrator = rp.integrators.NVU(U_0 = U_0, dt = 0.1)
    NVU_sim = rp.Simulation(configuration, pairpot, NVU_integrator, 
                            scalar_output=2, steps_between_momentum_reset=100, 
                            num_timeblocks = 8, steps_per_timeblock = 1024,
                            storage = 'memory')

    #Running the NVU simulation
    for block in NVU_sim.timeblocks():
        pass 

    
    #Calculating the configurational temperature
    columns = ['U', 'lapU', 'Fsq']
    data = np.array(rp.extract_scalars(NVU_sim.output, columns, first_block=4))
    df = pd.DataFrame(data.T, columns=columns)
    df['Tconf'] = df['Fsq']/df['lapU']
    Tconf = np.mean(df['Tconf'],axis=0)
    assert 0.68 < Tconf < 0.72, print("Tconf should be around 0.7, but is",
                                      f"{Tconf}. For this test, assertionError",
                                      "arises if Tconf is not in the interval",
                                      "[0.68; 0.72]. Try the test again if Tconf",
                                      "is close to the interval")
    assert np.allclose(np.mean(df['U'])/configuration.N,U_0), print("For this test,",
                                                    "assertionError arises if",
                                                    "the average potential",
                                                    "energy <U> =", 
                                                    f"{np.mean(df['U'])}",
                                                    "is not close enough to the",
                                                    f"set energy: U_0 = {U_0}.\n",
                                                    "The closeness is defined by",
                                                    "np.allclose(<U>,U_0).")
    
    configuration = rp.make_configuration_fcc(nx=4, ny=4, nz=4, rho=1.200)
    configuration.randomize_velocities(T=1.6)
    configuration.ptype[::5] = 1
    
    NVT_integrator = rp.integrators.NVT(temperature = 0.7, tau = 0.2, dt = 0.004)
    NVT_sim = rp.Simulation(configuration, pairpot, NVT_integrator, 
                            scalar_output=2, steps_between_momentum_reset = 100, 
                            num_timeblocks = 8, steps_per_timeblock = 8*1024,
                            storage = 'memory')
    #Running the NVT simulation
    for block in NVT_sim.timeblocks():
        pass 


    
    #Finding the average potential energy (= U_0) of the run.
    columns = ['U']
    data = np.array(rp.extract_scalars(NVT_sim.output, columns, first_block=4))
    df = pd.DataFrame(data.T, columns=columns)
    U_0 = np.mean(df['U'])/configuration.N
    

    #Setting up the NVU integrator and simulation. Note, that dt = dl.
    NVU_integrator = rp.integrators.NVU(U_0 = U_0, dt = 0.1)
    NVU_sim = rp.Simulation(configuration, pairpot, NVU_integrator,
                            scalar_output=2, steps_between_momentum_reset=100,
                            num_timeblocks = 8, steps_per_timeblock = 8*1024,
                            storage = 'memory')

    #Running the NVU simulation
    for block in NVU_sim.timeblocks():
        pass


    #Calculating the configurational temperature
    columns = ['U', 'lapU', 'Fsq']
    data = np.array(rp.extract_scalars(NVU_sim.output, columns, first_block=4))
    df = pd.DataFrame(data.T, columns=columns)
    df['Tconf'] = df['Fsq']/df['lapU']
    Tconf = np.mean(df['Tconf'],axis=0)
    assert 0.68 < Tconf < 0.72, print("Tconf should be around 0.7, but is",
                                      f"{Tconf}. For this test, assertionError",
                                      "arises if Tconf is not in the interval",
                                      "[0.68; 0.72].Try the test again if Tconf",
                                      "is close to the interval")
    assert np.allclose(np.mean(df['U'])/configuration.N,U_0), print("For this test,",
                                                    "assertionError arises if",
                                                    "the average potential",
                                                    "energy <U> =",
                                                    f"{np.mean(df['U'])}",
                                                    "is not close enough to the",
                                                    f"set energy: U_0 = {U_0}.\n",
                                                    "The closeness is defined by",
                                                    "np.allclose(<U>,U_0).")




if __name__ == '__main__':
    config.CUDA_LOW_OCCUPANCY_WARNINGS = False
    test_NVU()
