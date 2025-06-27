import pytest
import numpy as np
from gamdpy.runtime_actions.time_scheduler import *

global stepmax
global ntimeblocks

stepmax = 10000
ntimeblocks = 10

def test_base():
    base_scheduler = BaseScheduler()
    with pytest.raises(NotImplementedError):
        base_scheduler.setup(stepmax=stepmax, ntimeblocks=ntimeblocks)

def test_log2():
    time_scheduler = Logarithmic2()
    time_scheduler.setup(stepmax, ntimeblocks)
    try_steps = time_scheduler.steps
    try_indexes = time_scheduler.indexes
    steps, indexes = [0], [0]
    i = 0
    while True:
        if 2**i>stepmax:
            break
        steps.append(2**i)
        indexes.append(i+1)
        i += 1
    assert (try_steps==steps).all()
    assert (try_indexes==indexes).all()
    assert len(try_indexes) == time_scheduler.nsaves

def test_log():
    base = 1.1
    time_scheduler = Logarithmic(base=base)
    time_scheduler.setup(stepmax, ntimeblocks)
    try_steps = time_scheduler.steps
    try_indexes = time_scheduler.indexes
    steps = [0]
    i = 0
    while True:
        if base**i>stepmax:
            break
        if int(base**i) not in steps:
            steps.append(int(base**i))
        i += 1
    indexes = np.arange(len(steps))
    assert (try_steps==steps).all()
    assert (try_indexes==indexes).all()
    assert len(try_indexes) == time_scheduler.nsaves

def test_lin_deltastep():
    deltastep = 51
    time_scheduler = Linear(steps_between_output=deltastep)
    time_scheduler.setup(stepmax, ntimeblocks)
    try_steps = time_scheduler.steps
    try_indexes = time_scheduler.indexes
    steps = np.arange(0, stepmax+1, deltastep)
    indexes = np.arange(len(steps))
    assert (try_steps==steps).all()
    assert (try_indexes==indexes).all()
    assert len(try_indexes) == time_scheduler.nsaves

def test_lin_npoints():
    npoints = 13
    deltastep = stepmax // npoints
    time_scheduler = Linear(npoints=npoints)
    time_scheduler.setup(stepmax, ntimeblocks)
    try_steps = time_scheduler.steps
    try_indexes = time_scheduler.indexes
    steps = np.arange(0, stepmax+1, deltastep)
    indexes = np.arange(len(steps))
    assert (try_steps==steps).all()
    assert (try_indexes==indexes).all()
    assert len(try_indexes) == time_scheduler.nsaves

def test_geom():
    npoints = 13
    time_scheduler = Geometric(npoints=npoints)
    time_scheduler.setup(stepmax, ntimeblocks)
    try_steps = time_scheduler.steps
    try_indexes = time_scheduler.indexes
    xx = stepmax**(1.0/npoints)
    steps = np.array([0] + [xx**(i+1) - 1 for i in range(1, npoints)], dtype='int')
    indexes = np.arange(len(steps))
    assert (try_steps==steps).all()
    assert (try_indexes==indexes).all()
    assert len(try_indexes) == time_scheduler.nsaves
    assert npoints == time_scheduler.nsaves

if __name__ == '__main__':
    test_base()
    test_log2()
    test_log()
    test_lin_deltastep()
    test_lin_npoints()
    test_geom()
