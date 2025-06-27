import numpy as np
import math


class BaseScheduler():
    """
    Class used to:
        - define steps to save configuration at;
        - get functions for the numba kernel to check whether to save.

    An instance of TimeScheduler must be passed to TrajectorySaver, either explicitly or implicitly
    (i.e. passing appropriate keywords to TrajectorySaver, that will create an instance of TimeScheduler).

    Example:

    ..code-block:: python
        import gamdpy as gp
        scheduler = gp.TimeScheduler(schedule='log', base=1.5)
        runtime_actions = [gp.TrajectorySaver(schedule=scheduler),]

    alternatively

    ..code-block:: python
        import gamdpy as gp
        runtime_actions = [gp.TrajectorySaver(schedule='log', base=1.5),]

    See below for indications about kwargs required by different schedules. 
    Default is 'log2', which does not require any arguments.
    The list `runtime_actions` must then be passed to a Simulation instance.
    """

    def __init__(self):
        self.known_schedules = ['log2', 'log', 'lin', 'geom']
        self.stepmax = 10000

    def setup(self, stepmax, ntimeblocks):
        # This is necessary aside from __init__ because in TrajectorySaver
        # `steps_per_timeblock` is initialised only in a `setup` method

        # `stepmax` is by construction the same as `steps_per_timeblock` in TrajectorySaver
        # it makes sense to keep it as an attribute since it may be needed in the future for other schedules
        self.stepmax = stepmax
        self.ntimeblocks = ntimeblocks # currently not used
        # with open('debug.txt', 'w') as f:
        #     f.write(str(self.stepmax))

        self.stepcheck_func = self._get_stepcheck()
        self.steps, self.indexes = self._compute_steps()
        # self.stepsall = self._compute_stepsall()

        # TODO: dreprecate this
        # if self.schedule=='geom':
        #     # the 'geom' schedule must return each save index only once; this must be after _compute_steps()
        #     assert self.nsaves==self.npoints, 'Too many points, schedule distorsion; try fewer points'

        '''
        # no specific kwarg is required
        if self.schedule=='log2':
            self.stepcheck_func = self._get_stepcheck_log2()

        # `base` kwarg is accepted but not required (default is Euler number)
        elif self.schedule=='log':
            self.base = self._kwargs.get('base', np.exp(1.0))
            self.stepcheck_func = self._get_stepcheck_log()

        # `steps_between_output` kwarg is required
        elif self.schedule=='lin':
            deltastep = self._kwargs.get('steps_between_output', None)
            npoints = self._kwargs.get('npoints', None)
            if deltastep is not None:
                self.deltastep = deltastep
                self.stepcheck_func = self._get_stepcheck_lin()
            elif npoints is not None:
                raise NotImplementedError("Passing 'npoints' to 'lin' schedule should be possible")
                #self.npoints = npoints
            else:
                raise TypeError("'steps_between_output' or 'npoints' is required for schedule 'lin'")

        # `npoints` kwarg is required
        elif self.schedule=='geom':
            npoints = self._kwargs.get('npoints', None)
            if npoints is not None:
                self.npoints = npoints
                self.stepcheck_func = self._get_stepcheck_geom()
            else:
                raise TypeError("'npoints' is required for schedule 'geom'")

        # TODO: using custom steps would require passing arrays to stepcheck functions, which is a potential issue
        # elif self.schedule=='custom':
        #     pass
        '''

    def _get_stepcheck(self):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement this method")

    def _compute_steps(self):
        steps = []
        indexes = []
        # we want to include the last step IFF it raises a True flag
        for step in range(self.stepmax+1):
            flag, idx = self.stepcheck_func(step)
            if flag: 
                steps.append(step)
                indexes.append(idx)
        return np.array(steps), np.array(indexes)

    @property
    def nsaves(self):
        return len(self.steps)

class Logarithmic2(BaseScheduler):

    def __init__(self):
        super().__init__()

    def _get_stepcheck(self):
        def stepcheck(step):
            flag = False
            idx = -1 # this is for python calls of the function
            if step == 0:
                flag = True
                idx = 0
            else:
                b = np.int32(math.log2(np.float32(step)))
                c = 2 ** b
                if step == c:
                    flag = True
                    idx = b + 1
            return flag, idx
        return stepcheck

class Logarithmic(BaseScheduler):
    
    # def __init__(self, base=np.exp(1.0)):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def _get_stepcheck(self):
        base = self.base
        def stepcheck(step):
            # determine flag
            flag = False
            if step==0 or step==1:
                flag = True
            else:
                virtual_index = int(np.log(step+1)/np.log(base))
                virtual_step = int(base**virtual_index)
                if virtual_step==step:
                    flag = True
            if not flag:
                return False, -1
            idx = 0
            # find the save index by counting previous true flags
            for i in range(1, step+1):
                if i==0 or i==1:
                    idx += 1
                else:
                    virtual_index = int(np.log(i+1)/np.log(base))
                    virtual_step = int(base**virtual_index)
                    if virtual_step==i:
                        idx += 1
            return True, idx
        return stepcheck

class Linear(BaseScheduler):

    def __init__(self, steps_between_output=None, npoints=None):
        super().__init__()
        self.steps_between_output = steps_between_output
        self.npoints = npoints

    def _get_stepcheck(self):
        # this must go here because the needed super() attributes are defined in setup(), not __init__()
        if self.steps_between_output is not None and self.npoints is None:
            self.deltastep = self.steps_between_output
        elif self.npoints is not None:
            # this needs testing
            self.deltastep = self.stepmax // self.npoints
        deltastep = self.deltastep
        def stepcheck(step):
            if step%deltastep==0:
                return True, step//deltastep
            return False, -1
        return stepcheck

class Geometric(BaseScheduler):

    def __init__(self, npoints):
        super().__init__()
        self.npoints = npoints

    def _get_stepcheck(self):
        stepmax = self.stepmax
        npoints = self.npoints
        def stepcheck(step):
            if step==0:
                return True, 0
            xx = stepmax**(1.0/npoints)
            for idx in range(1, npoints):
                c = xx**(idx+1)-1
                if step==int(c):
                    return True, idx
            return False, -1
        return stepcheck
