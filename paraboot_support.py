'''
Paraboot support

Extracted code from library modules to reduce dependencies for bootstrap
calculations.
'''
#------------------------------------------------------------------------------
# from simulate_cells
#------------------------------------------------------------------------------

import os
import numpy as np
from motorlab.kinematics import get_dir, get_speed

class ParameterSet(object):
    pass

class Cell(object):
    def __init__(self, B):
        assert(type(B) == dict)
        assert(B.has_key('k')) # mean rate
        self.B = B

class TunedCell(Cell):
    def __init__(self, B=dict(), model='k'):
        Cell.__init__(self, B)
        self.model = model

class KDPSCell(TunedCell):
    def __init__(self, B={}, model='kd'):
        '''
        B : dict
            'k' : float
                baseline
            'D' : ndarray, shape (3,)
                B_D coefficient (PD * MD)
        model : str
        '''
        assert(B.has_key('k'))
        assert(B.has_key('D'))
        assert(B.has_key('P'))
        assert(B.has_key('s'))
        TunedCell.__init__(self, B=B, model=model)

    def encode(self, bnd):
        '''
        Return spike counts corresponding to given direction and encoding model.

        Parameters
        ----------
        bnd :BinnedData
          uses bnd.pos, bnd.bin_edges
        '''
        pos = bnd.pos
        time = bnd.bin_edges
        ntask, nrep, nedge, ndim = pos.shape
        model = self.model

        assert(ndim == 3)
        nbin = nedge - 1
        lshape = ntask, nrep, nbin
        rate = np.zeros(lshape, dtype=float)
        assert(model == 'kd') # only model currently implemented 
        
        rate += self.B['k']                           # baseline
        dr    = get_dir(pos)
        rate += np.dot(dr, self.B['D'])               # direction
        rate += np.dot(pos[...,:-1,:], self.B['P'])   # position
        speed = get_speed(pos, time, tax=2, spax=-1)
        rate += self.B['s'] * speed
        return rate

# -----------------------------------------------------------------------------
# from tuning_project
# -----------------------------------------------------------------------------
ds_frank = 'frank-osmd'
ds_tupac = 'tupac-uid'

intermediate_dir = '/data'
unit_file = intermediate_dir + '/unit_names_list.txt'

code_dir = os.path.split(os.path.realpath(__file__))[0]
log_dir  = code_dir + '/log'
job_dir  = code_dir + '/job'   

par_std = ParameterSet()
par_std.unit_gam_file_pat = intermediate_dir + '/unit_gam_results' + \
    '/gam_%s_%s.npz' # % (dsname, unit_name.lower())
parameters = { 'std' : par_std }
nrep = 20 # int(1e3)
nrep_per_batch = 10
nbat = nrep / nrep_per_batch
