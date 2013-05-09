'''
Calculate distribution of test statistic

.. math:: \Delta \tau_K^j = \tau_{K,\mathrm{const}^j} - \tau_{K,\mathrm{vary}^j}

using parametric bootstrapping. Each \tau_{K,\mathrm{const}} is calculated from:

.. math::

    \hat{B} = \mathrm{gam}(\mathrm{model}, Y_\mathrm{train}, X_\mathrm{train})
    \hat{Y} = \hat{B} X_\mathrm{test}
    \tau_{K,\mathrm{model}} = \tau_K(\hat{Y}, Y_\mathrm{test})

where Y for the test statistic distribution calculation is derived as random
Poisson variates from the fit of the real data with null (constant) model.
'''

import os, sys, logging

paper_dir = '/data'
proj_dir = paper_dir + '/code'
if not proj_dir in sys.path:
    sys.path.append(proj_dir)
fig_sim_dir = paper_dir + '/figures/fig_ctrl_sim'
if not fig_sim_dir in sys.path:
    sys.path.append(fig_sim_dir)
from paraboot_support import KDPSCell

import numpy as np
from motorlab.tuning.gam_plus import GAMFitManyModels
import motorlab.tuning.gam_plus as gam

from motorlab.binned_data import load_binned_data, \
    make_bnd_with_count_from_rate

import paraboot_support as tp
pars = tp.parameters['std']
null_model = 'kdps'
other_model = 'kdpsX'

def _convert_Bgam_to_Bsim(Bgam):
    Bsim = {}
    for k, v in Bgam.iteritems():
        if v.size == 1:
            Bsim[k] = v
        elif v.size > 1:
            Bsim[k.upper()] = v
        else:
            raise ValueError('should not happen')
    return Bsim

def _get_paraboot_dataset(real, Bnull):
    Bsim = _convert_Bgam_to_Bsim(Bnull)
    cell = KDPSCell(Bsim)
    rate = cell.encode(real)
    # add extra dimension for units
    rate.shape = list(rate.shape[:2]) + [1] + [rate.shape[2]]
    paraboot_dataset = make_bnd_with_count_from_rate(rate, real, \
        ignore_prev=True, rename_units=True)
    return paraboot_dataset

def _get_gam_fit(data, models=[null_model, other_model]):
    dsname    = data.dsname # leech dsname - not standard part of binned_data
    assert data.unit_names.size == 1
    unit_name = data.unit_names[0]
    lag       = data.lags[0]
    align     = data.align   
    metadata = [dsname, unit_name, lag, align]
    gam_fit  = gam.gam_predict_cv2(data, models, metadata, nfold=10, \
        family='poisson', verbosity=0)
    return gam_fit

def _get_delta_tauK(dataset):
    gam_fit    = _get_gam_fit(dataset)
    tauK, P = gam.calc_kendall_tau(gam_fit)
    return tauK[null_model], tauK[other_model], tauK[null_model] - tauK[other_model]

def _get_save_file(dsname, unit_name, batch):
    base = tp.intermediate_dir
    unit_dir = base + '/dtk_%s' % unit_name.lower()
    if not os.path.isdir(unit_dir):
        os.mkdir(unit_dir)
    file_name = unit_dir + '/dtk_%s_%s_%03d.txt' % \
            (dsname, unit_name.lower(), batch)
    return open(file_name, 'a')

def _write_tauK_null_distribution(dsname, real, batch, Bnull, nrep=1):
    '''
    Write to file `nrep` times parametric bootstrapped :math:\Delta \tau_K
    values
    '''
    assert len(real.unit_names) == 1
    unit_name = real.unit_names[0]

    #f = _get_save_file(dsname, unit_name, batch)
    #print "Going to write %s" % f.name
    
    with _get_save_file(dsname, unit_name, batch) as f:
        for j in xrange(nrep):
            pb_dataset_j = _get_paraboot_dataset(real, Bnull)
            logging.debug(np.nansum(pb_dataset_j.count))
            pb_dataset_j.dsname = dsname # not a standard part of binned_data
            static_tauj, dynamic_tauj, diff_tauj = \
                _get_delta_tauK(pb_dataset_j)
            f.write('iter %d: static %0.8f vary %0.8f diff %0.8f\n' % \
                (j, static_tauj, dynamic_tauj, diff_tauj))
            f.flush()
    f.close()

def _load_real_dataset(dsname, unit_name):
    '''
    Notes
    -----
    frank has bnd data saved in
        `intermediate_dir + '/bnd_frank-osmd_all_std.npz`
    tupac has bnd data saved in individual files in
        `intermediate_dir/bnd_tupac-uid_files/bnd_tupac-uid_V_000??_`
            `CenterOut_Unit???_?_100ms.npz`
    '''
    bnd_dir = tp.intermediate_dir + '/bnd_%s_files' % (dsname)
    bnd_name = bnd_dir + '/bnd_%s_%s_100ms.npz' % (dsname, unit_name)
    real_dataset = load_binned_data(bnd_name)
    return real_dataset

def _load_gam_fit(dsname, unit_name):
    gam_file_name = pars.unit_gam_file_pat % (dsname, unit_name.lower())
    assert os.path.isfile(gam_file_name), "missing %s" % (gam_file_name)

    # this probably won't work since these are old format files
    # could do with another function .from_old_file to do conversion
    gam_fit = GAMFitManyModels.from_npz_file(gam_file_name)
    return gam_fit

def _get_Bnull(gam_fit):
    '''
    Derive approximate B coefficients by averaging over cross-validation runs
    '''
    fit = gam_fit.fits[null_model]
    Bnull = gam.unpack_coefficients(fit.coef, fit.coef_names)
    ncv = Bnull['k'].shape[0]
    for k, v in Bnull.iteritems():
        assert v.shape[-1] == ncv, "Last dimension should be %d long" % (ncv)
        # average along last dimension
        Bnull[k] = np.mean(v, axis=-1)
    return Bnull

def write_delta_tauK_distrib_one_cell(dsname, unit_name, batch, nrep=10):
    '''
    Calculate, using parametric bootstrapping, the P value that the test
    statistic tau_K is a member of the distribution generated when the cell
    is generated by a constant PD model.

    Parameters
    ----------
    cell : string
      string code for cell to run
    '''
    logging.info('running batch %d' % batch)
    cell    = dsname, unit_name
    real    = _load_real_dataset(*cell)
    gam_fit = _load_gam_fit(*cell)
    Bnull   = _get_Bnull(gam_fit)
    
    _write_tauK_null_distribution(dsname, real, batch, Bnull, nrep=nrep)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dsname")
    parser.add_argument("unitname")
    parser.add_argument("batch",    nargs="?", default=1, type=int)
    parser.add_argument("--nrep",   help="number of repetitions to perform", \
        type=int, default=int(1))
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    print args
    
    if args.trace:
        from amcmorl_py_tools.run_tools import switch_on_call_tracing
        print "Tracing"
        switch_on_call_tracing('')

    logging.basicConfig(filename=tp.log_dir + '/dtk_%s_%s_%d.log' % \
        (args.dsname, args.unitname, args.batch), level=logging.DEBUG)
    write_delta_tauK_distrib_one_cell(args.dsname, args.unitname, args.batch, \
        nrep=args.nrep)
