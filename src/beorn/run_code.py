'''
Created on 1 March 2023
@author: Sambit Giri
File containing all the functions to run simulations.
'''

import numpy as np 
import os
from .run_simple_faster import *
from .cosmo import *

def run_coeval_EOR_CD(redshift, param):
    """
	Simulates a coeval cube with BEORN.
	
	Parameters
	----------
	redshift: float 
		The redshift
	param: Bunch 
		The parameter file created using the beorn.par().
	Returns
	-------
	grid_outputs: dict
        The simulated cubes.
	"""
    if param.sim.dens_fields is None or param.sim.halo_catalogs is None:
        if param.sim.dens_field_type == '21cmfast':
            matter_field = simulate_matter_21cmfast(redshift, param, IC=None)
            param.sim.dens_fields   = [matter_field['dens']]
            param.sim.halo_catalogs = [matter_field['halo_list']]
        else:
            print('BEORN is not coupled to {}.'.format(param.sim.dens_field_type)) 
            print('Please run this code manual and pass the density and halo catalogs.')
    initialise_run(param)
    profiles = model_profiles(param, method='simple')
    grid_outputs = paint_profiles(param, profiles=profiles)
    return grid_outputs

def initialise_run(param):
    if param.sim.data_dir is None:
        print('The outputs will not be saved during the runtime.')
    create_save_folders(folder_names=None, data_dir=param.sim.data_dir)
    return None

def model_profiles(param, method='simple'):
    model_name = param.sim.model_name
    try: pkl_name = param.sim.data_dir+'/profiles/{}_zi_{:.3f}.pkl'.format(model_name,param.solver.z_ini)
    except: pkl_name = param.sim.data_dir
    
    try:
        profiles = load_f(file = pkl_name)
        print('Profiles loaded from {}'.format(pkl_name))
    except:
        if method.lower()=='simple':
            profiles = compute_profiles(param, pkl_name=pkl_name)
        else:
            print('This method is not implemented.')
    return profiles

def model_post_HI_profiles(param, method='Model1'):
    model_name = param.sim.model_name
    try: pkl_name = param.sim.data_dir+'/profiles/{}_zi_{:.3f}.pkl'.format(model_name,param.solver.z_ini)
    except: pkl_name = param.sim.data_dir
    
    try:
        profiles = load_f(file = pkl_name)
        print('Profiles loaded from {}'.format(pkl_name))
    except:
        if method.lower()=='simple':
            profiles = compute_profiles(param, pkl_name=pkl_name)
        else:
            print('This method is not implemented.')
    return profiles

def paint_profiles(param, temp=True, lyal=True, ion=True, dTb=True, profiles=None):
    """
    Parameters
    ----------
    param : dictionnary containing all the input parameters

    Returns
    -------
    Does not return anything. Loop over all snapshots in param.sim.halo_catalogs and calls paint_profile_single_snap. Uses joblib for parallelisation.
    """

    start_time = datetime.datetime.now()
    LBox = param.sim.Lbox    # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    # catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name

    if param.sim.halo_catalogs is None:
        print('You should specify param.sim.halo_catalogs. It can be folder containing the halo_catalogs or a list.')
        print('The list can be of filenames or halo information in dictionaries.')

    print('Painting profiles on a grid with', nGrid, 'pixels per dim. Box size is', LBox ,'cMpc/h.')

    try: catalog_files = os.listdir(param.sim.halo_catalogs)
    except: catalog_files = param.sim.halo_catalogs
    dens_files = [param.sim.dens_fields] if type(param.sim.dens_fields)==str else param.sim.dens_fields

    if param.sim.n_jobs in [0,1,None,False]:
        grid_outputs = {}
        for ii, filename in enumerate(catalog_files):
            grid_output, grid_z = _paint_profile_single_snap(catalog_files[ii], param, temp=temp, lyal=lyal, ion=ion, dTb=dTb, 
                                                             delta_b=dens_files[ii],  profiles=profiles)
            grid_outputs['{:.3f}'.format(grid_z)] = grid_output
    else:
        print('n_jobs =', param.sim.n_jobs)
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=param.sim.n_jobs)(
                delayed(_paint_profile_single_snap)(catalog_files[ii], param, temp=temp, lyal=lyal, ion=ion, dTb=dTb, 
                                        delta_b=dens_files[ii],  profiles=profiles) for ii, filename in enumerate(catalog_files))
        grid_outputs = {'{:.3f}'.format(grid_z): grid_output for grid_output,grid_z in out}
 
    end_time = datetime.datetime.now()
    print('Finished painting the maps.')
    print('Runtime of painting the grids:', end_time-start_time)
    if param.sim.store_grids: 
        print('Grids are stored in', param.sim.data_dir+'/grid_output/')

    return grid_outputs


def _paint_profile_single_snap(filename_or_dict, param, temp=True, lyal=True, ion=True, dTb=True, delta_b=None, profiles=None):
    store_grids = param.sim.store_grids
    param.sim.store_grids = False 

    try:
        catalog = param.sim.halo_catalogs + filename_or_dict
        halo_catalog = load_f(catalog)
    except:
        halo_catalog = filename_or_dict

    model_name = param.sim.model_name
    LBox = param.sim.Lbox    # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    
    try:
        grid_filename = param.sim.data_dir+'/grid_output/Grid_{}_z_{:.3f}_model_{}.pkl'.format(nGrid, halo_catalog['z'], model_name)
        check_if_present = False if store_grids=='replace' else os.path.exists(grid_filename)
    except:
        grid_filename = None 
        check_if_present = False

    if check_if_present:
        print('The snapshot is already present in {}'.format(grid_filename))
        grid_output = load_f(grid_filename)
    else:
        print('----- Painting for redshift = {:.3f} -------'.format(halo_catalog['z']))
        if type(delta_b)==str: delta_b = load_delta_b(param, delta_b)
        grid_output = paint_profile_single_snap(halo_catalog, param, temp=temp, lyal=lyal, ion=ion, dTb=dTb, profiles=profiles, delta_b=delta_b)
        grid_output['dens'] = delta_b
        if store_grids: save_f(file=grid_filename, obj=grid_output)
        print('----- Redshift = {:.3f} is done -------'.format(halo_catalog['z']))

    return grid_output, halo_catalog['z']


def initialise_21cmfast(param):
    import py21cmfast as p21c

    data_dir = param.sim.data_dir #'./data/'
    user_params  = p21c.UserParams({"HII_DIM": param.sim.Ncell, "DIM": param.sim.Ncell*3, 
                                    "BOX_LEN": param.sim.Lbox/param.cosmo.h, 
                                    "USE_INTERPOLATION_TABLES": True,
                                    #"FIXED_IC": True,
                                   "N_THREADS": param.sim.n_jobs,
                                    })
    cosmo_params = p21c.CosmoParams(SIGMA_8=param.cosmo.s8,
                                    hlittle=param.cosmo.h,
                                    OMm=param.cosmo.Om,
                                    OMb=param.cosmo.Ob,
                                    POWER_INDEX=param.cosmo.ns,
                                    )
    Tvir = M_to_Tvir(param.source.M_min/param.cosmo.h, param.solver.z_end, param)
    astro_params = p21c.AstroParams({"ION_Tvir_MIN":20.0})
    random_seed  = param.sim.random_seed

    with p21c.global_params.use(INITIAL_REDSHIFT=300, CLUMPING_FACTOR=2.0):
        IC = p21c.initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                random_seed=random_seed,
                write=data_dir,
                direc=data_dir,
                )    
    import tools21cm as t2c
    pslin, klin = t2c.power_spectrum_1d(IC.hires_density, kbins=20, box_dims=user_params.BOX_LEN)
    return IC, pslin, klin

def simulate_matter_21cmfast(redshift, param, IC=None):
    import py21cmfast as p21c

    start_time = datetime.datetime.now()
    print('Simulating matter evolution with 21cmFast...')

    data_dir = param.sim.data_dir #'./data/'
    user_params  = p21c.UserParams({"HII_DIM": param.sim.Ncell, "DIM": param.sim.Ncell*3, 
                                    "BOX_LEN": param.sim.Lbox/param.cosmo.h, 
                                    "USE_INTERPOLATION_TABLES": True,
                                    #"FIXED_IC": True,
                                   "N_THREADS": param.sim.n_jobs,
                                    })
    cosmo_params = p21c.CosmoParams(SIGMA_8=param.cosmo.s8,
                                    hlittle=param.cosmo.h,
                                    OMm=param.cosmo.Om,
                                    OMb=param.cosmo.Ob,
                                    POWER_INDEX=param.cosmo.ns,
                                    )
    Tvir = M_to_Tvir(param.source.M_min/param.cosmo.h, param.solver.z_end, param)
    astro_params = p21c.AstroParams({"ION_Tvir_MIN":20.0})
    random_seed  = param.sim.random_seed

    with p21c.global_params.use(INITIAL_REDSHIFT=300, CLUMPING_FACTOR=2.0):
        if IC is None:
            IC, pslin, klin= initialise_21cmfast(param)

        perturbed_field = p21c.perturb_field(
                    redshift = redshift,
                    init_boxes = IC,
                    # user_params=user_params,
                    # cosmo_params=cosmo_params,
                    # astro_params=astro_params,
                    # random_seed=random_seed,
                    write=data_dir,
                    direc=data_dir,
                )
        halo_list = p21c.perturb_halo_list(
                    redshift = redshift,
                    init_boxes = IC,
                    # user_params=user_params,
                    # cosmo_params=cosmo_params,
                    # astro_params=astro_params,
                    # random_seed=random_seed,
                    write=data_dir,
                    direc=data_dir,
                )
    
    output = {'dens': perturbed_field.density,
              'halo_list': {'X': halo_list.halo_coords[:,0]*param.sim.Lbox/user_params.HII_DIM,
                            'Y': halo_list.halo_coords[:,1]*param.sim.Lbox/user_params.HII_DIM,
                            'Z': halo_list.halo_coords[:,2]*param.sim.Lbox/user_params.HII_DIM,
                            'M': halo_list.halo_masses*param.cosmo.h,
                            'z': redshift,
                            } 
              }
    end_time = datetime.datetime.now()
    print('...done | Runtime =', end_time-start_time)
    return output

