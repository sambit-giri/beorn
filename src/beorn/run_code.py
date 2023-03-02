'''
Created on 1 March 2023
@author: Sambit Giri
File containing all the functions to run simulations.
'''

import numpy as np 
import os
from .run_simple_faster import *

def initialise_run(param):
    if param.sim.save_dir is None:
        print('The outputs will not be saved during the runtime.')
    create_save_folders(folder_names=None, save_dir=param.sim.save_dir)
    return None

def model_profiles(param, method='simple'):
    model_name = param.sim.model_name
    pkl_name = param.sim.save_dir+'/profiles/' + model_name + '_zi_{:.3f}.pkl'.format(param.solver.z)
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
        print('Grids are stored in', param.sim.save_dir+'/grid_output/')

    return grid_outputs


def _paint_profile_single_snap(filename, param, temp=True, lyal=True, ion=True, dTb=True, delta_b=None, profiles=None):
    store_grids = param.sim.store_grids
    param.sim.store_grids = False 
    catalog = param.sim.halo_catalogs + filename
    halo_catalog = load_f(catalog)

    model_name = param.sim.model_name
    LBox = param.sim.Lbox    # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    grid_filename = param.sim.save_dir+'/grid_output/Grid_{}_z_{:.3f}_model_{}.pkl'.format(nGrid, halo_catalog['z'], model_name)
    check_if_present = False if store_grids=='replace' else os.path.exists(grid_filename)

    if check_if_present:
        print('The snapshot is already present in {}'.format(grid_filename))
        grid_output = load_f(grid_filename)
    else:
        print('----- Painting for redshift = {:.3f} -------'.format(halo_catalog['z']))
        if type(delta_b)==str: delta_b = load_delta_b(param, delta_b)
        grid_output = paint_profile_single_snap(halo_catalog, param, temp=temp, lyal=lyal, ion=ion, dTb=dTb, profiles=profiles, delta_b=delta_b)
        if store_grids: save_f(file=grid_filename, obj=grid_output)
        print('----- Redshift = {:.3f} is done -------'.format(halo_catalog['z']))

    return grid_output, halo_catalog['z']
