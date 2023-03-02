"""
In this script we define functions that can be called to :
1. run the RT solver and compute the evolution of the T, x_HI profiles, and store them
2. paint the profiles on a grid.
"""
# import beorn as rad
from .simple_model_faster import *
from scipy.interpolate import splrep, splev, interp1d
import numpy as np
import pickle
import datetime
from .constants import cm_per_Mpc, M_sun, m_H, rhoc0, Tcmb0
from .cosmo import T_adiab, D
import os
from .profiles_on_grid import profile_to_3Dkernel, Spreading_Excess_Fast, put_profiles_group, stacked_lyal_kernel, stacked_T_kernel
from .couplings import x_coll,rho_alpha, S_alpha
from .global_qty import simple_xHII_approx
from os.path import exists
from .python_functions import load_f, save_f

def create_save_folders(folder_names=None, data_dir='./'):
    if folder_names is None:
        folder_names = ['profiles', 'grid_output', 'physics']
    if data_dir is not None:
        for name in folder_names:
            if not os.path.isdir(data_dir+'/'+name):
                os.mkdir(data_dir+'/'+name)
                print('A folder created for {} created in {}'.format(name,data_dir))
    return None


def compute_profiles(param, pkl_name=None):
    """
    This function computes the Temperature, Lyman-alpha, and ionisation fraction profiles that will then be used to produce the maps.

    Parameters
    ----------
    param : dictionnary containing all the input parameters

    Returns
    -------
    Does not return anything. However, it solve the RT equation for a range of halo masses, following their evolution from cosmic dawn to the end of reionization. It stores the profile in a directory called ./profiles.
    """
    start_time = datetime.datetime.now()

    print('Computing Temperature (Tk), Lyman-α and ionisation fraction (xHII) profiles...')
    create_save_folders(folder_names=None, data_dir=param.sim.data_dir)

    model_name = param.sim.model_name
    try: pkl_name = param.sim.data_dir+'/profiles/' + model_name + '_zi{}.pkl'.format(param.solver.z)
    except: pass
    grid_model = simple_solver_faster(param)
    grid_model.solve(param)
    if param.sim.data_dir is not None:
        pickle.dump(file=open(pkl_name, 'wb'), obj=grid_model)
        print('\n The profiles and stored in {}'.format(param.sim.data_dir+'/profiles/'))
    end_time = datetime.datetime.now()
    print('Runtime of computing the profiles:', end_time - start_time)
    return grid_model

def paint_profile_single_snap(filename_or_dict,param,temp=True,lyal=True,ion=True,dTb=True,delta_b=None,profiles=None):
    """
    Paint the Tk, xHII and Lyman alpha profiles on a grid for a single halo catalog named filename.

    Parameters
    ----------
    param : dictionnary containing all the input parameters
    filename : the name of the halo catalog, contained in param.sim.halo_catalogs.
    temp, lyal, ion, dTb : which map to paint.

    Returns
    -------
    Does not return anything. Paints and stores the grids int the directory grid_outputs.
    """

    catalog_dir = param.sim.halo_catalogs
    start_time = datetime.datetime.now()
    z_start = param.solver.z_ini
    model_name = param.sim.model_name
    M_Bin = np.logspace(np.log10(param.sim.M_i_min), np.log10(param.sim.M_i_max), param.sim.binn, base=10)

    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    try:
        catalog = catalog_dir + filename_or_dict
        halo_catalog = load_f(catalog)
    except:
        halo_catalog = filename_or_dict

    H_Masses, H_X, H_Y, H_Z = halo_catalog['M'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z']
    z = halo_catalog['z']

    ### To later add up the adiabatic temperature
    T_adiab_z = T_adiab(z, param)
    if delta_b is None: delta_b = load_delta_b(param,filename_or_dict) # rho/rhomean-1

    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    factor = 27 * Ob * h0 ** 2 / 0.023 * np.sqrt(0.15 / Om / h0 ** 2 / 10)
    coef = rhoc0 * h0 ** 2 * Ob * (1 + z) ** 3 * M_sun / cm_per_Mpc ** 3 / m_H

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = load_f(file = param.sim.data_dir+'/profiles/' + model_name + '_zi{}.pkl'.format(z_start)) if profiles is None else profiles
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]
    Indexing = np.argmin( np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))), axis=1)
    print('There are {} halos at z = {:.3f}'.format(H_Masses.size, z))

    if H_Masses.size == 0:
        print('There aint no sources')
        Grid_xHII = np.array([0])
        Grid_Temp = T_adiab_z * (1+delta_b)**(2/3)
        Grid_xal = np.array([0])
        Grid_xcoll = x_coll(z=z, Tk=Grid_Temp, xHI=(1 - Grid_xHII), rho_b=(delta_b + 1) * coef)
        Grid_dTb = factor * np.sqrt(1 + z) * (1 - Tcmb0 * (1 + z) / Grid_Temp) * (1 - Grid_xHII) * (delta_b + 1) * Grid_xcoll / (1 + Grid_xcoll)

    else:
        Ionized_vol = simple_xHII_approx(param,halo_catalog,profiles=profiles)[1]
        print('Quick calculation from the profiles predicts xHII = ',round(Ionized_vol,4))
        if Ionized_vol > 1:
            Grid_xHII = np.array([1])
            Grid_Temp = np.array([1])
            Grid_dTb = np.array([0])
            Grid_xal = np.array([0])
            print('universe is fully inoinzed. Return [1] for the xHII, T and [0] for dTb.')

        else:
            Pos_Bubles = np.vstack((H_X, H_Y, H_Z)).T  # Halo positions.
            Pos_Bubbles_Grid = np.array([Pos_Bubles / LBox * nGrid]).astype(int)[0]
            Pos_Bubbles_Grid[np.where(Pos_Bubbles_Grid == nGrid)] = nGrid - 1  # you don't want Pos_Bubbles_Grid==nGrid
            Grid_xHII_i = np.zeros((nGrid, nGrid, nGrid))
            Grid_Temp = np.zeros((nGrid, nGrid, nGrid))
            Grid_xal = np.zeros((nGrid, nGrid, nGrid))
            for i in range(len(M_Bin)):
                indices = np.where(Indexing == i)[0]  ## indices in H_Masses of halos that have an initial mass at z=z_start between M_Bin[i-1] and M_Bin[i]
                Mh_ = grid_model.Mh_history[ind_z,i]


                if len(indices) > 0 and Mh_ > param.source.M_min:
                    radial_grid = grid_model.r_grid_cell/(1+zgrid) #pMpc/h
                    x_HII_profile = np.zeros((len(radial_grid)))
                    x_HII_profile[np.where(radial_grid < grid_model.R_bubble[ind_z,i] / (1 + zgrid))] = 1
                    Temp_profile = grid_model.rho_heat[ind_z,:,i]

                    r_lyal = np.logspace(-5, 2, 1000, base=10)     ##    physical distance for lyal profile. Never goes further away than 100 pMpc/h (checked)
                    rho_alpha_ = rho_alpha(r_lyal, Mh_, zgrid, param)[0]
                    x_alpha_prof = 1.81e11 * (rho_alpha_) / (1 + zgrid)    # * S_alpha(zgrid, T_extrap, 1 - xHII_extrap)

                    ### This is the position of halos in base "nGrid". We use this to speed up the code.
                    ### We count with np.unique the number of halos in each cell. Then we do not have to loop over halo positions in profiles_on_grid/put_profiles_group
                    base_nGrid_position = Pos_Bubbles_Grid[indices][:, 0] + nGrid * Pos_Bubbles_Grid[indices][:,1] + nGrid ** 2 * Pos_Bubbles_Grid[indices][:, 2]
                    unique_base_nGrid_poz, nbr_of_halos = np.unique(base_nGrid_position, return_counts=True)

                    ZZ_indice = unique_base_nGrid_poz // (nGrid ** 2)
                    YY_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2) // nGrid
                    XX_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2 - YY_indice * nGrid)

                    ## Every halos in mass bin i are assumed to have mass M_bin[i].
                    if ion:
                        profile_xHII = interp1d(radial_grid * (1 + z), x_HII_profile, bounds_error=False, fill_value=(1, 0))
                        kernel_xHII = profile_to_3Dkernel(profile_xHII, nGrid, LBox)
                        if not np.any(kernel_xHII > 0):
                            ### if the bubble volume is smaller than the grid size,we paint central cell with ion fraction value
                            # kernel_xHII[int(nGrid / 2), int(nGrid / 2), int(nGrid / 2)] = np.trapz(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / nGrid / (1 + z)) ** 3
                            Grid_xHII_i[XX_indice,YY_indice,ZZ_indice] += np.trapz(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / nGrid / (1 + z)) ** 3 * nbr_of_halos

                        else :
                            renorm = np.trapz(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / (1 + z)) ** 3 / np.mean(kernel_xHII)
                            #extra_ion = put_profiles_group(Pos_Bubbles_Grid[indices], kernel_xHII * 1e-7 / np.sum(kernel_xHII)) * np.sum(kernel_xHII) / 1e-7 * renorm
                            extra_ion = put_profiles_group(np.array((XX_indice,YY_indice,ZZ_indice)),nbr_of_halos, kernel_xHII * 1e-7 / np.sum(kernel_xHII)) * np.sum(kernel_xHII) / 1e-7 * renorm
                           #bubble_volume = np.trapz(4 * np.pi * radial_grid ** 2 * x_HII_profile, radial_grid)
                           #print('bubble volume is ', len(indices) * bubble_volume,'pMpc, grid volume is', np.sum(extra_ion)* (LBox /nGrid/ (1 + z)) ** 3 )
                            Grid_xHII_i += extra_ion
                    if lyal:
                        ### We use this stacked_kernel functions to impose periodic boundary conditions when the lyal or T profiles extend outside the box size. Very important for Lyman-a.
                        kernel_xal = stacked_lyal_kernel(r_lyal * (1 + z), x_alpha_prof, LBox, nGrid, nGrid_min=32)
                        renorm = np.trapz(x_alpha_prof * 4 * np.pi * r_lyal ** 2, r_lyal) / (LBox / (1 + z)) ** 3 / np.mean( kernel_xal)
                        if np.any(kernel_xal > 0):
                            #Grid_xal += put_profiles_group(Pos_Bubbles_Grid[indices], kernel_xal * 1e-7 / np.sum(kernel_xal)) * renorm * np.sum( kernel_xal) / 1e-7  # we do this trick to avoid error from the fft when np.sum(kernel) is too close to zero.
                            Grid_xal += put_profiles_group(np.array((XX_indice,YY_indice,ZZ_indice)),nbr_of_halos,kernel_xal * 1e-7 / np.sum(kernel_xal)) * renorm * np.sum( kernel_xal) / 1e-7  # we do this trick to avoid error from the fft when np.sum(kernel) is too close to zero.
                    if temp:
                        kernel_T = stacked_T_kernel(radial_grid * (1 + z), Temp_profile, LBox, nGrid, nGrid_min=4)
                        renorm = np.trapz(Temp_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / (1 + z)) ** 3 / np.mean(kernel_T)
                        if np.any(kernel_T > 0):
                            #Grid_Temp += put_profiles_group(Pos_Bubbles_Grid[indices],  kernel_T * 1e-7 / np.sum(kernel_T)) * np.sum(kernel_T) / 1e-7 * renorm
                            Grid_Temp += put_profiles_group(np.array((XX_indice,YY_indice,ZZ_indice)),nbr_of_halos,  kernel_T * 1e-7 / np.sum(kernel_T)) * np.sum(kernel_T) / 1e-7 * renorm


                end_time = datetime.datetime.now()
                # print(len(indices), 'halos in mass bin ', i, '. It took : ', end_time - start_time,'to paint the profiles.')
                print('Mass bin {}/{} | n_haloes = {}, painting runtime = {}'.format(i+1,M_Bin.size,len(indices),end_time - start_time))
            # print(M_Bin.shape)

            Grid_Storage = np.copy(Grid_xHII_i)

            if np.sum(Grid_Storage) < nGrid ** 3 and ion:
                Grid_xHII = Spreading_Excess_Fast(param,Grid_Storage, pix_thresh=param.sim.thresh_pixel)
            else:
                Grid_xHII = np.array([1])

            time_spreadring_end = datetime.datetime.now()
            # print('It took:', time_spreadring_end - end_time, 'to spread the excess photons')
            print('Runtime to spread the excess photons:', time_spreadring_end - end_time)

            if np.all(Grid_xHII == 0):
                Grid_xHII = np.array([0])
            if np.all(Grid_xHII == 1):
                print('universe is fully inoinzed. Return [1] for Grid_xHII.')
                Grid_xHII = np.array([1])

            Grid_Temp += T_adiab_z * (1+delta_b)**(2/3)

            if dTb:
                Grid_xcoll = x_coll(z=z, Tk=Grid_Temp, xHI=(1-Grid_xHII), rho_b=(delta_b + 1) * coef)
                Grid_xtot = Grid_xcoll + Grid_xal/4/np.pi
                Grid_dTb = factor * np.sqrt(1 + z) * (1 - Tcmb0 * (1 + z) / Grid_Temp) * (1-Grid_xHII) * (delta_b + 1) * Grid_xtot / (1 + Grid_xtot)

    output_dict = {}
    if temp:
        output_dict['temp'] = Grid_Temp
        if param.sim.store_grids: save_f(file=param.sim.data_dir+'/grid_output/T_Grid'   + str(nGrid)  + model_name + '_snap' + filename_or_dict[4:-5], obj=Grid_Temp)
    if ion:
        output_dict['ion'] = Grid_xHII
        if param.sim.store_grids: save_f(file=param.sim.data_dir+'/grid_output/xHII_Grid'+ str(nGrid)  + model_name + '_snap' + filename_or_dict[4:-5], obj=Grid_xHII)
    if lyal:
        # We divide by 4pi to go to sr**-1 units
        output_dict['lyal'] = Grid_xal * S_alpha(z, Grid_Temp, 1 - Grid_xHII)/4/np.pi
        if param.sim.store_grids: save_f(file=param.sim.data_dir+'/grid_output/xal_Grid' + str(nGrid)  + model_name + '_snap' + filename_or_dict[4:-5], obj=Grid_xal * S_alpha(z, Grid_Temp, 1 - Grid_xHII)/4/np.pi)
    if dTb:
        output_dict['dTb'] = Grid_dTb
        if param.sim.store_grids: save_f(file=param.sim.data_dir+'/grid_output/dTb_Grid' + str(nGrid)  + model_name + '_snap' + filename_or_dict[4:-5], obj=Grid_dTb)

    return output_dict



def paint_boxes(param,temp =True,lyal=True,ion=True,dTb=True):
    """
    Parameters
    ----------
    param : dictionnary containing all the input parameters

    Returns
    -------
    Does not return anything. Loop over all snapshots in param.sim.halo_catalogs and calls paint_profile_single_snap. Uses joblib for parallelisation.
    """

    start_time = datetime.datetime.now()
    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name

    if catalog_dir is None :
        print('You should specify param.sim.halo_catalogs. Should be a file containing the rockstar halo catalogs.')

    print('Painting profiles on a grid with', nGrid,'pixels per dim. Box size is',LBox ,'cMpc/h.')


    if param.sim.mpi4py == 'yes':
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    elif param.sim.mpi4py == 'no':
        rank = 0
        size = 1
    else :
        print('param.sim.mpi4py should be yes or no')

    for ii, filename in enumerate(os.listdir(catalog_dir)):
        if rank == ii % size:
            print('Core nbr',rank,'is taking care of snap',filename[4:-5])
            if exists('./grid_output/xHII_Grid' + str(nGrid) + model_name + '_snap' + filename[4:-5]):
                print('xHII map for snapshot ',filename[4:-5],'already painted. Skiping.')
            else:
                print('----- Painting for snapshot nbr :', filename[4:-5], '-------')
                output_dict = paint_profile_single_snap(filename,param,temp=temp, lyal=lyal, ion=ion, dTb=dTb)
                print('----- Snapshot nbr :', filename[4:-5], ' is done -------')
 
    end_time = datetime.datetime.now()
    print('Finished painting the maps. They are stored in ./grid_output. It took in total: ', end_time-start_time,'to paint the grids.')
    print('  ')


def grid_dTb(param,ion = None):
    """
    Creates a grid of xcoll and dTb. Needs to read in Tk grid, xHII grid and density field on grid.
    """
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name
    nGrid = param.sim.Ncell
    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    factor = 27 * Ob * h0 ** 2 / 0.023 * np.sqrt(0.15 / Om / h0 ** 2 / 10)  # factor used in dTb calculation


    if param.sim.mpi4py == 'yes':
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    elif param.sim.mpi4py == 'no':
        rank = 0
        size = 1
    else :
        print('param.sim.mpi4py should be yes or no')


    for ii, filename in enumerate(os.listdir(catalog_dir)):
        if rank == ii % size:
            zz_ = load_f(catalog_dir+filename)['z']
            Grid_Temp           = pickle.load(file=open('./grid_output/T_Grid'    + str(nGrid) + model_name + '_snap' + filename[4:-5], 'rb'))
            if ion == 'exc_set':
                Grid_xHII = pickle.load(file=open('./grid_output/xHII_exc_set_' + str(nGrid) +'_' + model_name + '_snap' + filename[4:-5],'rb'))
            elif ion == 'Sem_Num':
                Grid_xHII = pickle.load(file=open('./grid_output/xHII_Sem_Num_' + str(nGrid) + '_' + model_name + '_snap' + filename[4:-5],'rb'))
            else :
                Grid_xHII           = pickle.load(file=open('./grid_output/xHII_Grid' + str(nGrid)  + model_name + '_snap' + filename[4:-5], 'rb'))
            #Grid_xtot_ov        = pickle.load(file=open('./grid_output/xtot_ov_Grid' + str(nGrid) + model_name + '_snap' + filename[4:-5], 'rb'))
            Grid_xal             = pickle.load(file=open('./grid_output/xal_Grid' + str(nGrid)+ model_name + '_snap' + filename[4:-5], 'rb'))

            dens_field = param.sim.dens_fields
            if dens_field is not None:
                delta_b = load_delta_b(param, filename)
            else :
                delta_b = 0 #rho/rhomean -1

            T_cmb_z = Tcmb0 * (1 + zz_)
            Grid_xHI = 1-Grid_xHII  ### neutral fraction

            #Grid_Sal = S_alpha(zz_, Grid_Temp, 1 - Grid_xHII)
            Grid_xal = Grid_xal #* Grid_Sal
            coef =  rhoc0 * h0 ** 2 *  Ob * (1 + zz_) ** 3 * M_sun / cm_per_Mpc ** 3 / m_H
            Grid_xcoll = x_coll(z=zz_, Tk=Grid_Temp, xHI=Grid_xHI, rho_b = (delta_b+1)*coef)
            Grid_Tspin = ((1 / T_cmb_z + (Grid_xcoll + Grid_xal) / Grid_Temp) / (1 + Grid_xcoll + Grid_xal)) ** -1

            #Grid_Tspin = ((1 / T_cmb_z + (Grid_xcoll+Grid_xal) / Grid_Temp) / (1 + Grid_xcoll+Grid_xal)) ** -1
            Grid_xtot = Grid_xcoll+Grid_xal

            Grid_dTb = factor * np.sqrt(1+zz_) * (1-T_cmb_z/Grid_Tspin) * Grid_xHI * (delta_b+1)
            #Grid_dTb = factor * np.sqrt(1 + zz_) * (1 - T_cmb_z / Grid_Tspin) * Grid_xHI * (delta_b+1)  #* Grid_xtot / (1 + Grid_xtot)
            #pickle.dump(file=open('./grid_output/Tspin_Grid' + str(nGrid)+ model_name + '_snap' + filename[4:-5], 'wb'),obj=Grid_Tspin)
            pickle.dump(file=open('./grid_output/dTb_Grid'+str(nGrid)+model_name+'_snap'+filename[4:-5],'wb'),obj = Grid_dTb)
            pickle.dump(file=open('./grid_output/xcoll_Grid'+str(nGrid)+model_name+'_snap'+filename[4:-5],'wb'),obj = Grid_xcoll)



def compute_GS(param, string='', RSD=False, ion=None):
    """
    Reads in the grids and compute the global quantities averaged.
    If RSD is True, will add RSD calculation
    If lyal_from_sfrd is True, we will compute xalpha from the sfrd (see eq 19. and 23. from HM paper 2011.12308) and then correct dTb to match this xalpha.
    If ion = exc_set, will read in the xHII maps produce from the excursion set formalism.
    """
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name
    nGrid = param.sim.Ncell
    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob,param.cosmo.h
    factor = 27 * (1 / 10) ** 0.5 * (Ob * h0 ** 2 / 0.023) * (Om * h0 ** 2 / 0.15) ** (-0.5)
    Tadiab = []
    z_ = []
    Tk = []
    dTb  =[]
    x_HII = []
    x_al = []
    #beta_a = []
    #beta_T = []
    #beta_r = []
    dTb_RSD = []

    for ii, filename in enumerate(os.listdir(catalog_dir)):
        zz_ = load_f(catalog_dir+filename)['z']

        if ion == 'exc_set' :
            Grid_xHII = pickle.load(file=open('./grid_output/xHII_exc_set_' + str(nGrid) +'_' + model_name + '_snap' + filename[4:-5], 'rb'))
        elif ion == 'Sem_Num':
            Grid_xHII = pickle.load(file=open('./grid_output/xHII_Sem_Num_' + str(nGrid) + '_' + model_name + '_snap' + filename[4:-5],'rb'))

        else :
            Grid_xHII = pickle.load( file=open('./grid_output/xHII_Grid' + str(nGrid)  + model_name + '_snap' + filename[4:-5],  'rb'))
        Grid_Temp = pickle.load( file=open('./grid_output/T_Grid' + str(nGrid) + model_name + '_snap' + filename[4:-5], 'rb'))
        #Grid_xtot_ov       = pickle.load(file=open('./grid_output/xtot_ov_Grid' + str(nGrid)  + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_dTb            = pickle.load(file=open('./grid_output/dTb_Grid'  + str(nGrid)  + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_xal            = pickle.load(file=open('./grid_output/xal_Grid'  + str(nGrid)+ model_name + '_snap' + filename[4:-5], 'rb'))

        xal_  = np.mean(Grid_xal)
        Tcmb = (1 + zz_) * Tcmb0

        z_.append(zz_)
        Tk.append(np.mean(Grid_Temp))
        #Tk_neutral.append(np.mean(Grid_Temp[np.where(Grid_xHII < param.sim.thresh_xHII)]))

        #T_spin.append(np.mean(Grid_Tspin[np.where(Grid_xHII < param.sim.thresh_xHII)]))
        x_HII.append(np.mean(Grid_xHII))
        x_al.append(xal_)
        #x_coll.append(xcol_)
        dTb.append(np.mean(Grid_dTb))
        #beta_a.append(xal_ / (xcol_ + xal_) / (1 + xcol_ + xal_))
        #beta_T.append(Tcmb /(Tk[ii]-Tcmb))
        #beta_r.append(-x_HII[ii] / (1 - x_HII[ii]))

        Tadiab.append(Tcmb0 * (1+zz_)**2/(1+param.cosmo.z_decoupl) )

        if RSD:
            dTb_RSD.append(np.mean(Grid_dTb /  RSD_field(param, load_delta_b(param,filename), zz_)))
        else :
            dTb_RSD.append(0)


    z_, Tk, x_HII, x_al, Tadiab, dTb, dTb_RSD = np.array(z_),np.array(Tk),np.array(x_HII),np.array(x_al),np.array(Tadiab), np.array(dTb),np.array(dTb_RSD)

    matrice = np.array([z_, Tk, x_HII, x_al, Tadiab, dTb, dTb_RSD])
    z_, Tk, x_HII, x_al, Tadiab, dTb, dTb_RSD = matrice[:, matrice[0].argsort()]  ## sort according to z_

    Tgam = (1 + z_) * Tcmb0
    T_spin = ((1 / Tgam +  x_al  / Tk) / (1 +x_al )) ** -1

    #### Here we compute Jalpha using HM formula. It is more precise since it accounts for halos at high redshift that mergerd and are not present at low redshift.
    #dTb_GS = factor * np.sqrt(1 + z_) * (1 - Tcmb0 * (1 + z_) / Tk) * (1-x_HII) * (x_coll + x_al) / (1 + x_coll + x_al)### dTb formula similar to coda HM code
    #dTb_GS_Tkneutral = factor * np.sqrt(1 + z_) * (1 - Tcmb0 * (1 + z_) / Tk_neutral) * (1-x_HII) * (x_coll + x_al) / (1 + x_coll + x_al)
    #beta_a = (x_al / (x_coll + x_al) / (1 + x_coll + x_al))
    #xtot = x_al + x_coll
    #dTb_GS_Tkneutral = dTb_GS_Tkneutral * xtot / (1 + xtot) / ((x_coll + x_al) / (1 + x_coll + x_al))

    Dict = {'Tk':Tk,'x_HII':x_HII,'x_al':x_al,'dTb':dTb,'dTb_RSD':dTb_RSD,'Tadiab':Tadiab,'z':z_,'T_spin':T_spin}

    pickle.dump(file=open('./physics/GS_'+string + str(nGrid) + model_name+'.pkl', 'wb'),obj=Dict)


def compute_PS(param,Tspin = False,RSD = False,ion = None,cross_corr = False):
    """
    Parameters
    ----------
    param : dictionnary containing all the input parameters
    Tspin : if True, will compute the spin temperature Power Spectrum as well as cross correlation with matter field and xHII field.
    cross_corr : Choose to compute the cross correlations. If False, it speeds up.
    Returns
    -------
    Computes the power spectra of the desired quantities

    """
    import tools21cm as t2c
    start_time = datetime.datetime.now()
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name
    nGrid = param.sim.Ncell
    #Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    Lbox = param.sim.Lbox  #Mpc/h
    if isinstance(param.sim.kbin,int):
        kbins = np.logspace(np.log10(param.sim.kmin), np.log10(param.sim.kmax), param.sim.kbin, base=10) #h/Mpc
    elif isinstance(param.sim.kbin,str):
        kbins = np.loadtxt(param.sim.kbin)
    else :
        print('param.sim.kbin should be either a path to a text files containing kbins edges values or it should be an int.')

    z_arr = []
    for filename in os.listdir(catalog_dir): #count the number of snapshots
        zz_ = load_f(catalog_dir+filename)['z']
        z_arr.append(zz_)

    z_arr = np.sort(z_arr)
    nbr_snap = len(z_arr)

    PS_xHII = np.zeros((nbr_snap,len(kbins)-1))
    PS_T   = np.zeros((nbr_snap,len(kbins)-1))
    PS_xal = np.zeros((nbr_snap,len(kbins)-1))
    PS_rho = np.zeros((nbr_snap,len(kbins)-1))
    PS_dTb = np.zeros((nbr_snap,len(kbins)-1))
    if RSD :
        PS_dTb_RSD = np.zeros((nbr_snap,len(kbins)-1))

    if cross_corr:
        PS_T_lyal = np.zeros((nbr_snap,len(kbins)-1))
        PS_T_xHII  = np.zeros((nbr_snap,len(kbins)-1))
        PS_rho_xHII = np.zeros((nbr_snap,len(kbins)-1))
        PS_rho_xal = np.zeros((nbr_snap,len(kbins)-1))
        PS_rho_T   = np.zeros((nbr_snap,len(kbins)-1))
        PS_lyal_xHII = np.zeros((nbr_snap,len(kbins)-1))


    if Tspin :
        PS_Ts      = np.zeros((nbr_snap,len(kbins)-1))
        PS_rho_Ts  = np.zeros((nbr_snap,len(kbins)-1))
        PS_Ts_xHII = np.zeros((nbr_snap,len(kbins)-1))
        PS_T_Ts = np.zeros((nbr_snap,len(kbins)-1))


    for filename in os.listdir(catalog_dir):
        zz_ = load_f(catalog_dir+filename)['z']
        Grid_Temp           = pickle.load(file=open('./grid_output/T_Grid'    + str(nGrid) + model_name + '_snap' + filename[4:-5], 'rb'))
        if ion == 'exc_set':
            Grid_xHII           = pickle.load(file=open('./grid_output/xHII_exc_set_' + str(nGrid) +'_' + model_name + '_snap' + filename[4:-5], 'rb'))
        elif ion == 'Sem_Num':
            Grid_xHII = pickle.load(file=open('./grid_output/xHII_Sem_Num_' + str(nGrid) + '_' + model_name + '_snap' + filename[4:-5],'rb'))
        else :
            Grid_xHII = pickle.load( file=open('./grid_output/xHII_Grid' + str(nGrid) + model_name + '_snap' + filename[4:-5],  'rb'))
        Grid_dTb  = pickle.load(file=open('./grid_output/dTb_Grid'  + str(nGrid) + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_xal  = pickle.load(file=open('./grid_output/xal_Grid'  + str(nGrid) + model_name + '_snap' + filename[4:-5], 'rb'))

        if Tspin:
            T_cmb_z = Tcmb0*(1+zz_)
            #Grid_xcoll = pickle.load(file=open('./grid_output/xcoll_Grid' + str(nGrid) + model_name + '_snap' + filename[4:-5], 'rb'))
            Grid_Tspin = ((1 / T_cmb_z + Grid_xal / Grid_Temp) / (1  + Grid_xal)) ** -1

        if Grid_Temp.size == 1: ## to avoid error when measuring power spectrum
            Grid_Temp = np.full((nGrid, nGrid, nGrid),1)
        if Grid_xHII.size == 1:
            Grid_xHII = np.full((nGrid, nGrid, nGrid),0) ## to avoid div by zero
        if Grid_dTb.size == 1:
            Grid_dTb = np.full((nGrid, nGrid, nGrid), 1)
        if Grid_xal.size == 1:
            Grid_xal = np.full((nGrid, nGrid, nGrid), 1)


        delta_XHII = Grid_xHII/ np.mean(Grid_xHII) - 1
        delta_T    = Grid_Temp/ np.mean(Grid_Temp) - 1
        delta_dTb  = Grid_dTb / np.mean(Grid_dTb)  - 1
        delta_x_al = Grid_xal / np.mean(Grid_xal)  - 1

        ii = np.where(z_arr == zz_)

        if Tspin :
            delta_Tspin = Grid_Tspin / np.mean(Grid_Tspin) - 1

        dens_field = param.sim.dens_fields
        if dens_field is not None:
            delta_rho = load_delta_b(param,filename)
            PS_rho[ii]      = t2c.power_spectrum.power_spectrum_1d(delta_rho, box_dims=Lbox , kbins=kbins)[0]
            if cross_corr:
                PS_rho_xHII[ii] = t2c.power_spectrum.cross_power_spectrum_1d(delta_XHII, delta_rho,box_dims=Lbox, kbins=kbins)[0]
                PS_rho_xal[ii] = t2c.power_spectrum.cross_power_spectrum_1d(delta_x_al, delta_rho, box_dims=Lbox, kbins=kbins)[0]
                PS_rho_T[ii]   = t2c.power_spectrum.cross_power_spectrum_1d(delta_T, delta_rho, box_dims=Lbox, kbins=kbins)[0]
            if Tspin :
                PS_rho_Ts[ii] = t2c.power_spectrum.cross_power_spectrum_1d(delta_Tspin, delta_rho, box_dims=Lbox, kbins=kbins)[0]
        else:
            delta_rho = 0,0  #rho/rhomean-1
            print('no density field provided.')


        if RSD:
            Grid_dTb_RSD = Grid_dTb / RSD_field(param, delta_rho, zz_)
            delta_Grid_dTb_RSD = Grid_dTb_RSD/np.mean(Grid_dTb_RSD)-1
        else :
            delta_Grid_dTb_RSD = 0

        z_arr[ii]  = zz_
        PS_xHII[ii], k_bins = t2c.power_spectrum.power_spectrum_1d(delta_XHII, box_dims=Lbox, kbins=kbins)
        PS_T[ii]   = t2c.power_spectrum.power_spectrum_1d(delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_xal[ii] = t2c.power_spectrum.power_spectrum_1d(delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_dTb[ii] = t2c.power_spectrum.power_spectrum_1d(delta_dTb, box_dims=Lbox, kbins=kbins)[0]
        if RSD:
            PS_dTb_RSD[ii] = t2c.power_spectrum.power_spectrum_1d(delta_Grid_dTb_RSD, box_dims=Lbox, kbins=kbins)[0]

        if cross_corr:
            PS_T_lyal[ii] = t2c.power_spectrum.cross_power_spectrum_1d(delta_T, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
            PS_T_xHII[ii] = t2c.power_spectrum.cross_power_spectrum_1d(delta_T, delta_XHII, box_dims=Lbox, kbins=kbins)[0]
            PS_lyal_xHII[ii]  = t2c.power_spectrum.cross_power_spectrum_1d(delta_x_al, delta_XHII, box_dims=Lbox, kbins=kbins)[0]

        if Tspin:
            PS_Ts[ii] = t2c.power_spectrum.power_spectrum_1d(delta_Tspin, box_dims=Lbox, kbins=kbins)[0]
            PS_Ts_xHII[ii] = t2c.power_spectrum.cross_power_spectrum_1d(delta_Tspin, delta_XHII, box_dims=Lbox, kbins=kbins)[0]
            PS_T_Ts[ii] = t2c.power_spectrum.cross_power_spectrum_1d(delta_Tspin, delta_T, box_dims=Lbox, kbins=kbins)[0]


    Dict = {'z':z_arr,'k':k_bins,'PS_xHII': PS_xHII, 'PS_T': PS_T, 'PS_xal': PS_xal, 'PS_dTb': PS_dTb,  'PS_rho': PS_rho}

    if RSD :
        Dict['PS_dTb_RSD'] = PS_dTb_RSD
    if cross_corr:
        Dict['PS_T_lyal'], Dict['PS_T_xHII'], Dict['PS_rho_xHII'], Dict['PS_rho_xal'], Dict['PS_rho_T'], Dict['PS_lyal_xHII'] = PS_T_lyal,PS_T_xHII,PS_rho_xHII,PS_rho_xal,PS_rho_T, PS_lyal_xHII
    if Tspin:
        Dict['PS_Ts'], Dict['PS_rho_Ts'], Dict['PS_xHII_Ts'],Dict['PS_T_Ts'] = PS_Ts, PS_rho_Ts, PS_Ts_xHII,PS_T_Ts
    end_time = datetime.datetime.now()

    print('Computing the power spectra took : ', start_time -end_time)
    pickle.dump(file=open('./physics/PS_' + str(nGrid) + model_name + '.pkl', 'wb'), obj=Dict)



def load_delta_b(param,filename):
    """
    Load the delta_b grid profiles.
    """
    LBox = param.sim.Lbox
    nGrid = param.sim.Ncell
    dens_field = param.sim.dens_fields

    if param.sim.dens_field_type == 'pkdgrav':
        if dens_field is not None :
            try: dens = np.fromfile(dens_field + filename[4:-5] + '.0', dtype=np.float32)
            except: dens = np.fromfile(filename, dtype=np.float32)
            pkd  = dens.reshape(nGrid, nGrid, nGrid)
            pkd  = pkd.T  ### take the transpose to match X_ion map coordinates
            V_total = LBox ** 3
            V_cell  = (LBox / nGrid) ** 3
            mass    = pkd * rhoc0 * V_total
            rho_m   = mass / V_cell
            delta_b = (rho_m) / np.mean(rho_m)-1
        else:
            delta_b = np.array([0])  # rho/rhomean-1 (usual delta here..)

    elif param.sim.dens_field_type == '21cmFAST':
        try: delta_b = load_f(dens_field + filename[4:-5] + '.0')
        except: delta_b = load_f(filename)
    else :
        print('param.sim.dens_field_type should be either 21cmFAST or pkdgrav.')
    return delta_b



def RSD_field(param,density_field,zz):
    """
    eq 4 from 411, 955–972 (Mesinger 2011, 21cmFAST..):  dvr/dr(k) = -kr**2/k**2 * dD/da(z)/D(z) * a * delta_nl(k) * da/dt
    And da/dt = H * a
    Take density field, go in Fourier space, transform it, go back to real space to get dvr/dr.
    Divide dTb to the output of this function to add RSD corrections.

    Parameters
    ----------
    density_field : delta_b, output of load_delta_b

    Returns
    ---------
    Returns a meshgrid containing values of -->(dv/dr/H+1) <--. Dimensionless.
    """

    import scipy
    Ncell = param.sim.Ncell
    Lbox  = param.sim.Lbox
    delta_k = scipy.fft.fftn(density_field)

    scale_factor = np.linspace(1 /40, 1 / 7, 100)
    growth_factor = np.zeros(len(scale_factor))
    for i in range(len(scale_factor)):
        growth_factor[i] = D(scale_factor[i], param)
    dD_da = np.gradient(growth_factor, scale_factor)

    kx_meshgrid = np.zeros((density_field.shape))
    ky_meshgrid = np.zeros((density_field.shape))
    kz_meshgrid = np.zeros((density_field.shape))

    kx_meshgrid[np.arange(0, Ncell, 1), :, :] = np.arange(0, Ncell, 1)[:, None, None] * 2 * np.pi / Lbox
    ky_meshgrid[:, np.arange(0, Ncell, 1), :] = np.arange(0, Ncell, 1)[None, :, None] * 2 * np.pi / Lbox
    kz_meshgrid[:, :, np.arange(0, Ncell, 1)] = np.arange(0, Ncell, 1)[None, None, :] * 2 * np.pi / Lbox

    k_sq = np.sqrt(kx_meshgrid ** 2 + ky_meshgrid ** 2 + kz_meshgrid ** 2)

    aa = 1/(zz+1)
    dv_dr_k_over_H  = -kx_meshgrid ** 2 / k_sq * np.interp(aa, scale_factor, dD_da) * delta_k / D(aa,param) * aa * aa
    dv_dr_k_over_H[np.where(np.isnan(dv_dr_k_over_H))] = np.interp(aa, scale_factor, dD_da) * delta_k[np.where(np.isnan(dv_dr_k_over_H))] / D(aa,param) * aa * aa   ## to deal with nan value for k=0

    dv_dr_over_H = np.real(scipy.fft.ifftn(dv_dr_k_over_H))  #### THIS IS dv_dr/H

    return dv_dr_over_H + 1





def saturated_Tspin(param,ion = None):
    """
    Computes the power spectrum and GS under the assumption that Tspin>>Tgamma (saturated).
    """
    print('Computing GS and PS under the assumption Tspin >> Tgamma')
    start_time = datetime.datetime.now()
    import tools21cm as t2c
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name
    nGrid = param.sim.Ncell
    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    factor = 27 * Ob * h0 ** 2 / 0.023 * np.sqrt(0.15 / Om / h0 ** 2 / 10)  # factor used in dTb calculation

    Lbox = param.sim.Lbox  # Mpc/h
    if isinstance(param.sim.kbin, int):
        kbins = np.logspace(np.log10(param.sim.kmin), np.log10(param.sim.kmax), param.sim.kbin, base=10)  # h/Mpc
    elif isinstance(param.sim.kbin, str):
        kbins = np.loadtxt(param.sim.kbin)
    else:
        print( 'param.sim.kbin should be either a path to a text files containing kbins edges values or it should be an int.')

    nbr_snap = 0
    for filename in os.listdir(catalog_dir):  # count the number of snapshots
        nbr_snap+=1

    PS_xHII = np.zeros((nbr_snap, len(kbins) - 1))
    PS_rho = np.zeros((nbr_snap, len(kbins) - 1))
    PS_dTb = np.zeros((nbr_snap, len(kbins) - 1))

    zz, xHII, dTb = [], [], []
    print('Looping over redshifts....')
    for ii, filename in enumerate(os.listdir(catalog_dir)):
        zz_ = load_f(catalog_dir + filename)['z']
        dens_field = param.sim.dens_fields
        if dens_field is not None:
            delta_b = load_delta_b(param, filename)
        else:
            delta_b = 0
        zz.append(zz_)

        if ion == 'exc_set':
            Grid_xHII = pickle.load( file=open('./grid_output/xHII_exc_set_' + str(nGrid) + '_' + model_name + '_snap' + filename[4:-5],'rb'))
        elif ion == 'Sem_Num':
            Grid_xHII = pickle.load(file=open('./grid_output/xHII_Sem_Num_' + str(nGrid) + '_' + model_name + '_snap' + filename[4:-5],'rb'))
        else:
            Grid_xHII = pickle.load(file=open('./grid_output/xHII_Grid' + str(nGrid) + model_name + '_snap' + filename[4:-5],'rb'))

        Grid_dTb = factor * np.sqrt(1 + zz_) * (1 - Grid_xHII) * (delta_b + 1)


        if Grid_xHII.size == 1 and zz_>20: ## arbitrary
            Grid_xHII = np.full((nGrid, nGrid, nGrid), 0)  ## to avoid div by zero
        if Grid_xHII.size == 1 and zz_<20: ## arbitrary
            Grid_xHII = np.full((nGrid, nGrid, nGrid), 1)  ## to avoid div by zero
        if Grid_dTb.size == 1:
            Grid_dTb = np.full((nGrid, nGrid, nGrid), 1)

        delta_XHII = Grid_xHII / np.mean(Grid_xHII) - 1
        delta_dTb = Grid_dTb / np.mean(Grid_dTb) - 1
        xHII.append(np.mean(Grid_xHII))
        dTb.append(np.mean(Grid_dTb))
        PS_rho[ii] = t2c.power_spectrum.power_spectrum_1d(delta_b, box_dims=Lbox, kbins=kbins)[0]
        PS_xHII[ii], k_bins = t2c.power_spectrum.power_spectrum_1d(delta_XHII, box_dims=Lbox, kbins=kbins)
        PS_dTb[ii]  = t2c.power_spectrum.power_spectrum_1d(delta_dTb, box_dims=Lbox, kbins=kbins)[0]

    z_arr, xHII, dTb = np.array(zz), np.array(xHII), np.array(dTb)
    Dict = {'z': z_arr, 'k': k_bins, 'dTb': dTb, 'xHII': xHII, 'PS_dTb': PS_dTb, 'PS_xHII': PS_xHII, 'PS_rho': PS_rho}
    end_time = datetime.datetime.now()

    print('Computing the power spectra under the assumption Tspin >> Tgamma took : ', start_time - end_time)
    pickle.dump( file=open('./physics/GS_PS_Tspin_saturated_' + str(nGrid) + '_' + model_name + '.pkl', 'wb'), obj=Dict)

    return Grid_dTb