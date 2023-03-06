import numpy as np
import matplotlib.pyplot as plt
import beorn

param = beorn.par()

#sim
param.sim.M_i_min = 3e1 * np.exp(0.79*(25-40))
param.sim.M_i_max = 8e7 * np.exp(0.79*(25-40))
param.sim.model_name = 'py21cmfast_test'
param.sim.cores = 1			    # nbr of cores to use
param.sim.binn = 40				# nbr of halo mass bin


#solver
param.solver.z_ini = 40
param.solver.z_end = 6
param.solver.Nz = 200

#cosmo
param.cosmo.Om = 0.31
param.cosmo.Ob = 0.045
param.cosmo.Ol = 0.69
param.cosmo.h  = 0.68
param.cosmo.ps = "PCDM_Planck.dat"
param.cosmo.corr_fct = 'corr_fct.dat'

## Source parameters
#lyal
param.source.N_al = 9690#1500
param.source.alS_lyal = 0.0
#ion
param.source.Nion = 3000     #5000
#xray
param.source.E_min_xray = 500
param.source.E_max_xray = 10000
param.source.E_min_sed_xray = 200
param.source.E_max_sed_xray = 10000
param.source.alS_xray =  1.5
param.source.cX = 3.4e40
#fesc
param.source.f0_esc = 0.2
param.source.pl_esc = 0.5
#fstar
param.source.f_st = 0.14
param.source.g1 = 0.49
param.source.g2 = -0.61
param.source.g3 = 4
param.source.g4= -4
param.source.Mp= 1.6e11 * param.cosmo.h
param.source.Mt= 1e9
# Minimum star forming halo
param.source.M_min= 1e8

# Box size and Number of pixels
param.sim.Lbox = 100
param.sim.Ncell = 128
param.sim.halo_catalogs = None
param.sim.thresh_pixel = 20*(param.sim.Ncell/128)**3
param.sim.dens_fields = None
param.sim.dens_field_type = '21cmfast'
param.sim.data_dir = 'py21cmfast_test/'
param.sim.n_jobs = 1 #1
param.sim.store_grids = False #True #'replace' # True # False

# # Simulate matter evolution
# redshift = 8.0
# matter_field = beorn.simulate_matter_21cmfast(redshift, param, IC=None)

# param.sim.dens_fields   = [matter_field['dens']]
# param.sim.halo_catalogs = [matter_field['halo_list']]

# # Simulate cosmic dawn
# beorn.initialise_run(param)
# profiles = beorn.model_profiles(param, method='simple')
# grid_outputs = beorn.paint_profiles(param, profiles=profiles)

redshift = 8.0
grid_outputs = beorn.run_coeval_EOR_CD(redshift, param)

fig, axs = plt.subplots(1,3,figsize=(14,4))
ax, sl = axs[0], grid_outputs['{:.3f}'.format(redshift)]['ion'][1]
ax.set_title('$x_\mathrm{HII}$')
pm = ax.pcolormesh(np.linspace(0,param.sim.Lbox,sl.shape[0]), np.linspace(0,param.sim.Lbox,sl.shape[1]),
                sl, cmap='Greys')
fig.colorbar(pm, ax=ax)
ax, sl = axs[1], grid_outputs['{:.3f}'.format(redshift)]['dens'][1]
ax.set_title('$\delta_\mathrm{b}$') #ax.set_title('$T_\mathrm{k}$')
pm = ax.pcolormesh(np.linspace(0,param.sim.Lbox,sl.shape[0]), np.linspace(0,param.sim.Lbox,sl.shape[1]),
                sl, cmap='jet')
fig.colorbar(pm, ax=ax)
ax, sl = axs[2], grid_outputs['{:.3f}'.format(redshift)]['dTb'][1]
ax.set_title('$\delta T_\mathrm{b}$')
pm = ax.pcolormesh(np.linspace(0,param.sim.Lbox,sl.shape[0]), np.linspace(0,param.sim.Lbox,sl.shape[1]),
                sl, cmap='jet')
fig.colorbar(pm, ax=ax)
for ax in axs.flatten():
    ax.set_xlabel('Mpc', fontsize=15)
    ax.set_ylabel('Mpc', fontsize=15)
plt.tight_layout()
plt.show()

# Step 3 : Statitical measures
import tools21cm as t2c

ps_dn, ks = t2c.power_spectrum_1d(grid_outputs['{:.3f}'.format(redshift)]['dens'], kbins=10, box_dims=param.sim.Lbox)
ps_in, ks = t2c.power_spectrum_1d(grid_outputs['{:.3f}'.format(redshift)]['ion'], kbins=10, box_dims=param.sim.Lbox)
# ps_Tk, ks = t2c.power_spectrum_1d(grid_outputs['{:.3f}'.format(redshift)]['temp'], kbins=10, box_dims=param.sim.Lbox)
ps_dT, ks = t2c.power_spectrum_1d(grid_outputs['{:.3f}'.format(redshift)]['dTb'], kbins=10, box_dims=param.sim.Lbox)

fig, axs = plt.subplots(1,2,figsize=(10,4))
ax = axs[0]
ax.loglog(ks, ps_dT*ks**3/2/np.pi**2)
ax.set_ylabel('$\Delta^2_\mathrm{21}$', fontsize=15)
ax.axis([7e-2,7,1,1e3])
ax = axs[1]
ax.loglog(ks, ps_dT*ks**3/2/np.pi**2/grid_outputs['{:.3f}'.format(redshift)]['dTb'].mean()**2, label='$\delta_\mathrm{21}$')
ax.loglog(ks, ps_dn*ks**3/2/np.pi**2, ls='--', label='$\delta_\mathrm{b}$')
ax.loglog(ks, ps_in*ks**3/2/np.pi**2, ls='-.', label='$\delta_\mathrm{x}$')
# ax.loglog(ks, ps_Tk*ks**3/2/np.pi**2, ls=':', label='$\delta_\mathrm{T}$')
ax.set_ylabel('$\Delta^2_\mathrm{\delta}$', fontsize=15)
ax.legend()
for ax in axs: ax.set_xlabel('k [1/Mpc]', fontsize=15)
plt.tight_layout()
plt.show()