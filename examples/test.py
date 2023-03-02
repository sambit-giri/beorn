import numpy as np
import matplotlib.pyplot as plt
import beorn

param = beorn.par()

#sim
param.sim.M_i_min = 3e1 * np.exp(0.79*(25-40))
param.sim.M_i_max = 8e7 * np.exp(0.79*(25-40))

param.sim.model_name = 'simple_test'
param.sim.cores = 1			    # nbr of cores to use
param.sim.binn = 40				# nbr of halo mass bin


#solver
param.solver.z = 40
param.solver.z_end = 6
param.solver.Nz = 200

#cosmo
param.cosmo.Om = 0.31
param.cosmo.Ob = 0.045
param.cosmo.Ol = 0.69
param.cosmo.h = 0.68


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
param.sim.halo_catalogs ='simple_test/Halo_Catalogs/' ## path to dir with halo catalogs
param.sim.thresh_pixel = 20*(param.sim.Ncell/128)**3
param.sim.dens_fields = 'simple_test/density_field/grid'+str(param.sim.Ncell)+'_Reio_512_B100_CDM.00047.0'
param.sim.dens_field_type = 'pkdgrav'
param.sim.save_dir = 'simple_test'
param.sim.n_jobs = 1 #1
param.sim.store_grids = True #'replace' # True # False

# Step 0: Initialisation
beorn.initialise_run(param)

# Step 1 : Compute the profiles
profiles = beorn.model_profiles(param)

# Step 2 : Paint profiles in Boxes
# beorn.paint_boxes(param)
grid_outputs = beorn.paint_profiles(param, profiles=profiles)

fig, axs = plt.subplots(1,3,figsize=(14,4))
ax, sl = axs[0], grid_outputs['9.890']['ion'][1]
ax.set_title('$x_\mathrm{HII}$')
pm = ax.pcolormesh(np.linspace(0,param.sim.Lbox,sl.shape[0]), np.linspace(0,param.sim.Lbox,sl.shape[1]),
                sl, cmap='Greys')
fig.colorbar(pm, ax=ax)
ax, sl = axs[1], grid_outputs['9.890']['temp'][1]
ax.set_title('$T_\mathrm{k}$')
pm = ax.pcolormesh(np.linspace(0,param.sim.Lbox,sl.shape[0]), np.linspace(0,param.sim.Lbox,sl.shape[1]),
                sl, cmap='jet')
fig.colorbar(pm, ax=ax)
ax, sl = axs[2], grid_outputs['9.890']['dTb'][1]
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

ps, ks = t2c.power_spectrum_1d(grid_outputs['9.890']['dTb'], kbins=10, box_dims=param.sim.Lbox)

fig, ax = plt.subplots(1,1,figsize=(5,4))
ax.loglog(ks, ps*ks**3/2/np.pi**2)
ax.set_xlabel('k [1/Mpc]')
ax.set_ylabel('$\Delta^2_\mathrm{21}$')
plt.show()