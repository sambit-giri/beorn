import numpy as np
import beorn

param = beorn.par()

#sim
param.sim.M_i_min = 3e1 * np.exp(0.79*(25-40))
param.sim.M_i_max = 8e7 * np.exp(0.79*(25-40))

param.sim.model_name = 'test'
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
Ncell = 128
Lbox = 100
param.sim.Lbox = Lbox
param.sim.Ncell = Ncell
param.sim.halo_catalogs ='./Halo_Catalogs/' ## path to dir with halo catalogs
param.sim.thresh_pixel = 20*(Ncell/128)**3
param.sim.dens_field = './density_field/grid'+str(Ncell)+'_Reio_512_B100_CDM.000'
param.sim.dens_field_type = 'pkdgrav'

# Step 1 : Compute the profiles
beorn.compute_profiles(param)

# Step 2 : Paint Boxes
beorn.paint_boxes(param)

# Step 3 : Compute GS
beorn.compute_GS(param,RSD=False)

# Step 4 : Compute PS
kmin = 1/Lbox
kmax = Ncell/Lbox
kbin  = int(6* np.log10(kmax/kmin))
param.sim.kmin = kmin
param.sim.kmax = kmax
param.sim.kbin = kbin

beorn.compute_PS(param, Tspin = False, RSD = False, cross_corr = False)