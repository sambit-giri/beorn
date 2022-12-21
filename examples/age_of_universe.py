import numpy as np 
import simple_python_package_template

# Parameters 
param = simple_python_package_template.param()
print('Cosmological parameters')
print(param.cosmo.__dict__)
print('Code parameters')
print(param.code.__dict__)

# Ages
zs = np.linspace(param.code.zmin,param.code.zmax,param.code.Nz)
t0 = simple_python_package_template.age_estimator(param, zs)

# Plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1,figsize=(5,4))
ax.plot(zs, t0)
ax.set_xlabel('Redshift')
ax.set_ylabel('Age')
plt.show()
