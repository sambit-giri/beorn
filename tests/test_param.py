import numpy as np 
import beorn

def test_cosmo_param():
	param = beorn.param()
	assert np.abs(param.cosmo.h-0.68)<0.001