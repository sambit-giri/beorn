import numpy as np 

def z_to_a(z):
    """
    Convert redshift to scale factor.

    Parameters
    ----------
    z: float 
        Redshift

    Returns
    -------
    a: float
        The corresponding scale factor
    """
    return 1/(1+z)

def a_to_z(a):
    """
    Convert scale factor to redshift.

    Parameters
    ----------
    a: float 
        Scale factor

    Returns
    -------
    z: float
        The corresponding redshift
    """
    return 1./a-1