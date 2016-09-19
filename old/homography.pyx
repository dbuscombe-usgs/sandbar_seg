"""
Daniel Buscombe, June 2016
"""
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

from scipy.linalg import svd, inv, pinv

def Haffine_from_points(np.ndarray fp, np.ndarray tp):
    """ find H, affine transformation, such that 
        tp is affine transf of fp"""

    if fp.shape != tp.shape:
        raise RuntimeError, 'number of points do not match'

    #condition points
    #-from points-
    m = np.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1))
    C1 = np.diag([1/maxstd, 1/maxstd, 1]) 
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = np.dot(C1,fp)

    #-to points-
    m = np.mean(tp[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = np.dot(C2,tp)

    #conditioned points have mean zero, so translation is zero
    A = np.concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U,S,V = svd(A.T)

    #create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = np.concatenate((np.dot(C,pinv(B)),np.zeros((2,1))), axis=1) 
    H = np.vstack((tmp2,[0,0,1]))

    #decondition
    H = np.dot(inv(C2),np.dot(H,C1))

    return H / H[2][2]
    
