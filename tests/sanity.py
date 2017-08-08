from tests.utils.glcm import glcm, xglcm
import glcm as op
import numpy as np
from scipy import ndimage
from skimage.feature import greycomatrix, greycoprops

nb_bins = 4
nb_dims = 512
nb_chans = 3
ranges = [1]
dirs = [1, 2, 3, 4]

#patch = ndimage.imread("./img/01-002.png").astype(np.int)
patch = (np.random.rand(nb_dims, nb_dims, nb_chans) * nb_bins).astype(np.int32)

#--- GLCM -------------------------------------------------------------------------------------------------------------
py_glcm = glcm(patch, ranges, dirs, nb_bins)
cglcm = op.glcm(patch, ranges, dirs, mode="raw", bins=nb_bins, normalized=False, check=False, symmetric=False)

if(np.allclose(py_glcm, cglcm)):
    print("GLCM (raw, discrete, asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("GLCM (raw, discrete, asymmetric): \033[91m\033[1m fail!\033[0;0m")

py_glcm = glcm(patch, ranges, dirs, nb_bins, normalized=True)
cglcm = op.glcm(patch, ranges, dirs, mode="raw", bins=nb_bins, normalized=True, check=False, symmetric=False)

if(np.allclose(py_glcm, cglcm)):
    print("GLCM (raw, normalized, asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("GLCM (raw, normalized, asymmetric): \033[91m\033[1m fail!\033[0;0m")

py_glcm = glcm(patch, ranges, dirs, nb_bins, sum=True, normalized=True, symmetric=True)
cglcm = op.glcm(patch, ranges, dirs, mode="sum", bins=nb_bins, normalized=True, check=False, symmetric=True)

if(np.allclose(py_glcm, cglcm)):
    print("GLCM (sum, normalized, symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("GLCM (sum, normalized, symmetric): \033[91m\033[1m fail!\033[0;0m")

py_glcm = glcm(patch, ranges, dirs, nb_bins, sum=True, normalized=False, symmetric=True)
cglcm = op.glcm(patch, ranges, dirs, mode="sum", bins=nb_bins, normalized=False, check=False, symmetric=True)

if(np.allclose(py_glcm, cglcm)):
    print("GLCM (sum, discrete, symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("GLCM (sum, discrete, symmetric): \033[91m\033[1m fail!\033[0;0m")


'''
#--- X-GLCM -----------------------------------------------------------------------------------------------------------

x_glcm = xglcm(patch, ranges, dirs, nb_bins)
x_cglcm = op.xglcm(patch, ranges, dirs, mode="raw", bins=nb_bins, normalized=False, check=False, symmetric=False)

if(np.allclose(x_glcm,x_cglcm)):
    print("X-GLCM (raw, discrete, asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("X-GLCM (raw, discrete, asymmetric): \033[91m\033[1m fail!\033[0;0m")

x_glcm = xglcm(patch, ranges, dirs, nb_bins, normalized=True)
x_cglcm = op.xglcm(patch, ranges, dirs, mode="raw", bins=nb_bins, normalized=True, check=False, symmetric=False)

if(np.allclose(x_glcm,x_cglcm)):
    print("X-GLCM (raw, normalized, asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("X-GLCM (raw, normalized, asymmetric): \033[91m\033[1m fail!\033[0;0m")

x_glcm = xglcm(patch, ranges, dirs, nb_bins, symmetric=True)
x_cglcm = op.xglcm(patch, ranges, dirs, mode="raw", bins=nb_bins, normalized=False, check=False, symmetric=True)

if(np.allclose(x_glcm,x_cglcm)):
    print("X-GLCM (raw, discrete, symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("X-GLCM (raw, discrete, asymmetric): \033[91m\033[1m fail!\033[0;0m")

x_glcm = xglcm(patch, ranges, dirs, nb_bins, normalized=True, symmetric=True)
x_cglcm = op.xglcm(patch, ranges, dirs, mode="raw", bins=nb_bins, normalized=True, check=False, symmetric=True)

if(np.allclose(x_glcm,x_cglcm)):
    print("X-GLCM (raw, normalized, symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("X-GLCM (raw, normalized , asymmetric): \033[91m\033[1m fail!\033[0;0m")
'''
x_glcm = xglcm(patch, ranges, dirs, nb_bins, normalized=True, symmetric=True, sum=True)
x_cglcm = op.xglcm(patch, ranges, dirs, mode="sum", bins=nb_bins, normalized=True, check=False, symmetric=True)

if(np.allclose(x_glcm,x_cglcm)):
    print("X-GLCM (sum, normalized, symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("X-GLCM (sum, normalized, asymmetric): \033[91m\033[1m fail!\033[0;0m")


