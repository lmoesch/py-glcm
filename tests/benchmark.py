import SimpleITK as sitk
import sys, os
import glcm as op
import numpy as np
import time
import radiomics
from scipy import ndimage
from skimage.feature import greycomatrix, greycoprops

#patch = ndimage.imread("./img/01-002.png").astype(np.int)
patch = (np.random.rand(1000, 1000, 1) * 256).astype(np.int32)
mask = np.ones(patch.shape)

itkPatch = sitk.GetImageFromArray(patch)
itkMask = sitk.GetImageFromArray(mask)

t = time.clock()
for i in range(10):
    glcm = radiomics.glcm.RadiomicsGLCM(itkPatch, itkMask, distances=[1])
t = time.clock() - t
print("Time elapsed for py-radiomics: %f" % t)

patch = patch.reshape((1000, 1000))

t = time.clock()
for i in range(10):
    glcm = greycomatrix(patch, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, symmetric=True, normed=True)
t = time.clock() - t
print("Time elapsed for scitkit: %f" % t)

t = time.clock()
for i in range(10):
    glcm = op.glcm(patch, [1], [1, 2, 3, 4], mode="raw", bins=256, normalized=True, check=False)
t = time.clock() - t
print("Time elapsed for c-glcm: %f" % t)


