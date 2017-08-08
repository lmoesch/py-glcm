import glcm as op
import numpy as np
from tests.utils.features import *

bins = 8

patch = (np.random.rand(100, 100) * bins).astype(np.int32)

c_glcm = op.glcm(patch, [1], [1, 2, 3, 4], mode="raw", bins=bins, normalized=True, symmetric=True, check=False)
asym_glcm = op.glcm(patch, [1], [1, 2, 3, 4], mode="raw", bins=bins, normalized=True, symmetric=False, check=False)
features = op.glcm_features(c_glcm, 0xFFFFFF, True)
asym_features = op.glcm_features(asym_glcm, 0xFFFFFF, normalized=True, symmetric=False)

print("blub")
print(len(asym_features))
#scik_glcm = greycomatrix(patch, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 8, symmetric=True, normed=True)

#-- MEAN --------------------------------------------------------------------------------------------------------------


#-- COMBINED PROBABILITY ----------------------------------------------------------------------------------------------

#-- ASM ---------------------------------------------------------------------------------------------------------------
if(np.allclose(asm(c_glcm), features["ASM"])):
    print("ASM (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("ASM (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(asm(asym_glcm), asym_features["ASM"])):
    print("ASM (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("ASM (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- CONTRAST ----------------------------------------------------------------------------------------------------------
if(np.allclose(contrast(c_glcm), features["Contrast"])):
    print("Contrast (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Contrast (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(contrast(asym_glcm), asym_features["Contrast"])):
    print("Contrast (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Contrast (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- ENTROPY -----------------------------------------------------------------------------------------------------------
if(np.allclose(entropy(c_glcm), features["Entropy"])):
    print("Entropy (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Entropy (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(entropy(asym_glcm), asym_features["Entropy"])):
    print("Entropy (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Entropy (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- INVERSE DIFFERENCE ------------------------------------------------------------------------------------------------
if(np.allclose(inv_diff(c_glcm), features["IDF"])):
    print("Inverse Difference (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Inverse Difference (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(inv_diff(asym_glcm), asym_features["IDF"])):
    print("Inverse Difference (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Inverse Difference (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- INVERSE DIFFERENCE MOMENT -----------------------------------------------------------------------------------------
if(np.allclose(idm(c_glcm), features["IDM"])):
    print("Inverse Difference Moment (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Inverse Difference Moment (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(idm(asym_glcm), asym_features["IDM"])):
    print("Inverse Difference Moment (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Inverse Difference Moment (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- SUM AVERAGE -------------------------------------------------------------------------------------------------------
if(np.allclose(SumAvg(c_glcm), features["Sum Average"])):
    print("Sum Average (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Sum Average (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(SumAvg(asym_glcm), asym_features["Sum Average"])):
    print("Sum Average (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Sum Average (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- SUM ENTROPY -------------------------------------------------------------------------------------------------------
if(np.allclose(sum_entrp(c_glcm), features["Sum Entropy"])):
    print("Sum Entropy (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Sum Entropy (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(sum_entrp(asym_glcm), asym_features["Sum Entropy"])):
    print("Sum Entropy (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Sum Entropy (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- SUM VARIANCE ------------------------------------------------------------------------------------------------------
if(np.allclose(sum_var(c_glcm), features["Sum Variance"])):
    print("Sum Variance (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Sum Variance (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(sum_var(asym_glcm), asym_features["Sum Variance"])):
    print("Sum Variance (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Sum Variance (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- DIFF AVERAGE -------------------------------------------------------------------------------------------------------
if(np.allclose(diff_avg(c_glcm), features["Diff Average"])):
    print("Diff Average (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Diff Average (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(diff_avg(asym_glcm), asym_features["Diff Average"])):
    print("Diff Average (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Diff Average (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- AUTOCORRELATION ---------------------------------------------------------------------------------------------------
if(np.allclose(autocorrelation(c_glcm), features["Autocorrelation"])):
    print("Autocorrelation (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Autocorrelation (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(autocorrelation(asym_glcm), asym_features["Autocorrelation"])):
    print("Autocorrelation (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Autocorrelation (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- CLUSTER PROMINENCE ------------------------------------------------------------------------------------------------
if(np.allclose(cluster_prom(c_glcm), features["Cluster Prominence"])):
    print("Cluster Prominence (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Cluster Prominence (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(cluster_prom(asym_glcm), asym_features["Cluster Prominence"])):
    print("Cluster Prominence (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Cluster Prominence (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- CLUSTER SHADE -----------------------------------------------------------------------------------------------------
if(np.allclose(cluster_shade(asym_glcm), asym_features["Cluster Shade"])):
    print("Cluster Shade (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Cluster Shade (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(cluster_shade(c_glcm), features["Cluster Shade"])):
    print("Cluster Shade (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Cluster Shade (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- CLUSTER TENDENCY --------------------------------------------------------------------------------------------------
if(np.allclose(cluster_tend(c_glcm), features["Cluster Tendency"])):
    print("Cluster Tendency (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Cluster Tendency (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(cluster_tend(asym_glcm), asym_features["Cluster Tendency"])):
    print("Cluster Tendency (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Cluster Tendency (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- DISSIMILARITY -----------------------------------------------------------------------------------------------------
if(np.allclose(dissimilarity(c_glcm), features["Dissimilarity"])):
    print("Dissimilarity (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Dissimilarity (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(dissimilarity(asym_glcm), asym_features["Dissimilarity"])):
    print("Dissimilarity (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Dissimilarity (asymmetric): \033[91m\033[1m fail!\033[0;0m")

#-- DIFFERENCE AVERAGE ------------------------------------------------------------------------------------------------
if(np.allclose(diff_avg(c_glcm), features["Dissimilarity"])):
    print("Difference Average (symmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Difference Average (symmetric): \033[91m\033[1m fail!\033[0;0m")
if(np.allclose(dissimilarity(asym_glcm), asym_features["Dissimilarity"])):
    print("Difference Average (asymmetric): \033[92m\033[1m check!\033[0;0m")
else:
    print("Difference Average (asymmetric): \033[91m\033[1m fail!\033[0;0m")






