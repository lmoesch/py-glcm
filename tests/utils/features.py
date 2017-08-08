import numpy as np

eps = 2.22045e-16

def mean(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    mean = np.zeros((np.concatenate((dims, [2, nb_dim]))))

    for index in np.ndindex(dims):
        for px in range(nb_dim):
            for py in range(nb_dim):
                mean[index][0][px] += glcm[index][px][py]
                mean[index][1][py] += glcm[index][px][py]

    return mean

def combined_sum(glcm):
    nb_bins = glcm.shape[-1]

    cp = np.zeros((2*nb_bins))

    for px in range(nb_bins):
        for py in range(nb_bins):
            cp[px+py] += glcm[px, py]

    return cp

def combined_diff(glcm):
    nb_bins = glcm.shape[-1]

    cd = np.zeros((2 * nb_bins))

    for px in range(nb_bins):
        for py in range(nb_bins):
            cd[abs(px - py)] += glcm[px, py]

    return cd

#-- FEATURES ----------------------------------------------------------------------------------------------------------

def asm(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    asm = np.zeros((dims))

    for index in np.ndindex(dims):
        for px in range(nb_dim):
            for py in range(nb_dim):
                asm[index] += glcm[index][px][py]*glcm[index][px][py]

    return asm

def contrast(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    contrast = np.zeros((dims))

    for index in np.ndindex(dims):
        for px in range(nb_dim):
            for py in range(nb_dim):
                contrast[index] += (px-py)*(px-py)*glcm[index][px][py]

    return contrast

def entropy(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    entropy = np.zeros((dims))

    for index in np.ndindex(dims):
        for px in range(nb_dim):
            for py in range(nb_dim):
                entropy[index] -= glcm[index][px][py]*np.log10(glcm[index][px][py] + eps)

    return entropy

def idm(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    idm = np.zeros((dims))

    for index in np.ndindex(dims):
        for px in range(nb_dim):
            for py in range(nb_dim):
                idm[index] += (1/(1 + (px-py)*(px-py)))*glcm[index][px][py]

    return idm

def SumAvg(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    savg = np.zeros((dims))

    for index in np.ndindex(dims):
        cp = combined_sum(glcm[index][:])

        for px in range(2*nb_dim):
                savg[index] += (px + 2)*cp[px]

    return savg

def sum_entrp(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    sntrp = np.zeros((dims))

    for index in np.ndindex(dims):
        cp = combined_sum(glcm[index][:])

        for px in range(2*nb_dim):
            sntrp[index] += cp[px]*np.log10(cp[px] + eps)

    return sntrp

def sum_var(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    svar = np.zeros((dims))
    sa = SumAvg(glcm)

    for index in np.ndindex(dims):
        cp = combined_sum(glcm[index][:])

        for px in range(2*nb_dim):
            svar[index] += (px + 2 - sa[index])**2 * cp[px]

    return svar

def diff_avg(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    davg = np.zeros((dims))

    for index in np.ndindex(dims):
        cd = combined_diff(glcm[index][:])

        for px in range(nb_dim):
            davg[index] += px * cd[px]

    return davg

def autocorrelation(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    autocorr = np.zeros((dims))

    for index in np.ndindex(dims):
        for px in range(nb_dim):
            for py in range(nb_dim):
                autocorr[index] += glcm[index][px][py]*(px+1)*(py+1)

    return autocorr

def cluster_prom(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    cprom = np.zeros((dims))
    u = mean(glcm)

    for index in np.ndindex(dims):
        for px in range(nb_dim):
            for py in range(nb_dim):
                cprom[index] += (px + py + 2 - u[index][0][px] - u[index][1][py])**4 * glcm[index][px][py]

    return cprom

def cluster_shade(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    cshad = np.zeros((dims))
    u = mean(glcm)

    for index in np.ndindex(dims):
        for px in range(nb_dim):
            for py in range(nb_dim):
                cshad[index] += (px + py + 2 - u[index][0][px] - u[index][1][py])**3 * glcm[index][px][py]

    return cshad

def cluster_tend(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    ctend = np.zeros((dims))
    u = mean(glcm)

    for index in np.ndindex(dims):
        for px in range(nb_dim):
            for py in range(nb_dim):
                ctend[index] += (px + py + 2 - u[index][0][px] - u[index][1][py])**2 * glcm[index][px][py]

    return ctend

def dissimilarity(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    autocorr = np.zeros((dims))

    for index in np.ndindex(dims):
        for px in range(nb_dim):
            for py in range(nb_dim):
                autocorr[index] += abs(px-py) * glcm[index][px][py]

    return autocorr

def inv_diff(glcm):
    nb_dim = glcm.shape[-1]
    dims = glcm.shape[:-2]

    ivd = np.zeros((dims))

    for index in np.ndindex(dims):
        for px in range(nb_dim):
            for py in range(nb_dim):
                ivd[index] += glcm[index][px][py] / (1 + abs(px-py))

    return ivd

