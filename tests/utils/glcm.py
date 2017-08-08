import math
import itertools
import numpy as np

coords = [[0, 0], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]

def comb_idx(a, b, nb_chans):

    return int(a*(nb_chans-1) - (a-1)*a/2 + (b-a -1))

def xglcm(patch, ranges, dirs, nb_bins, normalized=False, symmetric=False, sum=False):

    nb_dims = patch.shape[1]
    nb_chans = patch.shape[2]

    nb_comb = int((math.factorial(nb_chans)/ (math.factorial((nb_chans-2))*2)))

    if (sum):
        if(symmetric):
            channel = 0
            x_glcm = np.zeros((len(ranges), nb_comb, nb_bins, nb_bins)).astype(np.int32)

            for dist, dir, px, py, chan_src, chan_dst in itertools.product(range(len(ranges)),
                                                                           range(len(dirs)),
                                                                           range(nb_dims),
                                                                           range(nb_dims),
                                                                           range(nb_chans),
                                                                           range(nb_chans)):
                if chan_dst <= chan_src:
                    continue

                direction = coords[dirs[dir]] * ranges[dist]

                if nb_dims > px + direction[0] >= 0 and nb_dims > py + direction[1] >= 0:
                    src = patch[px, py, chan_src]
                    dst = patch[px + direction[0], py + direction[1], chan_dst]

                    channel = comb_idx(chan_src, chan_dst, nb_chans)

                    x_glcm[dist, channel, dst, src] += 1
                    x_glcm[dist, channel, src, dst] += 1

            if (normalized):
                for dist, chan in itertools.product(range(len(ranges)),
                                                                       range(nb_comb)):
                    x_glcm = x_glcm.astype(np.float64)
                    x_glcm[dist, chan, :, :] /= np.sum(x_glcm[dist, chan, :, :])

            return x_glcm

    else:
        x_glcm = np.zeros((len(ranges), len(dirs), patch.shape[2], patch.shape[2], nb_bins, nb_bins)).astype(np.int32)

        for dist, dir, px, py, chan_src, chan_dst in itertools.product(range(len(ranges)),
                                                   range(len(dirs)),
                                                   range(nb_dims),
                                                   range(nb_dims),
                                                   range(nb_chans),
                                                   range(nb_chans)):
            direction = coords[dirs[dir]] * ranges[dist]

            if nb_dims > px + direction[0] >= 0 and nb_dims > py + direction[1] >= 0:
                    src = patch[px, py, chan_src]
                    dst = patch[px + direction[0], py + direction[1], chan_dst]

                    if(symmetric):
                        x_glcm[dist, dir, chan_src, chan_dst, dst, src] += 1
                    x_glcm[dist, dir, chan_src, chan_dst, src, dst] += 1

        if(normalized):
            for dist, dir, chan_src, chan_dst in itertools.product(range(len(ranges)),
                                                                   range(len(dirs)),
                                                                   range(nb_chans),
                                                                   range(nb_chans)):
                x_glcm = x_glcm.astype(np.float64)
                x_glcm[dist, dir, chan_src, chan_dst, :, :] /= np.sum(x_glcm[dist, dir, chan_src, chan_dst, :, :])

        return x_glcm

def glcm(patch, ranges, dirs, nb_bins, normalized=False, symmetric=False, sum=False):

    nb_dims = patch.shape[1]
    nb_chans = patch.shape[2]

    if(sum):
        glcm = np.zeros((len(ranges), patch.shape[2], nb_bins, nb_bins)).astype(np.int32)

        for dist, dir, px, py, chan_src in itertools.product(range(len(ranges)),
                                                             range(len(dirs)),
                                                             range(nb_dims),
                                                             range(nb_dims),
                                                             range(nb_chans)):
            direction = coords[dirs[dir]] * ranges[dist]

            if nb_dims > px + direction[0] >= 0 and nb_dims > py + direction[1] >= 0:
                src = patch[px, py, chan_src]
                dst = patch[px + direction[0], py + direction[1], chan_src]

                if (symmetric):
                    glcm[dist, chan_src, dst, src] += 1
                glcm[dist, chan_src, src, dst] += 1

        if (normalized):
            for dist, chan_src in itertools.product(range(len(ranges)),
                                                         range(nb_chans)):
                glcm = glcm.astype(np.float64)
                glcm[dist, chan_src, :, :] /= np.sum(glcm[dist, chan_src, :, :])

        return glcm


    else:
        glcm = np.zeros((len(ranges), len(dirs), patch.shape[2], nb_bins, nb_bins)).astype(np.int32)

        for dist, dir, px, py, chan_src in itertools.product(range(len(ranges)),
                                                   range(len(dirs)),
                                                   range(nb_dims),
                                                   range(nb_dims),
                                                   range(nb_chans)):
            direction = coords[dirs[dir]] * ranges[dist]

            if nb_dims > px + direction[0] >= 0 and nb_dims > py + direction[1] >= 0:
                    src = patch[px, py, chan_src]
                    dst = patch[px + direction[0], py + direction[1], chan_src]

                    if(symmetric):
                        glcm[dist, dir, chan_src, dst, src] += 1
                    glcm[dist, dir, chan_src, src, dst] += 1

        if(normalized):
            for dist, dir, chan_src in itertools.product(range(len(ranges)),
                                                                   range(len(dirs)),
                                                                   range(nb_chans)):
                glcm = glcm.astype(np.float64)
                glcm[dist, dir, chan_src, :, :] /= np.sum(glcm[dist, dir, chan_src, :, :])

        return glcm