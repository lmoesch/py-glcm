#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <bitset>

/* --- INTRINSICS -------------------------------------------------------------------------------------------------- */
#pragma intrinsic(__popcnt)

/* --- MAKROS ------------------------------------------------------------------------------------------------------ */

#define MAX(a,b) ((a < b) ?  (b) : (a))

#define PyArray_GETPTR5(obj, i, j, k, l, m) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1] + \
                                            (k)*PyArray_STRIDES(obj)[2] + \
                                            (l)*PyArray_STRIDES(obj)[3] + \
                                            (m)*PyArray_STRIDES(obj)[4]))

#define PyArray_GETPTR6(obj, i, j, k, l, m, n) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1] + \
                                            (k)*PyArray_STRIDES(obj)[2] + \
                                            (l)*PyArray_STRIDES(obj)[3] + \
                                            (m)*PyArray_STRIDES(obj)[4] + \
                                            (n)*PyArray_STRIDES(obj)[5]))

/* ----------------------------------------------------------------------------------------------------------------- */

#define ASM         1 <<  0
#define CONTRAST    1 <<  1
#define CORRELATION 1 <<  2
#define SSQ         1 <<  3
#define INVDIFFM    1 <<  4
#define SUMAVG      1 <<  5
#define SUMVAR      1 <<  6
#define SUMENTRP    1 <<  7
#define ENTROPY     1 <<  8
#define DIFFVAR     1 <<  9
#define DIFFENTRP   1 << 10
#define AUTOCORR    1 << 11
#define CLSTRPROM   1 << 12
#define CLSTRSHAD   1 << 13
#define CLSTRTEND   1 << 14
#define DISSIM      1 << 15
#define INVDIFF     1 << 16
#define DIFFAVG     1 << 15

#define eps  2.22045e-16

/* --- UTILS ------------------------------------------------------------------------------------------------------- */
int factorial(int n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_bin(PyArrayObject* iarr, int bins)
{
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    PyArrayObject *op[2], *oarr;
    npy_intp *stride, *innersizeptr, size, max;
    npy_uint32 op_flags[2];
    PyArray_Descr *op_dtypes[2];
    char **dataptr;
    int i, *dst;

    PyArray_ArgFunc *argmax;

    argmax = PyArray_DESCR(iarr)->f->argmax;
    argmax(PyArray_DATA(iarr), PyArray_SIZE(iarr), &max, NULL);

    op[0] = iarr;
    op[1] = NULL;
    op_flags[0] = NPY_ITER_READONLY;
    op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[0] = NULL;
    op_dtypes[1] = PyArray_DescrFromType(NPY_INT);

    iter = NpyIter_MultiNew(2, op, NPY_ITER_EXTERNAL_LOOP,
                                   NPY_KEEPORDER,
                                   NPY_UNSAFE_CASTING,
                            op_flags, op_dtypes);

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
    dataptr = NpyIter_GetDataPtrArray(iter);
    stride = NpyIter_GetInnerStrideArray(iter);

    /* TODO: There has to be a way to do this via ufunc for every datatype
             Maybbe port it to C++ and use templates?
    */
    if(PyArray_ISINTEGER(iarr)){
        int src;
        double amax = (double)((int*) PyArray_DATA(iarr))[max];
        do {
            size = *innersizeptr;
            for(i = 0; i < size; i++, dataptr[0] += stride[0], dataptr[1] += stride[1]){
                    src = *((int*) dataptr[0]);
                    dst = (int*) dataptr[1];

                    *dst = (int) ((src/amax) * bins + 0.5);
            }
        } while(iternext(iter));
    }else{
        double src, amax = ((double*) PyArray_DATA(iarr))[max];

        do {
            size = *innersizeptr;
            for(i = 0; i < size; i++, dataptr[0] += stride[0], dataptr[1] += stride[1]){
                    src = *((double*) dataptr[0]);
                    dst = (int*) dataptr[1];

                    *dst = (int) ((src/amax) * bins + 0.5);
            }
        } while(iternext(iter));
    }

    oarr = (PyArrayObject*) NpyIter_GetOperandArray(iter)[1];
    Py_INCREF(oarr);

    NpyIter_Deallocate(iter);

    return oarr;
}

NPY_NO_EXPORT npy_intp
PyArray_get_max_idx(npy_intp ndim, npy_intp* dims)
{
    npy_intp i, maxdims=1;

    for(i = 0; i < ndim; i++)
    {
        maxdims *= dims[i];
    }

    return  maxdims;
}

NPY_NO_EXPORT npy_intp*
PyArray_get_multi_index(npy_intp ndim, npy_intp* dims, npy_intp index)
{
    npy_intp idx[NPY_MAXDIMS] = {0};
    int i, div=1;

    for(i = ((int) ndim) - 1; i >= 0; i--){
        idx[i] = (npy_intp) (index / div) % (int) dims[i];
        div *= (int) dims[i];
    }

    return idx;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_glcm_gen_dirs(PyArrayObject* dirs, PyArrayObject* dists, int symmetric)
{
    PyArrayObject *oarr;
    npy_intp shape[3]={0, 0, 2};
    int data, dir_bits=0, i, j, *dir_x, *dir_y;

    for(i = 0; i < PyArray_DIMS(dirs)[0]; i++){
        data = *((int *) PyArray_GETPTR1(dirs, i));

        if(data > 4 && symmetric){
            dir_bits |= (1 << (data - 4));
        }
        else{
            dir_bits |= (1 << data);
        }
    }

    shape[0] = PyArray_DIMS(dists)[0];
    shape[1] = __popcnt(dir_bits); // only works with MSVC, for GCC use __builtin_popcount();

    oarr = (PyArrayObject *) PyArray_ZEROS(3, shape, NPY_INT, 0);

    for(i = 0; i < PyArray_DIMS(dists)[0]; i++){
        int k = 0, dist = *((int*) PyArray_GETPTR1(dists, i));
        for(j = 0; j < 9; j++){
            if((dir_bits >> j) & 1){
                dir_x = (int*) PyArray_GETPTR3(oarr, i, k, 0);
                dir_y = (int*) PyArray_GETPTR3(oarr, i, k, 1);
                switch(j){
                    case 0: *dir_x = 0; *dir_y = 0; break;
                    case 1: *dir_x = 0; *dir_y = dist; break;
                    case 2: *dir_x = dist; *dir_y = dist; break;
                    case 3: *dir_x = dist; *dir_y = 0; break;
                    case 4: *dir_x = dist; *dir_y = -dist; break;
                    case 5: *dir_x = 0; *dir_y = -dist; break;
                    case 6: *dir_x = -dist; *dir_y = -dist; break;
                    case 7: *dir_x = -dist; *dir_y = 0; break;
                    case 8: *dir_x = -dist; *dir_y = dist; break;
                    default: PyErr_SetString(PyExc_TypeError,
                                "directions have to be between 0 and 8");
                             return NULL;
                }
                k++;
            }
        }
    }

    return oarr;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_normalize(PyArrayObject* glcm)
{
    PyArrayObject *sum;
    npy_intp ndim, *shape;
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    int px,py;
    char **dataptr;
    npy_intp *strideptr, idx[NPY_MAXDIMS] = {0};

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);

    sum = (PyArrayObject*) PyArray_ZEROS((int) (ndim - 2), shape, NPY_INT, 0);
    //glcm = (PyArrayObject*) PyArray_Cast(glcm, NPY_DOUBLE);

    iter = NpyIter_New(sum, NPY_ITER_READWRITE |
                        NPY_ITER_MULTI_INDEX,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
        return NULL;
    }
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);

    NpyIter_GetMultiIndexFunc *get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL); /* I'm a totally documented function! */

    do {
        int *dst = (int*) *dataptr;
        double *src;
        npy_intp stride = *strideptr;

        get_multi_index(iter, idx);
        src = (double*) PyArray_GetPtr(glcm, idx);

        for(px = 0; px < shape[ndim - 2] ; px++){
            for(py = 0; py < shape[ndim - 3]; py++, src++){
                 *dst += (int) *src;
            }
        }

        /*src = PyArray_GetPtr(glcm, idx);

        for(px = 0; px < shape[ndim - 2] ; px++){
            for(py = 0; py < shape[ndim - 3]; py++, src++){
                 *src /= *dst;
            }
        }*/

        dst += stride;
    } while(iternext(iter));


    NpyIter_Deallocate(iter);

    return glcm;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_pad(PyArrayObject* inarr, int width)
{
    PyArrayObject *oarr=NULL;
    PyArray_Descr *dtype;
    npy_intp *ishape, oshape[NPY_MAXDIMS];
    int i, ndim;

    ndim = PyArray_NDIM(inarr);
    dtype = PyArray_DTYPE(inarr);
    ishape = PyArray_SHAPE(inarr);

    for (i = 0; i < ndim; i++){
        oshape[i] = ishape[i] + 2 * width;
    }

    oarr = (PyArrayObject *) PyArray_ZEROS((int) ndim, oshape, dtype->type, 0);

    if(ndim == 2){
        for (i = 0; i < ishape[0]; i++){
            memcpy(PyArray_GETPTR2(oarr, i+width, width), PyArray_GETPTR2(inarr, i, 0), ishape[1] * dtype->elsize);
        }
    }
    else if(ndim == 3){
        /* TODO: implement 3D padding */
    }
    else{
        PyErr_SetString(PyExc_TypeError,
                "cannot pad arrays with less than 2 or more than 3 dimensions");
    }

    return oarr;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_glcm_mean(PyArrayObject* glcm, int symmetric)
{
    PyArrayObject *mean;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    shape[ndim-2] = 2;
    mean = (PyArrayObject*) PyArray_ZEROS((int) ndim, shape, NPY_DOUBLE, 0);

    shape[ndim-2] = bins; //Todo: Superhacky...

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index(ndim - 2, shape, idx);

        dst = (double *) PyArray_GetPtr(mean, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);

        if(symmetric){
            double *data = src;

            for(px = 0; px < bins; px++){
                data = &(src[px*bins]);

                for(py = 0; py <= (px-1); py++, data++){
                    *(dst + px) += (*data);
                    *(dst + py) += (*data);
                    *(dst + bins + px) += (*data);
                    *(dst + bins + py) += (*data);
                }
                *(dst + px) += (*data);
                *(dst + bins + px) += (*data);
            }
        }else{
             for(px = 0; px < bins; px++){
                for(py = 0; py < bins; py++, src++){
                    *(dst + px) += (*src);
                    *(dst + bins + py) += (*src);
                }
             }
        }
    }

    return mean;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_comb_prob(PyArrayObject* glcm, int symmetric, char* mode)
{
    PyArrayObject *prob;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    shape[ndim-2] = 2*bins - 1;
    prob = (PyArrayObject*) PyArray_ZEROS((int) (ndim - 1), shape, NPY_DOUBLE, 0);
    shape[ndim-2] = bins;

    max_idx = PyArray_get_max_idx(ndim - 2, shape);

    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index((int) (ndim - 2), shape, idx);

        dst = (double *) PyArray_GetPtr(prob, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);

        //Todo: Add mode for sum and diff
        if(symmetric){
            double *data = src;

            for(px = 0; px < bins; px++){
                data = &(src[px*bins]);

                for(py = 0; py <= (px - 1); py++, data++){
                    dst[px + py] += 2 * (*data);
                }
                dst[px+py] += *data;
            }
        }else{
             for(px = 0; px < bins; px++){
                for(py = 0; py < bins; py++, src++){
                    dst[px + py] += (*src);
                }
             }
        }
    }

    return prob;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_comb_diff(PyArrayObject* glcm, int symmetric)
{
    PyArrayObject *prob;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    prob = (PyArrayObject*) PyArray_ZEROS((int) (ndim - 1), shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);

    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index((int) (ndim - 2), shape, idx);

        dst = (double *) PyArray_GetPtr(prob, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);

        if(symmetric){
            double *data = src;

            for(px = 0; px < bins; px++){
                data = &(src[px*bins]);

                for(py = 0; py <= (px - 1); py++, data++){
                    dst[px - py] += 2 * (*data);
                }
                dst[px - py] += *data;
            }
        }else{
             for(px = 0; px < bins; px++){
                for(py = 0; py < px; py++, src++){
                    dst[px - py] += (*src);
                }
                for(py = px; py < bins; py++, src++){
                    dst[py - px] += (*src);
                }
             }
        }
    }

    return prob;
}
/* --- GLCM -------------------------------------------------------------------------------------------------------- */

template <typename PTR_OUT, int NPY_OTYPE> NPY_NO_EXPORT PyArrayObject*
PyArray_glcm_sum(PyArrayObject* iarr, PyArrayObject* dirs, PyArrayObject* dists, int bins, int symmetric, int normalized)
{
    PyArrayObject *glcm;
    npy_intp glcm_dims[NPY_MAXDIMS];
    int dist,dx,dy,dd,ddst,ddx,ddy,arr_dimx,arr_dimy,arr_ndch,dir_dim,dist_dim, ids,idt, *dpx, *dpy;

    PTR_OUT *val, add=1.0;

    if(PyArray_NDIM(iarr) == 2){
        arr_dimx = (int) PyArray_DIMS(iarr)[0];
        arr_dimy = (int) PyArray_DIMS(iarr)[1];
        dist_dim = (int) PyArray_DIMS(dists)[0];
        dir_dim = (int) PyArray_DIMS(dirs)[1];

        glcm_dims[0] = dist_dim;
        glcm_dims[1] = bins;
        glcm_dims[2] = bins;

        glcm = (PyArrayObject*) PyArray_ZEROS(3, glcm_dims, NPY_OTYPE, 0);

        if(symmetric){
            for(ddst = 0; ddst < dist_dim; ddst++){
                dist = *((int*) PyArray_GETPTR1(dists, ddst));

                for (dd = 0; dd < dir_dim; dd++){
                    ddx = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 0));
                    ddy = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 1));

                    if(normalized){
                        int npx = ((arr_dimx - MAX(0, ddx)) - MAX(0, -ddx)) * ((arr_dimy - MAX(0, ddy)) - MAX(0, -ddy)) * 2 * dir_dim;
                        add = 1.0/npx;
                    }

                    for (dx = MAX(0, -ddx); dx < (arr_dimx - MAX(0, ddx)); dx++){
                        dpx = (int*) PyArray_GETPTR2(iarr, dx, MAX(0, -ddy));
                        dpy = (int*) PyArray_GETPTR2(iarr, dx + ddx, MAX(0, -ddy) + ddy);

                        for (dy = MAX(0, -ddy); dy < (arr_dimy - MAX(0, ddy)); dy++, dpx++, dpy++){
                            val = (PTR_OUT*) PyArray_GETPTR3(glcm, ddst, *dpx, *dpy);
                            *val += add;

                            val = (PTR_OUT*) PyArray_GETPTR3(glcm, ddst, *dpy, *dpx);
                            *val += add;
                        }
                    }
                 }
            }
        }else{
            for(ddst = 0; ddst < dist_dim; ddst++){
                dist = *((int*) PyArray_GETPTR1(dists, ddst));

                for (dd = 0; dd < dir_dim; dd++){
                    ddx = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 0));
                    ddy = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 1));

                    if(normalized){
                        int npx = ((arr_dimx - MAX(0, ddx)) - MAX(0, -ddx)) * ((arr_dimy - MAX(0, ddy)) - MAX(0, -ddy)) * dir_dim;
                        add = 1.0/npx;
                    }

                    for (dx = MAX(0, -ddx); dx < (arr_dimx - MAX(0, ddx)); dx++){
                        dpx = (int*) PyArray_GETPTR2(iarr, dx, MAX(0, -ddy));
                        dpy = (int*) PyArray_GETPTR2(iarr, dx + ddx, MAX(0, -ddy) + ddy);

                        for (dy = MAX(0, -ddy); dy < (arr_dimy - MAX(0, ddy)); dy++){
                            val = (PTR_OUT*) PyArray_GETPTR3(glcm, ddst, *dpx, *dpy);
                            *val += add;
                        }
                    }
                 }
            }
        }

        return glcm;
    }
    else if(PyArray_NDIM(iarr) == 3) {
        int dc_src;

        arr_dimx = (int) PyArray_DIMS(iarr)[0];
        arr_dimy = (int) PyArray_DIMS(iarr)[1];
        arr_ndch = (int) PyArray_DIMS(iarr)[2];
        dist_dim = (int) PyArray_DIMS(dists)[0];
        dir_dim = (int) PyArray_DIMS(dirs)[1];

        glcm_dims[0] = dist_dim;
        glcm_dims[1] = arr_ndch;
        glcm_dims[2] = bins;
        glcm_dims[3] = bins;

        glcm = (PyArrayObject*) PyArray_ZEROS(4, glcm_dims, NPY_OTYPE, 0);

        if(symmetric){
            for(ddst = 0; ddst < dist_dim; ddst++){

                dist = *((int*) PyArray_GETPTR1(dists, ddst));
                for (dd = 0; dd < dir_dim; dd++){
                    ddx = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 0));
                    ddy = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 1));

                    if(normalized){
                        int npx = ((arr_dimx - MAX(0, ddx)) - MAX(0, -ddx)) * ((arr_dimy - MAX(0, ddy)) - MAX(0, -ddy)) * 2 * dir_dim;
                        add = 1.0/npx;
                    }
                    for(dc_src = 0; dc_src < arr_ndch; dc_src++){
                        for (dx = MAX(0, -ddx); dx < (arr_dimx - MAX(0, ddx)); dx++){
                            dpx = (int*) PyArray_GETPTR3(iarr, dx, MAX(0, -ddy), dc_src);
                            dpy = (int*) PyArray_GETPTR3(iarr, dx + ddx, MAX(0, -ddy) + ddy, dc_src);

                            for (dy = MAX(0, -ddy); dy < (arr_dimy - MAX(0, ddy)); dy++, dpy+=arr_ndch, dpx+=arr_ndch){
                                val = (PTR_OUT*) PyArray_GETPTR4(glcm, ddst, dc_src, *dpx, *dpy);
                                *val += add;

                                val = (PTR_OUT*) PyArray_GETPTR4(glcm, ddst, dc_src, *dpy, *dpx);
                                *val += add;
                            }
                        }
                    }
                }
            }
        }else{
            for(ddst = 0; ddst < dist_dim; ddst++){
                dist = *((int*) PyArray_GETPTR1(dists, ddst));

                for (dd = 0; dd < dir_dim; dd++){
                    ddx = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 0));
                    ddy = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 1));

                    if(normalized){
                        int npx = ((arr_dimx - MAX(0, ddx)) - MAX(0, -ddx)) * ((arr_dimy - MAX(0, ddy)) - MAX(0, -ddy)) * dir_dim;
                        add = 1.0/npx;
                    }

                    for(dc_src = 0; dc_src < arr_ndch; dc_src++){
                        for (dx = MAX(0, -ddx); dx < (arr_dimx - MAX(0, ddx)); dx++){
                            dpx = (int*) PyArray_GETPTR3(iarr, dx, MAX(0, -ddy), dc_src);
                            dpy = (int*) PyArray_GETPTR3(iarr, dx + ddx, MAX(0, -ddy) + ddy, dc_src);

                            for (dy = MAX(0, -ddy); dy < (arr_dimy - MAX(0, ddy)); dy++, dpx+=arr_ndch, dpy+=arr_ndch){
                                val = (PTR_OUT*) PyArray_GETPTR4(glcm, ddst, dc_src, *dpx, *dpy);
                                *val += add;
                            }
                        }
                    }
                }
            }
        }

        return glcm;
    }
    else{
        PyErr_SetString(PyExc_TypeError,
                "invalid array dimensions");
        return NULL;
    }

    return NULL;
}

template <typename PTR_OUT, int NPY_OTYPE> NPY_NO_EXPORT PyArrayObject*
PyArray_glcm_raw(PyArrayObject* iarr, PyArrayObject* dirs, PyArrayObject* dists, int bins, int symmetric, int normalized)
{
    PyArrayObject *glcm;
    npy_intp glcm_dims[NPY_MAXDIMS];
    int dist,dx,dy,dd,ddst,ddx,ddy,arr_dimx,arr_dimy,arr_ndch,dir_dim,dist_dim, ids,idt, *dpx, *dpy;

    PTR_OUT *val, add=1.0;

    if(PyArray_NDIM(iarr) == 2){
        arr_dimx = (int) PyArray_DIMS(iarr)[0];
        arr_dimy = (int) PyArray_DIMS(iarr)[1];
        dist_dim = (int) PyArray_DIMS(dists)[0];
        dir_dim = (int) PyArray_DIMS(dirs)[1];

        glcm_dims[0] = dist_dim;
        glcm_dims[2] = bins;
        glcm_dims[3] = bins;
        glcm_dims[1] = dir_dim;

        glcm = (PyArrayObject*) PyArray_ZEROS(4, glcm_dims, NPY_OTYPE, 0);

        if(symmetric){
            for(ddst = 0; ddst < dist_dim; ddst++){
                dist = *((int*) PyArray_GETPTR1(dists, ddst));

                for (dd = 0; dd < dir_dim; dd++){
                    ddx = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 0));
                    ddy = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 1));

                    if(normalized){
                        int npx = ((arr_dimx - MAX(0, ddx)) - MAX(0, -ddx)) * ((arr_dimy - MAX(0, ddy)) - MAX(0, -ddy)) * 2;
                        add = 1.0/npx;
                    }

                    for (dx = MAX(0, -ddx); dx < (arr_dimx - MAX(0, ddx)); dx++){
                        dpx = (int*) PyArray_GETPTR2(iarr, dx, MAX(0, -ddy));
                        dpy = (int*) PyArray_GETPTR2(iarr, dx + ddx, MAX(0, -ddy) + ddy);

                        for (dy = MAX(0, -ddy); dy < (arr_dimy - MAX(0, ddy)); dy++, dpx++, dpy++){
                            val = (PTR_OUT*) PyArray_GETPTR4(glcm, ddst, dd, *dpx, *dpy);
                            *val += add;

                            val = (PTR_OUT*) PyArray_GETPTR4(glcm, ddst, dd, *dpy, *dpx);
                            *val += add;
                        }
                    }
                 }
            }
        }else{
            for(ddst = 0; ddst < dist_dim; ddst++){
                dist = *((int*) PyArray_GETPTR1(dists, ddst));

                for (dd = 0; dd < dir_dim; dd++){
                    ddx = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 0));
                    ddy = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 1));

                    if(normalized){
                        int npx = ((arr_dimx - MAX(0, ddx)) - MAX(0, -ddx)) * ((arr_dimy - MAX(0, ddy)) - MAX(0, -ddy));
                        add = 1.0/npx;
                    }

                    for (dx = MAX(0, -ddx); dx < (arr_dimx - MAX(0, ddx)); dx++){
                        dpx = (int*) PyArray_GETPTR2(iarr, dx, MAX(0, -ddy));
                        dpy = (int*) PyArray_GETPTR2(iarr, dx + ddx, MAX(0, -ddy) + ddy);

                        for (dy = MAX(0, -ddy); dy < (arr_dimy - MAX(0, ddy)); dy++){
                            val = (PTR_OUT*) PyArray_GETPTR4(glcm, ddst, dd, *dpx, *dpy);
                            *val += add;
                        }
                    }
                 }
            }
        }

        return glcm;
    }
    else if(PyArray_NDIM(iarr) == 3) {
        int dc_src;

        arr_dimx = (int) PyArray_DIMS(iarr)[0];
        arr_dimy = (int) PyArray_DIMS(iarr)[1];
        arr_ndch = (int) PyArray_DIMS(iarr)[2];
        dist_dim = (int) PyArray_DIMS(dists)[0];
        dir_dim = (int) PyArray_DIMS(dirs)[1];

        glcm_dims[0] = dist_dim;
        glcm_dims[1] = dir_dim;
        glcm_dims[2] = arr_ndch;
        glcm_dims[3] = bins;
        glcm_dims[4] = bins;

        glcm = (PyArrayObject*) PyArray_ZEROS(5, glcm_dims, NPY_OTYPE, 0);

        if(symmetric){
            for(ddst = 0; ddst < dist_dim; ddst++){

                dist = *((int*) PyArray_GETPTR1(dists, ddst));
                for (dd = 0; dd < dir_dim; dd++){
                    ddx = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 0));
                    ddy = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 1));

                    if(normalized){
                        int npx = ((arr_dimx - MAX(0, ddx)) - MAX(0, -ddx)) * ((arr_dimy - MAX(0, ddy)) - MAX(0, -ddy)) * 2;
                        add = 1.0/npx;
                    }
                    for(dc_src = 0; dc_src < arr_ndch; dc_src++){
                        for (dx = MAX(0, -ddx); dx < (arr_dimx - MAX(0, ddx)); dx++){
                            dpx = (int*) PyArray_GETPTR3(iarr, dx, MAX(0, -ddy), dc_src);
                            dpy = (int*) PyArray_GETPTR3(iarr, dx + ddx, MAX(0, -ddy) + ddy, 0);

                            for (dy = MAX(0, -ddy); dy < (arr_dimy - MAX(0, ddy)); dy++, dpy+=arr_ndch, dpx+=arr_ndch){
                                val = (PTR_OUT*) PyArray_GETPTR5(glcm, ddst, dd, dc_src, *dpx, *dpy);
                                *val += add;

                                val = (PTR_OUT*) PyArray_GETPTR5(glcm, ddst, dd, dc_src, *dpy, *dpx);
                                *val += add;
                            }
                        }
                    }
                }
            }
        }else{
            for(ddst = 0; ddst < dist_dim; ddst++){
                dist = *((int*) PyArray_GETPTR1(dists, ddst));

                for (dd = 0; dd < dir_dim; dd++){
                    ddx = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 0));
                    ddy = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 1));

                    if(normalized){
                        int npx = ((arr_dimx - MAX(0, ddx)) - MAX(0, -ddx)) * ((arr_dimy - MAX(0, ddy)) - MAX(0, -ddy));
                        add = 1.0/npx;
                    }

                    for(dc_src = 0; dc_src < arr_ndch; dc_src++){
                        for (dx = MAX(0, -ddx); dx < (arr_dimx - MAX(0, ddx)); dx++){
                            dpx = (int*) PyArray_GETPTR3(iarr, dx, MAX(0, -ddy), dc_src);
                            dpy = (int*) PyArray_GETPTR3(iarr, dx + ddx, MAX(0, -ddy) + ddy, dc_src);

                            for (dy = MAX(0, -ddy); dy < (arr_dimy - MAX(0, ddy)); dy++, dpx+=arr_ndch, dpy+=arr_ndch){
                                val = (PTR_OUT*) PyArray_GETPTR5(glcm, ddst, dd, dc_src, *dpx, *dpy);
                                *val += add;
                            }
                        }
                    }
                }
            }
        }

        return glcm;
    }
    else{
        PyErr_SetString(PyExc_TypeError,
                "invalid array dimensions");
        return NULL;
    }

    return NULL;
}

template <typename PTR_IN, typename PTR_OUT, int NPY_OTYPE> NPY_NO_EXPORT PyArrayObject*
PyArray_xglcm_sum(PyArrayObject* iarr, PyArrayObject* dirs, PyArrayObject* dists, int bins, int symmetric, int normalized)
{
    PyArrayObject *glcm;
    npy_intp glcm_dims[NPY_MAXDIMS];
    int idx_dist, idx_dir, idx_px, idx_py, arr_dimx, arr_dimy, arr_ndch, dist_dim, dir_dim, arr_chan_cmb, dist, dirx, diry, chan_src, chan_dst, npx;

    PTR_OUT *val, cntrb=1.0;
    PTR_IN *dp_src, *dp_dst;

    arr_dimx = (int) PyArray_DIMS(iarr)[0];
    arr_dimy = (int) PyArray_DIMS(iarr)[1];
    arr_ndch = (int) PyArray_DIMS(iarr)[2];

    dist_dim = (int) PyArray_DIMS(dists)[0];
    dir_dim = (int) PyArray_DIMS(dirs)[1];

    arr_chan_cmb = factorial(arr_ndch) / (2*factorial(arr_ndch-2));

    if(symmetric){

        glcm_dims[0] = dist_dim;
        glcm_dims[1] = arr_chan_cmb;
        glcm_dims[2] = bins;
        glcm_dims[3] = bins;

        glcm = (PyArrayObject*) PyArray_ZEROS(4, glcm_dims, NPY_OTYPE, 0);

        for(idx_dist = 0; idx_dist < dist_dim; idx_dist++){
            for (idx_dir = 0; idx_dir < dir_dim; idx_dir++){
                dirx = *((int*) PyArray_GETPTR3(dirs, idx_dist, idx_dir, 0));
                diry = *((int*) PyArray_GETPTR3(dirs, idx_dist, idx_dir, 1));

                if(normalized){
                    int npx = ((arr_dimx - MAX(0, dirx)) - MAX(0, -dirx)) * ((arr_dimy - MAX(0, diry)) - MAX(0, -diry)) * 2 * dir_dim;
                    cntrb = 1.0/npx;
                }

                int ch_dst = 0;

                for(chan_src = 0; chan_src < arr_ndch; chan_src++){
                    for(chan_dst = chan_src + 1; chan_dst < arr_ndch; chan_dst++){
                        for (idx_px = MAX(0, -dirx); idx_px < (arr_dimx - MAX(0, dirx)); idx_px++){
                            dp_src = (PTR_IN*) PyArray_GETPTR3(iarr, idx_px, MAX(0, -diry), chan_src);
                            dp_dst = (PTR_IN*) PyArray_GETPTR3(iarr, idx_px + dirx, MAX(0, -diry) + diry, chan_dst);

                            for (idx_py = MAX(0, -diry); idx_py < (arr_dimy - MAX(0, diry)); idx_py++, dp_src+=arr_ndch, dp_dst+=arr_ndch){
                                val = (PTR_OUT*) PyArray_GETPTR4(glcm, idx_dist, ch_dst, *dp_src, *dp_dst);
                                *val += cntrb;

                                val = (PTR_OUT*) PyArray_GETPTR4(glcm, idx_dist, ch_dst, *dp_dst, *dp_src);
                                *val += cntrb;
                            }
                        }

                        ch_dst++;
                    }
                }
            }
        }
    }else{

    }

    return glcm;
}

template <typename PTR_OUT, int NPY_OTYPE> NPY_NO_EXPORT PyArrayObject*
PyArray_xglcm_raw(PyArrayObject* iarr, PyArrayObject* dirs, PyArrayObject* dists, int bins, int symmetric, int normalized)
{
    PyArrayObject *glcm;
    npy_intp glcm_dims[NPY_MAXDIMS];
    int dist,dx,dy,dc_dst, dc_src,dd,ddst,ddx,ddy,arr_dimx,arr_dimy, arr_ndch, dir_dim,dist_dim, ids,idt, *dpx, *dpy, *dpc;

    PTR_OUT *val, add=1.0;

    if(PyArray_NDIM(iarr) == 3){
        arr_dimx = (int) PyArray_DIMS(iarr)[0];
        arr_dimy = (int) PyArray_DIMS(iarr)[1];
        arr_ndch = (int) PyArray_DIMS(iarr)[2];
        dist_dim = (int) PyArray_DIMS(dists)[0];
        dir_dim = (int) PyArray_DIMS(dirs)[1];

        glcm_dims[0] = dist_dim;
        glcm_dims[1] = dir_dim;
        glcm_dims[2] = arr_ndch;
        glcm_dims[3] = arr_ndch;
        glcm_dims[4] = bins;
        glcm_dims[5] = bins;

        glcm = (PyArrayObject*) PyArray_ZEROS(6, glcm_dims, NPY_OTYPE, 0);

        if(symmetric){
            for(ddst = 0; ddst < dist_dim; ddst++){

                dist = *((int*) PyArray_GETPTR1(dists, ddst));
                for (dd = 0; dd < dir_dim; dd++){
                    ddx = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 0));
                    ddy = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 1));

                    if(normalized){
                        int npx = ((arr_dimx - MAX(0, ddx)) - MAX(0, -ddx)) * ((arr_dimy - MAX(0, ddy)) - MAX(0, -ddy)) * 2;
                        add = 1.0/npx;
                    }
                    for(dc_src = 0; dc_src < arr_ndch; dc_src++){
                        for (dx = MAX(0, -ddx); dx < (arr_dimx - MAX(0, ddx)); dx++){
                            dpx = (int*) PyArray_GETPTR3(iarr, dx, MAX(0, -ddy), dc_src);
                            dpy = (int*) PyArray_GETPTR3(iarr, dx + ddx, MAX(0, -ddy) + ddy, 0);

                            for (dy = MAX(0, -ddy); dy < (arr_dimy - MAX(0, ddy)); dy++, dpx+=arr_ndch){
                                for(dc_dst = 0; dc_dst < arr_ndch; dc_dst++, dpy++){
                                    val = (PTR_OUT*) PyArray_GETPTR6(glcm, ddst, dd, dc_src, dc_dst, *dpx, *dpy);
                                    *val += add;

                                    val = (PTR_OUT*) PyArray_GETPTR6(glcm, ddst, dd, dc_src, dc_dst, *dpy, *dpx);
                                    *val += add;
                                }
                            }
                        }
                    }
                }
            }
        }else{
            for(ddst = 0; ddst < dist_dim; ddst++){
                dist = *((int*) PyArray_GETPTR1(dists, ddst));

                for (dd = 0; dd < dir_dim; dd++){
                    ddx = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 0));
                    ddy = *((int*) PyArray_GETPTR3(dirs, ddst, dd, 1));

                    if(normalized){
                        int npx = ((arr_dimx - MAX(0, ddx)) - MAX(0, -ddx)) * ((arr_dimy - MAX(0, ddy)) - MAX(0, -ddy));
                        add = 1.0/npx;
                    }
                    for(dc_src = 0; dc_src < arr_ndch; dc_src++){
                        for (dx = MAX(0, -ddx); dx < (arr_dimx - MAX(0, ddx)); dx++){
                            dpx = (int*) PyArray_GETPTR3(iarr, dx, MAX(0, -ddy), dc_src);
                            dpy = (int*) PyArray_GETPTR3(iarr, dx + ddx, MAX(0, -ddy) + ddy, 0);

                            for (dy = MAX(0, -ddy); dy < (arr_dimy - MAX(0, ddy)); dy++, dpx+=arr_ndch){
                                for(dc_dst = 0; dc_dst < arr_ndch; dc_dst++, dpy++){
                                    val = (PTR_OUT*) PyArray_GETPTR6(glcm, ddst, dd, dc_src, dc_dst, *dpx, *dpy);
                                    *val += add;
                                }
                            }
                        }
                    }
                }
            }
        }
        return glcm;
    }
    else{
        PyErr_SetString(PyExc_TypeError,
                "invalid array dimensions");
        return NULL;
    }

    return NULL;
}

/* --- GLCM FEATURES ----------------------------------------------------------------------------------------------- */

NPY_NO_EXPORT PyArrayObject*
PyArray_angular_second_moment(PyArrayObject* glcm)
{
    PyArrayObject *feat;
    npy_intp ndim, *shape;
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    int px,py;
    char **dataptr;
    npy_intp *strideptr, idx[NPY_MAXDIMS] = {0};

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);

    feat = (PyArrayObject*) PyArray_ZEROS((int) (ndim - 2), shape, NPY_DOUBLE, 0);

    iter = NpyIter_New(feat, NPY_ITER_WRITEONLY |
                        NPY_ITER_MULTI_INDEX,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
        return NULL;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);

    NpyIter_GetMultiIndexFunc *get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL); /* I'm a totally documented function! */

    double *src;

    do {
        double *dst = (double *) *dataptr;
        npy_intp stride = *strideptr;

        get_multi_index(iter, idx);
        src = (double*) PyArray_GetPtr(glcm, idx);

        for(px = 0; px < shape[ndim - 2] ; px++){
            for(py = 0; py < shape[ndim - 1]; py++, src++){
                *dst += (*src) * (*src);
            }
        }

    } while(iternext(iter));

    NpyIter_Deallocate(iter);

    return feat;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_contrast(PyArrayObject* glcm)
{
    PyArrayObject *feat;
    npy_intp ndim, *shape;
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    int px,py;
    char **dataptr;
    npy_intp *strideptr, idx[NPY_MAXDIMS] = {0};

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);

    feat = (PyArrayObject*) PyArray_ZEROS((int) (ndim - 2), shape, NPY_DOUBLE, 0);

    iter = NpyIter_New(feat, NPY_ITER_WRITEONLY |
                        NPY_ITER_MULTI_INDEX,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
        return NULL;
    }
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);

    NpyIter_GetMultiIndexFunc *get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL); /* I'm a totally documented function! */

    if(PyArray_ISINTEGER(glcm)){
        int *src;

        do {
            double *dst = (double*) *dataptr;
            npy_intp stride = *strideptr;

            get_multi_index(iter, idx);
            src = (int*)PyArray_GetPtr(glcm, idx);

            for(px = 0; px < shape[ndim - 2] ; px++){
                for(py = 0; py < shape[ndim - 1]; py++, src++){
                     *dst += (px - py)*(px - py)*(*src);
                }
            }

            dst += stride;
        } while(iternext(iter));
    }else{
        double *src;

        do {
            double *dst = (double*) *dataptr;
            npy_intp stride = *strideptr;

            get_multi_index(iter, idx);
            src = (double*) PyArray_GetPtr(glcm, idx);

            for(px = 0; px < shape[ndim - 2] ; px++){
                for(py = 0; py < shape[ndim - 1]; py++, src++){
                     *dst += (px - py)*(px - py) * (*src);
                }
            }

            dst += stride;
        } while(iternext(iter));
    }

    NpyIter_Deallocate(iter);

    return feat;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_inverse_diff_moment(PyArrayObject* glcm, int symmetric)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS((int) (ndim - 2), shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index(ndim - 2, shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);

        if(symmetric){
            double *data = src;

            for(px = 0; px < bins; px++){
                data = &(src[px*bins]);

                for(py = 0; py <= (px - 1); py++, data++){
                    *dst += 2 * ((1./(1 + (px-py) * (px-py))) * (*data));
                }
                *dst += (1./(1 + (px-py) * (px-py))) * (*data);
            }
        }else{
             for(px = 0; px < bins; px++){
                for(py = 0; py < bins; py++, src++){
                    *dst += (1./(1 + (px-py) * (px-py))) * (*src);
                }
             }
        }
    }

    return feature;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_entropy(PyArrayObject* glcm, int symmetric)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS((int) (ndim - 2), shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index(ndim - 2, shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);

        if(symmetric){
            double *data = src;

            for(px = 0; px < bins; px++){
                data = &(src[px*bins]);

                for(py = 0; py <= (px - 1); py++, data++){
                    *dst -= 2 * (*data)*log10(*data + eps);
                }
                *dst -= (*data)*log10(*data + eps);
            }
        }else{
             for(px = 0; px < bins; px++){
                for(py = 0; py < bins; py++, src++){
                    *dst -= (*src)*log10(*src + eps);
                }
             }
        }
    }

    return feature;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_sum_average(PyArrayObject* glcm, PyArrayObject* cprob)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px;
    double *dst, *src, *prb;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS((int) (ndim - 2), shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index(ndim - 2, shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        prb = (double *) PyArray_GetPtr(cprob, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);

        for(px = 0; px < 2*bins - 1; px++){
            *dst += (px+2) * prb[px];
        }
    }

    return feature;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_sum_entropy(PyArrayObject* glcm, PyArrayObject* cprob)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px;
    double *dst, *src, *prb;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS((int) (ndim - 2), shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index((ndim - 2), shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        prb = (double *) PyArray_GetPtr(cprob, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);

        for(px = 0; px < 2*bins - 1; px++){
            *dst += prb[px] * log10(prb[px] + eps);
        }
    }

    return feature;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_sum_var(PyArrayObject* glcm, PyArrayObject* cprob, PyArrayObject* savg)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px;
    double *dst, *src, *prb, *sa;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS((int) (ndim - 2), shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index((int) (ndim - 2), shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        prb = (double *) PyArray_GetPtr(cprob, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);
        sa  = (double *) PyArray_GetPtr(savg, multi_idx);

        for(px = 0; px < 2*bins - 1; px++){
            *dst += (px + 2 - *sa)*(px + 2 - *sa)*prb[px];
        }
    }

    return feature;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_diff_average(PyArrayObject* glcm, PyArrayObject* dprob)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src, *prb;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS(ndim - 2, shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index(ndim - 2, shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        prb = (double *) PyArray_GetPtr(dprob, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);

        for(px = 0; px < bins; px++){
            *dst += px * prb[px];
        }
    }

    return feature;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_diff_entropy(PyArrayObject* glcm, PyArrayObject* dprob)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src, *prb;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS(ndim - 2, shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index(ndim - 2, shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        prb = (double *) PyArray_GetPtr(dprob, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);

        for(px = 0; px < bins; px++){
            *dst += prb[px] * log10(prb[px] + eps);
        }
    }

    return feature;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_autocorrelation(PyArrayObject* glcm, int symmetric)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS(ndim - 2, shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index(ndim - 2, shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);

        if(symmetric){
            double *data = src;

            for(px = 0; px < bins; px++){
                data = &(src[px*bins]);

                for(py = 0; py <= (px - 1); py++, data++){
                    *dst += 2*(*data)*(px+1)*(py+1);
                }
                *dst += (*data)*(px+1)*(py+1);
            }
        }else{
             for(px = 0; px < bins; px++){
                for(py = 0; py < bins; py++, src++){
                    *dst += (*src)*(px+1)*(py+1);
                }
             }
        }
    }

    return feature;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_cluster_prominence(PyArrayObject* glcm, PyArrayObject* javrg, int symmetric)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src, *mean;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS(ndim - 2, shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index(ndim - 2, shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);
        mean = (double *) PyArray_GetPtr(javrg, multi_idx);

        if(symmetric){
            double *data = src;

            for(px = 0; px < bins; px++){
                data = &(src[px*bins]);

                for(py = 0; py <= (px-1); py++, data++){
                    *(dst) += 2 * (px + py + 2 - *(mean + px) - *(mean + py)) * \
                                  (px + py + 2 - *(mean + px) - *(mean + py)) * \
                                  (px + py + 2 - *(mean + px) - *(mean + py)) * \
                                  (px + py + 2 - *(mean + px) - *(mean + py)) * \
                                  (*data);
                }
                *(dst) += (px + py + 2 - *(mean + px) - *(mean + py)) * \
                          (px + py + 2 - *(mean + px) - *(mean + py)) * \
                          (px + py + 2 - *(mean + px) - *(mean + py)) * \
                          (px + py + 2 - *(mean + px) - *(mean + py)) * \
                          (*data);
            }
        }else{
             for(px = 0; px < bins; px++){
                for(py = 0; py < bins; py++, src++){
                    *(dst) += (px + py + 2 - *(mean + px) - *(mean + bins + py)) * \
                              (px + py + 2 - *(mean + px) - *(mean + bins + py)) * \
                              (px + py + 2 - *(mean + px) - *(mean + bins + py)) * \
                              (px + py + 2 - *(mean + px) - *(mean + bins + py)) * \
                              (*src);
                }
             }
        }
    }

    return feature;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_cluster_shade(PyArrayObject* glcm, PyArrayObject* javrg, int symmetric)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src, *mean;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS(ndim - 2, shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index(ndim - 2, shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);
        mean = (double *) PyArray_GetPtr(javrg, multi_idx);

        if(symmetric){
            double *data = src;

            for(px = 0; px < bins; px++){
                data = &(src[px*bins]);

                for(py = 0; py <= (px-1); py++, data++){
                    *(dst) += 2 * (px + py + 2 - *(mean + px) - *(mean + py)) * \
                                  (px + py + 2 - *(mean + px) - *(mean + py)) * \
                                  (px + py + 2 - *(mean + px) - *(mean + py)) * \
                                  (*data);
                }
                *(dst) += (px + py + 2 - *(mean + px) - *(mean + py)) * \
                          (px + py + 2 - *(mean + px) - *(mean + py)) * \
                          (px + py + 2 - *(mean + px) - *(mean + py)) * \
                          (*data);
            }
        }else{
             for(px = 0; px < bins; px++){
                for(py = 0; py < bins; py++, src++){
                    *(dst) += (px + py + 2 - *(mean + px) - *(mean + bins + py)) * \
                              (px + py + 2 - *(mean + px) - *(mean + bins + py)) * \
                              (px + py + 2 - *(mean + px) - *(mean + bins + py)) * \
                              (*src);
                }
             }
        }
    }

    return feature;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_cluster_tendency(PyArrayObject* glcm, PyArrayObject* javrg, int symmetric)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src, *mean;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS(ndim - 2, shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index(ndim - 2, shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);
        mean = (double *) PyArray_GetPtr(javrg, multi_idx);

        if(symmetric){
            double *data = src;

            for(px = 0; px < bins; px++){
                data = &(src[px*bins]);

                for(py = 0; py <= (px-1); py++, data++){
                    *(dst) += 2 * (px + py + 2 - *(mean + px) - *(mean + py)) * \
                                  (px + py + 2 - *(mean + px) - *(mean + py)) * \
                                  (*data);
                }
                *(dst) += (px + py + 2 - *(mean + px) - *(mean + py)) * \
                          (px + py + 2 - *(mean + px) - *(mean + py)) * \
                          (*data);
            }
        }else{
             for(px = 0; px < bins; px++){
                for(py = 0; py < bins; py++, src++){
                    *(dst) += (px + py + 2 - *(mean + px) - *(mean + bins + py)) * \
                              (px + py + 2 - *(mean + px) - *(mean + bins + py)) * \
                              (*src);
                }
             }
        }
    }

    return feature;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_dissimilarity(PyArrayObject* glcm, int symmetric)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS(ndim - 2, shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index(ndim - 2, shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);

        if(symmetric){
            double *data = src;

            for(px = 0; px < bins; px++){
                data = &(src[px*bins]);

                for(py = 0; py <= (px - 1); py++, data++){
                    *dst += 2*(*data)*(px -py);
                }
                *dst += (*data)*(px - py);
            }
        }else{
             for(px = 0; px < bins; px++){
                for(py = 0; py < px; py++, src++){
                    *dst += (*src)*(px - py);
                }
                for(py = px; py < bins; py++, src++){
                    *dst += (*src)*(py - px);
                }
             }
        }
    }

    return feature;
}

NPY_NO_EXPORT PyArrayObject*
PyArray_inverse_diff(PyArrayObject* glcm, int symmetric)
{
    PyArrayObject *feature;
    npy_intp idx, max_idx, bins, ndim, *shape, *multi_idx, px, py;
    double *dst, *src;

    ndim = PyArray_NDIM(glcm);
    shape = PyArray_SHAPE(glcm);
    bins = shape[ndim -1];

    feature = (PyArrayObject*) PyArray_ZEROS(ndim - 2, shape, NPY_DOUBLE, 0);

    max_idx = PyArray_get_max_idx(ndim - 2, shape);
    for(idx = 0; idx < max_idx; idx++){
        multi_idx = PyArray_get_multi_index(ndim - 2, shape, idx);

        dst = (double *) PyArray_GetPtr(feature, multi_idx);
        src = (double *) PyArray_GetPtr(glcm, multi_idx);

        if(symmetric){
            double *data = src;

            for(px = 0; px < bins; px++){
                data = &(src[px*bins]);

                for(py = 0; py <= (px - 1); py++, data++){
                    *dst += 2*(*data)/(1. + (px - py));
                }
                *dst += (*data)/(1. + (px - py));
            }
        }else{
             for(px = 0; px < bins; px++){
                for(py = 0; py < px; py++, src++){
                    *dst += (*src)/(1. + (px - py));
                }
                for(py = px; py < bins; py++, src++){
                    *dst += (*src)/(1. + (py - px));
                }
             }
        }
    }

    return feature;
}

/* --- PYTHON FUNCTIONS -------------------------------------------------------------------------------------------- */

static PyObject*
array_bin(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    static  char *kwlist[] = {"array", "bins", NULL};
    PyObject *array;
    PyArrayObject *iarr, *oarr;
    npy_intp bins;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oi", kwlist,
                &array, &bins)) {
        return NULL;
    }

    iarr = (PyArrayObject*) PyArray_FROM_OTF(array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY); /* TODO: check for int array */
    if(iarr == NULL) return NULL;

    oarr = PyArray_bin(iarr, bins);

    //Py_DECREF(iarr);

    return  (PyObject*) oarr;
}

static PyObject*
array_glcm(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    static  char *kwlist[] = {"array", "dists", "dirs", "mode", "symmetric", "bins", "normalized", "check", NULL};
    PyObject *array, *directions, *distances;
    PyArrayObject *iarr, *dirs, *dists, *glcm;
    int check=1, bins=256, symmetric=1, normalized=1;
    char *mode;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOs|iiii", kwlist,
                &array, &distances, &directions, &mode, &symmetric, &bins, &normalized, &check)) {
        return NULL;
    }

    iarr = (PyArrayObject*)PyArray_FROM_OF(array, NPY_ARRAY_IN_ARRAY);
    if (iarr == NULL) return NULL;

    dirs = (PyArrayObject*)PyArray_FROM_OTF(directions, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (dirs == NULL) return NULL;

    dists = (PyArrayObject*)PyArray_FROM_OTF(distances, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (dists == NULL) return NULL;

    if(check){
        if(!PyArray_ISINTEGER(iarr)){
            PyErr_WarnEx(NULL, "input array is not of integer type and is therefore binned. This results in performance loss", 1);
            iarr = PyArray_bin(iarr, bins);
        }
        else {
            //iarr = PyArray_bin(iarr, bins); /* Todo: do this condionally */
        }
    }
    dirs = PyArray_glcm_gen_dirs(dirs, dists, symmetric);

    if(strcmp(mode, "sum") == 0){
        if(normalized){
            glcm = PyArray_glcm_sum<double, NPY_DOUBLE>(iarr, dirs, dists, bins, symmetric, normalized);
        }else{
            glcm = PyArray_glcm_sum<int, NPY_INT>(iarr, dirs, dists, bins, symmetric, normalized);
        }
    }else{
        if(normalized){
            glcm = PyArray_glcm_raw<double, NPY_DOUBLE>(iarr, dirs, dists, bins, symmetric, normalized);
        }else{
            glcm = PyArray_glcm_raw<int, NPY_INT>(iarr, dirs, dists, bins, symmetric, normalized);
        }
    }

    Py_DECREF(dirs);

    return (PyObject *) glcm;
}

static PyObject*
array_xglcm(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    static  char *kwlist[] = {"array", "dists", "dirs", "mode", "symmetric", "bins", "normalized", "check", NULL};
    PyObject *array, *directions, *distances;
    PyArrayObject *iarr, *dirs, *dists, *glcm;
    int check=1, bins=256, symmetric=1, normalized=1;
    char *mode;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOs|iiii", kwlist,
                &array, &distances, &directions, &mode, &symmetric, &bins, &normalized, &check)) {
        return NULL;
    }

    iarr = (PyArrayObject*)PyArray_FROM_OF(array, NPY_ARRAY_IN_ARRAY);
    if (iarr == NULL) return NULL;

    dirs = (PyArrayObject*)PyArray_FROM_OTF(directions, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (dirs == NULL) return NULL;

    dists = (PyArrayObject*)PyArray_FROM_OTF(distances, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (dists == NULL) return NULL;

    /* Todo: check for only 3D images */
    if(check){
        if(!PyArray_ISINTEGER(iarr)){
            PyErr_WarnEx(NULL, "input array is not of integer type and is therefore binned. This results in performance loss", 1);
            iarr = PyArray_bin(iarr, bins);
        }
        else {
            //iarr = PyArray_bin(iarr, bins); /* do this condionally */
        }
    }
    dirs = PyArray_glcm_gen_dirs(dirs, dists, symmetric);

    if(strcmp(mode, "sum") == 0){
        glcm = PyArray_xglcm_sum<int, double, NPY_DOUBLE>(iarr, dirs, dists, bins, symmetric, normalized);
    }else{
        if(normalized){
            glcm = PyArray_xglcm_raw<double, NPY_DOUBLE>(iarr, dirs, dists, bins, symmetric, normalized);
        }else{
            glcm = PyArray_xglcm_raw<int, NPY_INT>(iarr, dirs, dists, bins, symmetric, normalized);
        }
    }

    Py_DECREF(dirs);

    return (PyObject *) glcm;
}

static PyObject*
array_glcm_features(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    static  char *kwlist[] = {"array", "features", "normalized", "symmetric", NULL};
    PyObject *array, *dict;
    PyArrayObject *iarr, *oarr;
    int features, normalized=1, symmetric=1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oi|ii", kwlist,
                &array, &features, &normalized, &symmetric)) {
        return NULL;
    }

    iarr = (PyArrayObject*)PyArray_FROM_OTF(array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (iarr == NULL) return NULL;

    dict = PyDict_New();

    if(features & ASM)      PyDict_SetItemString(dict, "ASM", \
                                                        (PyObject*) PyArray_angular_second_moment(iarr));
    if(features & CONTRAST) PyDict_SetItemString(dict, "Contrast", \
                                                        (PyObject*) PyArray_contrast(iarr));
    if(features & INVDIFFM) PyDict_SetItemString(dict, "IDM", \
                                                        (PyObject*) PyArray_inverse_diff_moment(iarr, symmetric));
    if(features & INVDIFF)  PyDict_SetItemString(dict, "IDF", \
                                                        (PyObject*) PyArray_inverse_diff(iarr, symmetric));
    if(features & ENTROPY)  PyDict_SetItemString(dict, "Entropy", \
                                                        (PyObject*) PyArray_entropy(iarr, symmetric));
    if(features & AUTOCORR) PyDict_SetItemString(dict, "Autocorrelation", \
                                                        (PyObject*) PyArray_autocorrelation(iarr, symmetric));
    if(features & DISSIM)   PyDict_SetItemString(dict, "Dissimilarity", \
                                                        (PyObject*) PyArray_dissimilarity(iarr, symmetric));


    if(features & (SUMAVG | SUMVAR | SUMENTRP)){
        PyArrayObject *cprob, *savg=NULL;

        cprob = PyArray_comb_prob(iarr, symmetric, "sum");
        if(features & SUMAVG){
            savg = PyArray_sum_average(iarr, cprob);
            PyDict_SetItemString(dict, "Sum Average", (PyObject*) savg);
        }
        if(features & SUMVAR){
            if(!savg) savg = PyArray_sum_average(iarr, cprob);
            PyDict_SetItemString(dict, "Sum Variance", (PyObject*) PyArray_sum_var(iarr, cprob, savg));
        }
        if(features & SUMENTRP) PyDict_SetItemString(dict, "Sum Entropy", \
                                                            (PyObject*) PyArray_sum_entropy(iarr, cprob));
    }

    if(features & (DIFFAVG | DIFFVAR | DIFFENTRP)){
        PyArrayObject *dprob, *savg=NULL;

        dprob = PyArray_comb_diff(iarr, symmetric);

        if(features & DIFFAVG){
            PyDict_SetItemString(dict, "Diff Average", \
                                                        (PyObject*) PyArray_diff_average(iarr, dprob));
        }

        if(features & DIFFENTRP) PyDict_SetItemString(dict, "Diff Entropy", \
                                                            (PyObject*) PyArray_diff_entropy(iarr, dprob));
    }

    if(features & (CLSTRPROM | CLSTRSHAD | CLSTRTEND)){
        PyArrayObject *javrg;

        javrg = PyArray_glcm_mean(iarr, symmetric);
        if(features & CLSTRPROM) PyDict_SetItemString(dict, "Cluster Prominence", \
                                                        (PyObject*) PyArray_cluster_prominence(iarr, javrg, symmetric));
        if(features & CLSTRSHAD) PyDict_SetItemString(dict, "Cluster Shade", \
                                                        (PyObject*) PyArray_cluster_shade(iarr, javrg, symmetric));
        if(features & CLSTRTEND) PyDict_SetItemString(dict, "Cluster Tendency", \
                                                        (PyObject*) PyArray_cluster_tendency(iarr, javrg, symmetric));
    }

    return dict;
}

static PyObject*
array_pad(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    PyObject *arg1;
    PyArrayObject *arr1=NULL, *oarr=NULL;
    int width;

    if (!PyArg_ParseTuple(args, "Oi", &arg1, &width)) return NULL;

    arr1 = (PyArrayObject*)PyArray_FROM_OF(arg1, NPY_ARRAY_IN_ARRAY);
    if (arr1 == NULL) return NULL;

    oarr = PyArray_pad(arr1, width);

    Py_DECREF(arr1);

    return (PyObject*) oarr;
}

/* --- MODULE ------------------------------------------------------------------------------------------------------ */

static PyMethodDef glcm_module_methods[] = {
    {"bin",
        (PyCFunction)array_bin,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"glcm",
        (PyCFunction)array_glcm,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"xglcm",
        (PyCFunction)array_xglcm,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"glcm_features",
        (PyCFunction)array_glcm_features,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"pad",
        (PyCFunction)array_pad,
        METH_VARARGS, NULL},
	{ NULL, NULL, 0, NULL }   /* sentinel */
};

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"glcm",
	NULL,
	-1,
	glcm_module_methods
};

PyMODINIT_FUNC
PyInit_glcm(void)
{
    PyObject* module;

    import_array();

	module = PyModule_Create(&moduledef);

	PyModule_AddIntConstant(module, "asm",          ASM);
	PyModule_AddIntConstant(module, "contrast",     CONTRAST);
	PyModule_AddIntConstant(module, "correl",       CORRELATION);
	PyModule_AddIntConstant(module, "autocorrel",   AUTOCORR);
	PyModule_AddIntConstant(module, "ssq",          SSQ);
	PyModule_AddIntConstant(module, "sumavg",       SUMAVG);
	PyModule_AddIntConstant(module, "sumvar",       SUMVAR);
	PyModule_AddIntConstant(module, "sumentrp",     SUMENTRP);
	PyModule_AddIntConstant(module, "diffavg",      DIFFAVG);
	PyModule_AddIntConstant(module, "diffvar",      DIFFVAR);
	PyModule_AddIntConstant(module, "diffentrp",    DIFFENTRP);
	PyModule_AddIntConstant(module, "clusterprom",  CLSTRPROM);
	PyModule_AddIntConstant(module, "clustershade", CLSTRSHAD);
    PyModule_AddIntConstant(module, "clustertend",  CLSTRTEND);
    PyModule_AddIntConstant(module, "dissim",       DISSIM);
    PyModule_AddIntConstant(module, "idm",          INVDIFFM);
    PyModule_AddIntConstant(module, "idf",          INVDIFF);

	return module;
};