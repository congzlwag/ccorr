# cython: boundscheck=False, wraparound=False, cdivision=True
#cython: language_level=3
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

cimport cython
import numpy as np
cimport numpy as cnp
from cython.parallel import prange, parallel
from libc.math cimport sqrt, isnan
from libc.stdlib cimport abort, malloc, free


ctypedef fused wdtype:
    const unsigned char
    const float
    const double
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cCorrNormedW(const double[:,:] im, const double[:] temp_vec, const double vartemp,
                 const int[:] cols, const int[:] rows,
                 wdtype[:] mask, int max_disp=30, int cx=0, int cy=0):
    """
    Calculate cross-correlation between image and template, in a limited ROI of displacement k. This is the bad-pixel-tolerant version of cv.TM_CCOEFF_NORMED [See: https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dac6677e2af5e0fae82cc5339bfaef5038], and in order to be able to handle NAN on the bad pixels, I have to abandon FFT and do the convolution in the image space.
    
    Parameters:
    np.ndarray[double, ndim=2] im: the image, must be 2D
    np.ndarray[double, ndim=1] temp_vec: the vectorized template. For efficiency, it's required to be (a) vectorized into 1D using the sparse_coo mask represented with cols and rows, and (b) offset such that the (potentially weighted) average is 0
    double vartemp: the variance of temp_vec
    np.ndarray[int, ndim=1] cols, rows: the coordinates of the pixels of interest on the template image. `temp_vec = temp[cols, rows]`
    np.ndarray[float|double, ndim=1] mask: the weights for each pixel
    int max_disp: the maximal displacement in either dimension
    
    Return:
    a (2max_disp+1, 2max_disp+1) matrix, containing the cross-correlation for each displacement vector under consideration
    """
    cdef Py_ssize_t i, j, v
    cdef Py_ssize_t M = max_disp
    cdef Py_ssize_t Nv = cols.size
    cdef double prodmean, std, locmean
    cdef int dx, dy
    cdef const double* tempv_ptr = &temp_vec[0]
    cdef int xc = cx
    cdef int yc = cy
    
    dtype = np.float64
    out = np.zeros((2*M+1, 2*M+1), dtype=dtype)
    # The center of this (2M+1)x(2M+1) grid is xc,yc
    # imv = np.zeros(Nv, dtype=dtype);

    cdef double[:,:] out_view = out
    # cdef double[:] imv_view
    cdef double* local_imv_dptr
    
    with nogil, parallel():
        for i in prange(2*M+1, schedule='guided'):
            dy = i-M+yc
            local_imv_dptr = <double *> malloc(sizeof(double) * Nv)
            if local_imv_dptr is NULL:
                abort()
            for j in range(2*M+1):
                dx = j-M+xc
                for v in range(Nv):
                    local_imv_dptr[v] = im[rows[v]+dy, cols[v]+dx]
                locmean = wnanmean_loc(local_imv_dptr, mask)
                for v in range(Nv):
                    local_imv_dptr[v] -= locmean
                prodmean = wnanmeanprod(local_imv_dptr, tempv_ptr, mask)
                std = wnanmeanprod(local_imv_dptr, local_imv_dptr, mask)
                std = sqrt(vartemp*std)
                out_view[i,j] = prodmean / std
            free(local_imv_dptr)
    return out

# def wnanmean(np.ndarray[np.float64_t, ndim=1] arr, np.ndarray[np.float32_t, ndim=1] weights) -> double:
#     msk = np.isfinite(arr)
#     return np.average(arr[msk], weights=weights[msk])
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double wnanmean(const double[:] arr, wdtype[:] weights) nogil:
    """Calculate the weighted mean, with NAN ignored"""
    cdef Py_ssize_t n = weights.shape[0]
    cdef Py_ssize_t i
    cdef double total = 0.0
    cdef double wtotal = 0.0

    for i in range(n):
        if (not isnan(arr[i])) and weights[i]:
            total += arr[i] * weights[i]
            wtotal += weights[i]

    return total / wtotal

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double wnanmean_loc(const double* arr, wdtype[:] weights) nogil:
    cdef Py_ssize_t n = weights.shape[0]
    cdef Py_ssize_t i
    cdef double total = 0.0
    cdef double wtotal = 0.0

    for i in range(n):
        if (not isnan(arr[i])) and weights[i]:
            total += arr[i] * weights[i]
            wtotal += weights[i]

    return total / wtotal

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double wnanmeanprod(const double* arr1, const double* arr2, wdtype[:] weights) nogil:
    cdef Py_ssize_t n = weights.shape[0]
    cdef Py_ssize_t i
    cdef double total = 0.0
    cdef double wtotal = 0.0

    for i in range(n):
        if (not isnan(arr1[i])) and (not isnan(arr2[i])) and weights[i]:
            total += arr1[i] * arr2[i] * weights[i]
            wtotal += weights[i]

    return total / wtotal

def quadfit(cmat, rk=5, sigma_f=6, full_return=False):
    """
    quadfit(cmat, rk=5, sigma_f=6, full_return=False)
    2D quadratic fitting of a matrix
    
    The fitting model is 
    C(k) = c0 + c1 . (k-kM) + 1/2 (k-kM) . H . (k-kM)
    where c0, c1, and H are the fitting parameters
    and kM is the grid point that maximizes cmat, not a fitting parameter
    
    Parameters:
    cmat: the 2D array to be fitted
    rk:   the median position of the top-rk grid points of cmat will be chosen as the kM, i.e. MaxOnGrid
    sigma_f: the gaussian sigma for weighting the matrix entries during the fitting
    full_return: whether to return the intercept c0 and the fitted matrix or not
    
    Return:
    A dictionary containing the following key-value pairs
    MaxOnGrid: kM
    k*: the sub-pixel maximum of C(k)
    H:  the hessian H
    err: the root-mean-squared error of the fitting, note that this mean is weighted by the same weights in the fitting
    if full_return, additionally:
        intercept: c0
        fitted_mat: the fitted matrix
    """
    jM = np.unravel_index(np.argpartition(-cmat.ravel(), rk)[:rk], 
                          cmat.shape)[::-1]
    jM = np.median(np.asarray(jM), axis=1)
    xx = np.arange(cmat.shape[1]) - jM[0]
    yy = np.arange(cmat.shape[0]) - jM[1]
    dM = cmat.shape[0] // 2
    xx, yy = np.meshgrid(xx, yy)
    xx2,yy2 = xx**2, yy**2
    xxyy = 2*xx*yy
    w = np.exp(-0.5*(xx2+yy2)/(sigma_f**2)).ravel()
    # return w
    A = np.stack([xx, yy, xx2, xxyy, yy2])
    A.shape = (A.shape[0],-1)
    Amean = np.average(A, axis=1,weights=w)
    A -= Amean[:, None]
    B = cmat.ravel()
    # print((B @ w) / (w.sum()))
    Bmean = np.average(B, weights=w)
    B = B - Bmean
    AtA, AtB = (A * w) @ A.T, (A * w) @ B
    slop = np.linalg.solve(AtA, AtB)
    intcp = Bmean - slop @ Amean
    err = slop @ AtA @ slop - 2 * slop @ AtB + (B*w) @ B
    err /= w.sum()
    err = err**0.5
    g = slop[:2]
    H = np.array([[slop[2],slop[3]],[slop[3],slop[4]]])*2
    dr = -np.linalg.solve(H,g)
    Bfit = slop @ A + intcp
    out = {"MaxOnGrid":jM-dM, "k*": jM+dr-dM,
           "H":H, "err":err}
    if not full_return:
        return out
    out["intercept"] = intcp
    out["fitted_mat"] = Bfit.reshape(cmat.shape)
    return out

def calc_ellipse_scale2(kxys_, hessians_, errs_, center_xy=0):
    """
    calc_ellipse_scale2(kxys_, hessians_, errs_, center_xy=0)
    Calculate the s^2 factor for the uncertainty ellipse to cover the origin.
    This function is vectorized over axis0
    
    s^2 = k . (-H) . k / err
    
    Parameters:
    kxys_: (N,2) array, the k* of N shots
    hessians_: (N,2,2) array, the Hessian matrices of N shots
    errs_: (N,) array, the fitting RMSE of N shots
    
    Return:
    s^2 : the scale factors to cover the origin. For those negative, it means this estimate fails on the corresponding shots.
    """
    tmp = kxys_ - center_xy
    out = np.asarray([-(d @ h @ d) for d,h in zip(tmp, hessians_)])
    out /= errs_
    return out