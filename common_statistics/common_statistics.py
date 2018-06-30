import numpy as np
import scipy
import scipy.stats
import copy
import multiprocessing as mp
from itertools import repeat
from .config import PARALLEL


def snr(noise_cov_matrices, signal_matrix):
    """ Function for calculating signal to
    noise ratio as defined in paper
    """
    dim = signal_matrix.shape[0]
    varu = np.trace(signal_matrix, offset=0, axis1=0, axis2=1, dtype=None, out=None)
    vnum = noise_cov_matrices.shape[0]
    identity_matr = np.eye(dim, M=None, k=0, dtype='bool')
    identity_matr = identity_matr[np.newaxis, :, :]
    idxs = np.tile(identity_matr, (vnum, 1, 1))
    varn = sum(noise_cov_matrices[idxs])/vnum
    return varu/varn


def _calc_smallest_eig_val(cov):
    w, v = scipy.linalg.eigh(cov)
    return w[0]


def _quadr_form(list):
    # list[0] is a covariance
    # list[1] is a vector
    cov = list[0]
    q = list[1]
    return q.dot(cov.dot(q))


def _calc_const(covars):
    norms = []
    for l in range(len(covars)):
        norms.append(np.linalg.norm(covars[l], ord=2,
                                    axis=None, keepdims=False))
    return(np.max(norms)+1)


def _est_fixed_index(args):
    Slist = args[0]
    cov = args[1]
    c = args[2]
    # S - matrix columns are vectors
    # cov - covariance matrix
    if not Slist:
        w, v = scipy.linalg.eigh(cov)
        val = w[0]
        vec = v[:, 0]
    else:
        S = np.column_stack(Slist)
        Qs = np.dot(S, S.T)
        d = cov.shape[0]
        Matr = np.dot(np.dot(np.identity(d) - Qs, c*np.identity(d)-cov),
                      np.identity(d) - Qs)
        _, v = scipy.linalg.eigh(Matr)
        val = ((v[:, -1]).dot(cov)).dot(v[:, -1])
        vec = v[:, -1]
    return [val, vec]


def est_common_cov(covars, outliers=0):
    c = _calc_const(covars)
    covars = copy.deepcopy(covars)
    d = covars[0].shape[0]
    m = len(covars)
    S = []
    lams = []
    vals = []
    vecs = []
    # drop_idxs = []
    for i in range(0, d):
        if PARALLEL:
            with mp.Pool(15) as pool:
                output = pool.map(_est_fixed_index, zip(repeat(S), covars, repeat(c)))
        else:
            output = [_est_fixed_index(i) for i in zip(repeat(S), covars, repeat(c))]
        for j in range(m):  # -i*outliers):
            vals.append((output[j])[0])
            vecs.append((output[j])[1])
        partition_idxs = np.argpartition(vals, kth=outliers, axis=0, kind='introselect', order=None)
        idx = partition_idxs[outliers]
        # drop_idxs = copy.deepcopy(partition_idxs[0: outliers])
        # covars = np.delete(covars, drop_idxs, axis=0)
        S.append(vecs[idx])
        lams.append(vals[idx])
        vals = []
        vecs = []
    UniTaryMatrix = np.column_stack(S)
    LamBdaMatrX = np.diag(np.array(lams))
    return (UniTaryMatrix).dot(LamBdaMatrX.dot(UniTaryMatrix.transpose()))


def est_common_density2D(data, bw_method=0.3, outliers=0,
                         dimx=100, xmin=-3, xmax=3,
                         dimy=100, ymin=-3, ymax=3):
    transforms = []
    pdfs_of_subjects = []
    # -------KDE------#
    x1, x2 = np.meshgrid(np.linspace(xmin, xmax, num=dimx,
                                     endpoint=True, retstep=False,
                                     dtype=None),
                         np.linspace(ymin, ymax, num=dimy,
                                     endpoint=True))
    positions = np.vstack([x1.ravel(), x2.ravel()])
    vnum = len(data)
    for j in range(0, vnum):
        values = data[j]
        kernel = scipy.stats.gaussian_kde(values, bw_method=bw_method)
        z = np.reshape(kernel.evaluate(positions).T, x1.shape)
        Z = np.fft.fft2(z, s=None, axes=(-2, -1), norm=None)
        transforms.append(Z.flatten())
        pdfs_of_subjects.append(z)
    Zresult = np.zeros(dimx*dimy, dtype=np.complex)
    Zmatr = np.vstack(transforms)
    # --outliers--#
    # rows = np.abs(Zmatr).argmax(axis=0)
    rows = np.argpartition(np.abs(Zmatr), kth=vnum-outliers-1,
                           axis=0, kind='introselect',
                           order=None)[vnum-outliers-1]
    for j in range(0, dimx*dimy):
        Zresult[j] = Zmatr[rows[j], j]
    ZresMatr = np.reshape(Zresult, x1.shape)
    pUMatr = np.fft.ifft2(ZresMatr, s=None, axes=(-2, -1), norm=None)
    # -set non positive values to zero
    pUMatr = np.maximum(0, np.real(pUMatr))
    return pUMatr, pdfs_of_subjects, positions, xmin, xmax, ymin, ymax, dimx, dimy
