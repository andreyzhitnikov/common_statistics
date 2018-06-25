
import matplotlib.pyplot as plt
import sys
from pyriemann.utils.mean import mean_covariance, mean_euclid
from pyriemann.utils.distance import distance_euclid
from common_statistics import snr, est_common_cov, est_common_density2D
from numpy import linalg as LA
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import argparse
from nilearn.connectome import connectivity_matrices
import os
from scipy.stats import expon
from rtnorm import rtnorm
import numpy as np
import random
import math

plt.ion()

np.random.seed(seed=555)
parser = argparse.ArgumentParser()
parser.add_argument('--dim', required=False, default=3, type=int,
                    help='dimension of the simulation')
parser.add_argument('--overlap_eigenvalues', required=False, action="store_true",
                    help='if true the all eigenvalues are drawn from [0, b],\
                    else eigenvalue i drawn from interval [b*(i-1),b*i], for i=1,...')
parser.add_argument('--vnum', required=False, default=1000, type=int, help='number of subjects')
parser.add_argument('--visual', default=False, action="store_true", help='visual')
parser.add_argument('--show_corrs', default=False, action="store_true", help='show correlations of the subjects')
parser.add_argument('--repeat-one', default=0, type=int, help='repeat constant one matrix ')
parser.add_argument('--outliers', default=0, type=int, help='number of outliers')
parser.add_argument('--b', default=1, type=int, help='first eignevalue is drawn in the interval [0, b]')
args = parser.parse_args()


def _draw_eigenvalues(dim=3, b=10, distribution="unif", overlap=True):
        lambdas = []
        if(distribution == "unif"):
                for i in range(0, dim):
                        if overlap:
                                currlam = b * random.random()
                        else:
                                currlam = b * random.random() + i*b
                        lambdas.append(currlam)
                LambdaMatrix = np.diag(lambdas)
        elif (distribution == "exp"):
                LambdaMatrix = np.diag(expon.rvs(loc=0, scale=1, size=dim, random_state=None))
        return LambdaMatrix


def _draw_rotation_matrix(dim=3):
        if (dim == 3):
                phi = rtnorm(0, 2 * np.pi, mu=1, sigma=1, size=1, probabilities=False)
                theta = rtnorm(0, np.pi, mu=1, sigma=1, size=1, probabilities=False)
                u1 = np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])
                u2 = np.array([math.cos(theta)*math.cos(phi), math.cos(theta)*math.sin(phi), -1*math.sin(theta)])
                u3 = np.array([-1*math.sin(phi), math.cos(phi), 0])
                rotMatrix = np.column_stack((u1, u2, u3))
        else:
                # a = np.random.normal(loc=0.0, scale=1.0, size=(dim, dim))
                a = 100*np.random.rand(dim, dim)-50
                q, r = np.linalg.qr(a, mode='complete')
                r = np.diag(np.sign(np.diag(r)))
                rotMatrix = np.dot(q, r)
                rotMatrix = q
        return rotMatrix


def _draw_subject_specific_and_estimate(cov_u, dim, vnum,
                                        outliers_num, b, show_corrs=False,
                                        repeat_one=0, label=None):
        covars = []
        noise_covars = []
        for i in range(0, vnum-outliers_num):
            # ---construct covariance of subject#
            LambdaMatrix = _draw_eigenvalues(dim=args.dim, b=args.b, distribution="unif", overlap=args.overlap_eigenvalues)
            rotMatrix = _draw_rotation_matrix(dim=3)
            noise_cov = rotMatrix.dot(LambdaMatrix.dot(rotMatrix.transpose()))
            cov = np.add(cov_u, noise_cov)
            noise_covars.append(noise_cov)
            covars.append(cov)

        for i in range(repeat_one):
                covars.append(cov)
        # --------------------------------------#
        print("snr is :" + str(snr(np.asarray(noise_covars), cov_u)))
        if(show_corrs):
                factor = 1/float(b)
                f, axarr = plt.subplots(3, 3, figsize=(11, 5))
                idxs = np.random.choice(range(len(covars)), size=(3, 3))
                for i in range(idxs.shape[0]):
                        for j in range(idxs.shape[1]):
                                im = axarr[i, j].pcolormesh(np.flipud((connectivity_matrices.cov_to_corr(covars[idxs[i, j]])).T),
                                                            cmap=plt.cm.gist_earth_r,
                                                            edgecolors='k', vmin=-factor*b, vmax=factor*b, **{'linewidth': 0.1})
                                axarr[i, j].set_axis_off()
                                axarr[i, j].set_aspect('equal', 'box')
                                if(j == 2):
                                        divider = make_axes_locatable(axarr[i, j])
                                        cax = divider.append_axes("right", size="5%", pad=0.05)
                                        cb = f.colorbar(im, cax=cax)
                                        tick_locations = (-factor*b, 0, factor*b)
                                        tick_labels = ('{:{width}.{prec}f}'.format(-factor*b, width=1, prec=1),
                                                       '{:{width}.{prec}f}'.format(0, width=1, prec=1),
                                                       '{:{width}.{prec}f}'.format(factor*b, width=1, prec=1))
                                        cb.locator = ticker.FixedLocator(tick_locations)
                                        cb.formatter = ticker.FixedFormatter(tick_labels)
                                        cb.update_ticks()
                if label is None:
                        plt.savefig('./subjects_covars.eps', format='eps', dpi=1000)
                else:
                        plt.savefig('./subjects_covars_' + str(label) + '.eps', format='eps', dpi=1000)
        # construct outliers
        w, v = LA.eig(cov_u)
        for i in range(0, outliers_num):
                LambdaMatrix = _draw_eigenvalues(dim=args.dim, b=args.b, distribution="unif", overlap=args.overlap_eigenvalues)
                rotMatrix = _draw_rotation_matrix(dim=3)
                cov = rotMatrix.dot(LambdaMatrix.dot(rotMatrix.transpose()))
                covars.append(cov)
        our_est = est_common_cov(covars, outliers=args.outliers)
        eucl_mean = mean_euclid(np.array(covars))
        riemann_mean = mean_covariance(np.array(covars), metric='riemann', sample_weight=None)
        return our_est, eucl_mean, riemann_mean


cov_u1 = np.zeros((args.dim, args.dim))
cov_u2 = np.identity(args.dim)
LambdaMatrix = _draw_eigenvalues(dim=args.dim, b=args.b, distribution="unif", overlap=args.overlap_eigenvalues)
rotMatrix = _draw_rotation_matrix(dim=3)
cov_u3 = rotMatrix.dot(LambdaMatrix.dot(rotMatrix.transpose()))
LambdaMatrix = _draw_eigenvalues(dim=args.dim, b=args.b, distribution="unif", overlap=args.overlap_eigenvalues)
rotMatrix = _draw_rotation_matrix(dim=3)
cov_u4 = rotMatrix.dot(LambdaMatrix.dot(rotMatrix.transpose()))

covs_u = [cov_u1, cov_u2, cov_u3, cov_u4]
est_covsu = []
eucl_mean = []
riemann_mean = []
for i, cu in enumerate(covs_u):
    our, eucl, riemann = _draw_subject_specific_and_estimate(cu, args.dim,
                                                             args.vnum, args.outliers,
                                                             args.b, args.show_corrs,
                                                             args.repeat_one, i)
    est_covsu.append(our)
    print("Estimation" + str(i) + " : Euclidean distance between our estimator and ground thruth is : " + str(distance_euclid(our, cu)))
    eucl_mean.append(eucl)
    print("Estimation" + str(i) + " : Euclidean distance between euclidean mean and ground thruth is : " + str(distance_euclid(eucl, cu)))
    riemann_mean.append(riemann)
    print("Estimation" + str(i) + " : Euclidean distance between riemannian mean and ground thruth is : " + str(distance_euclid(riemann, cu)))


if (args.visual):
    f, axarr = plt.subplots(4, 4, figsize=(11, 5))
    for i, cu in enumerate(covs_u):
        # -----covariances---#
        margins = 1
        upper_bound = cu.max()+margins
        lower_bound = cu.min()-margins
        middle = (upper_bound+lower_bound)*0.5
        im = axarr[i, 0].pcolormesh(np.flipud(cu.T),
                                    cmap=plt.cm.gist_earth_r,
                                    edgecolors='k',
                                    vmin=lower_bound,
                                    vmax=upper_bound,
                                    **{'linewidth': 0.1})
        axarr[i, 0].set_axis_off()
        axarr[i, 0].set_aspect('equal', 'box')
        if(i == 0): axarr[i, 0].set_title('Ground Truth', **{'size': 10})

        im = axarr[i, 1].pcolormesh(np.flipud(est_covsu[i].T),
                                    cmap=plt.cm.gist_earth_r,
                                    edgecolors='k',
                                    vmin=lower_bound,
                                    vmax=upper_bound,
                                    **{'linewidth': 0.1})
        axarr[i, 1].set_axis_off()
        axarr[i, 1].set_aspect('equal', 'box')
        if (i == 0): axarr[i, 1].set_title('Our Method', **{'size': 10})

        im = axarr[i, 2].pcolormesh(np.flipud((eucl_mean[i]).T),
                                    cmap=plt.cm.gist_earth_r,
                                    edgecolors='k', vmin=lower_bound, vmax=upper_bound,
                                    **{'linewidth': 0.1})
        axarr[i, 2].set_axis_off()
        axarr[i, 2].set_aspect('equal', 'box')
        if (i == 0): axarr[i, 2].set_title('Euclidean mean', **{'size': 10})

        im = axarr[i, 3].pcolormesh(np.flipud((riemann_mean[i]).T),
                                    cmap=plt.cm.gist_earth_r,
                                    edgecolors='k',
                                    vmin=lower_bound, vmax=upper_bound, **{'linewidth': 0.1})
        axarr[i, 3].set_axis_off()
        axarr[i, 3].set_aspect('equal', 'box')
        divider = make_axes_locatable(axarr[i, 3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = f.colorbar(im, cax=cax)
        tick_locations = (lower_bound, middle, upper_bound)
        tick_labels = ("<"+'{:{width}.{prec}f}'.format(lower_bound, width=1, prec=1),
                       '{:{width}.{prec}f}'.format(middle, width=1, prec=1),
                       ">"+'{:{width}.{prec}f}'.format(upper_bound, width=1, prec=1))
        cb.locator = ticker.FixedLocator(tick_locations)
        cb.formatter = ticker.FixedFormatter(tick_labels)
        cb.update_ticks()
        if (i == 0): axarr[i, 3].set_title('Riemann mean', **{'size': 10})
    plt.savefig('./estimated_covars.eps', format='eps', dpi=1000)
