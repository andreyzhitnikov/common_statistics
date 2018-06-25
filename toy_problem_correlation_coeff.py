import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from pyriemann.utils.mean import mean_covariance, mean_euclid, mean_logeuclid, mean_wasserstein, mean_harmonic
from pyriemann.utils.distance import distance_euclid, distance_riemann
from common_statistics import est_common_cov
from rtnorm import rtnorm
import random
from joblib import Parallel, delayed
import pickle
import argparse

np.random.seed(555)
random.seed(555)


parser = argparse.ArgumentParser()
parser.add_argument('--visual', default=True, action="store_true", help='visual')
parser.add_argument('--visual_only', default=False, action="store_true", help='visual')
parser.add_argument('--num_cores', required=False, default=7, type=int, help='number of cores for parralelization')
args = parser.parse_args()


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def _draw_eigenvalues(dim=3, b=10, overlap=True):
        lambdas = []
        for i in range(0, dim):
            if overlap:
                currlam = b * random.random()
            else:
                currlam = b * random.random() + i*b
            lambdas.append(currlam)
        LambdaMatrix = np.diag(lambdas)
        return LambdaMatrix


def _draw_rotation_matrix(dim=3):
    theta = 2*np.pi * random.random()
    rotMatrix = np.column_stack((np.array([math.cos(theta), math.sin(theta)]),
                                 np.array([-1 * math.sin(theta), math.cos(theta)])))
    return rotMatrix


def processInput(kk, num_of_trials, cov_u, dim, vnum, bound):
    pearson_corr = []
    pearson_corr_riemann = []
    pearson_corr_eucl = []
    covars = []
    np.random.seed(kk)
    for k in range(0, num_of_trials):
        for i in range(0, vnum):
            # ---construct view----#
            LambdaMatrix = _draw_eigenvalues(dim=2, b=bound, overlap=False)
            rotMatrix = _draw_rotation_matrix(dim=dim)
            cov = np.add(cov_u, rotMatrix.dot(LambdaMatrix.dot(rotMatrix.transpose())))
            covars.append(cov)
        # --------------------------------------#
        est_cov_u = est_common_cov(covars, outliers=0)
        pearson_corr.append(est_cov_u[1, 0]/(math.sqrt(est_cov_u[0, 0])*math.sqrt(est_cov_u[1, 1])))
        riemann_mean = mean_covariance(np.array(covars), metric='riemann', sample_weight=None)
        eucl_mean = mean_covariance(np.array(covars), metric='euclid', sample_weight=None)
        pearson_corr_riemann.append(riemann_mean[1, 0]/(math.sqrt(riemann_mean[0, 0])*math.sqrt(riemann_mean[1, 1])))
        pearson_corr_eucl.append(eucl_mean[1, 0] / (math.sqrt(eucl_mean[0, 0]) * math.sqrt(eucl_mean[1, 1])))
    return pearson_corr, pearson_corr_riemann, pearson_corr_eucl


# ---define covariance of u----#
c_u = np.array([[1, 0.5], [0.5, 1]])
pearson_corr_original = c_u[1, 0]/(math.sqrt(c_u[0, 0])*math.sqrt(c_u[1, 1]))
dim = c_u.shape[0]

bound = 1.5


num_of_trials_in_batch = 30
vmin = 100
vmax = 1000
step = 100
pearson_corr_std = []
pearson_corr_riemann_std = []
pearson_corr_eucl_std = []

pearson_corr_mean = []
pearson_corr_riemann_mean = []
pearson_corr_eucl_mean = []

if(args.visual_only):
    first_part = pickle.load(open("./pearson_bias_variance1.pkl", "rb"))
    pearson_corr_std = first_part["pearson_our_corr_std"]
    pearson_corr_riemann_std = first_part["pearson_corr_riemann_std"]
    pearson_corr_eucl_std = first_part["pearson_corr_eucl_std"]
    pearson_corr_mean = first_part["pearson_corr_mean_our"]
    pearson_corr_riemann_mean = first_part["pearson_corr_riemann_mean"]
    pearson_corr_eucl_mean = first_part["pearson_corr_eucl_mean"]
else:
    seeds = np.random.random_integers(5, high=1000, size=args.num_cores)
    for vnum in range(vmin, vmax, step):
        results = Parallel(n_jobs=args.num_cores)(delayed(processInput)(seeds[kk], num_of_trials_in_batch, c_u, dim, vnum, bound) for kk in range(args.num_cores))
        # --------------------#
        pearson_corr = []
        pearson_corr_riemann = []
        pearson_corr_eucl = []

        for m in range(0, args.num_cores):
            pearson_corr.append(results[m][0])
            pearson_corr_riemann.append(results[m][1])
            pearson_corr_eucl.append(results[m][2])

        pearson_corr_std.append(np.std(np.hstack(pearson_corr)))
        pearson_corr_riemann_std.append(np.std(np.hstack(pearson_corr_riemann)))
        pearson_corr_eucl_std.append(np.std(np.hstack(pearson_corr_eucl)))

        pearson_corr_mean.append(np.mean(np.hstack(pearson_corr)))
        pearson_corr_riemann_mean.append(np.mean(np.hstack(pearson_corr_riemann)))
        pearson_corr_eucl_mean.append(np.mean(np.hstack(pearson_corr_eucl)))


    # --pickle dump--#
    pickle.dump({"pearson_our_corr_std": pearson_corr_std,
                 "pearson_corr_riemann_std": pearson_corr_riemann_std,
                 "pearson_corr_eucl_std": pearson_corr_eucl_std,
                 "pearson_corr_mean_our": pearson_corr_mean,
                 "pearson_corr_riemann_mean": pearson_corr_riemann_mean,
                 "pearson_corr_eucl_mean": pearson_corr_eucl_mean}, open("./pearson_bias_variance1.pkl", "wb"))


if(args.visual_only or args.visual):
    plt.figure(figsize=(5, 3))

    plt.errorbar(range(vmin, vmax, step), pearson_corr_mean, label='our', yerr=pearson_corr_std, **{'linewidth': 2})
    plt.errorbar(range(vmin, vmax, step), pearson_corr_riemann_mean, label='riemann mean', yerr=pearson_corr_riemann_std, **{'linewidth': 2})
    plt.errorbar(range(vmin, vmax, step), pearson_corr_eucl_mean, label='eucl mean', yerr=pearson_corr_eucl_std, **{'linewidth': 2})

    plt.xticks(range(vmin, vmax, step))
    plt.xlabel('number of views')
    plt.ylabel('coefficient')
    plt.ylim((0, 0.6))
    plt.legend(loc='lower left')
    plt.tight_layout()

    # --figure dump---#
    plt.savefig("./pearson_as_numofviews.eps", format="eps")


pearson_corr_std = []
pearson_corr_riemann_std = []
pearson_corr_eucl_std = []

pearson_corr_mean = []
pearson_corr_riemann_mean = []
pearson_corr_eucl_mean = []

vnum = 300
boundmin = 0
boundmax = 2
boundstep = 0.1
if(args.visual_only):
    second_part = pickle.load(open("../pearson_bias_variance2.pkl", "rb"))
    pearson_corr_std = second_part["pearson_our_corr_std"]
    pearson_corr_riemann_std = second_part["pearson_corr_riemann_std"]
    pearson_corr_eucl_std = second_part["pearson_corr_eucl_std"]
    pearson_corr_mean = second_part["pearson_corr_mean_our"]
    pearson_corr_riemann_mean = second_part["pearson_corr_riemann_mean"]
    pearson_corr_eucl_mean = second_part["pearson_corr_eucl_mean"]
else:
    for bound in frange(boundmin, boundmax, boundstep):
        results = Parallel(n_jobs=args.num_cores)(delayed(processInput)(seeds[kk], num_of_trials_in_batch, c_u, dim, vnum, bound) for kk in range(args.num_cores))
        # --------------------#
        pearson_corr = []
        pearson_corr_riemann = []
        pearson_corr_eucl = []

        for m in range(0, args.num_cores):
            pearson_corr.append(results[m][0])
            pearson_corr_riemann.append(results[m][1])
            pearson_corr_eucl.append(results[m][2])

        pearson_corr_std.append(np.std(np.hstack(pearson_corr)))
        pearson_corr_riemann_std.append(np.std(np.hstack(pearson_corr_riemann)))
        pearson_corr_eucl_std.append(np.std(np.hstack(pearson_corr_eucl)))

        pearson_corr_mean.append(np.mean(np.hstack(pearson_corr)))
        pearson_corr_riemann_mean.append(np.mean(np.hstack(pearson_corr_riemann)))
        pearson_corr_eucl_mean.append(np.mean(np.hstack(pearson_corr_eucl)))

    pickle.dump({"pearson_our_corr_std": pearson_corr_std,
                 "pearson_corr_riemann_std": pearson_corr_riemann_std,
                 "pearson_corr_eucl_std": pearson_corr_eucl_std,
                 "pearson_corr_mean_our": pearson_corr_mean,
                 "pearson_corr_riemann_mean": pearson_corr_riemann_mean,
                 "pearson_corr_eucl_mean": pearson_corr_eucl_mean}, open("../pearson_bias_variance2.pkl", "wb"))


if(args.visual_only or args.visual):
    plt.figure(figsize=(5, 3))

    plt.errorbar(list(frange(boundmin, boundmax, boundstep)), pearson_corr_mean, label='our', yerr=pearson_corr_std, **{'linewidth': 2})
    plt.errorbar(list(frange(boundmin, boundmax, boundstep)), pearson_corr_riemann_mean, label='riemann mean', yerr=pearson_corr_riemann_std, **{'linewidth': 2})
    plt.errorbar(list(frange(boundmin, boundmax, boundstep)), pearson_corr_eucl_mean, label='eucl mean', yerr=pearson_corr_eucl_std, **{'linewidth': 2})

    plt.xticks(list(frange(boundmin, boundmax, boundstep)))
    plt.xlabel('bound value')
    plt.ylabel('coefficient')
    plt.ylim((0, 0.6))
    plt.legend(loc='lower left')
    plt.tight_layout()

    # --figure dump--#
    plt.savefig("./pearson_as_bound.eps", format="eps")

plt.show()












