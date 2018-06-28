import numpy as np
from common_statistics import est_common_density2D
import matplotlib.pyplot as plt
import scipy.stats
# ---create views---
vnum = 16
num = int(np.floor(5000/vnum)*vnum)


def generate_u_triangle(num):
    r = 30
    ss = r/2*np.random.random(int(num/3))-r/2
    tt = r/2*np.random.random(int(num/3))
    uu = r*np.random.random(int(num/3))-r/2
    u1 = np.concatenate((ss, tt, uu))
    u2 = np.concatenate((2*ss+r/2, -2*tt+r/2, -r/2*np.ones(int(num/3))))
    return np.stack([u1, u2], axis=0)


np.random.seed(8409)


DeT = 100
h = 0.15
cond_num = 7
data = []

for i in range(0, vnum):
    while(True):
        A = np.random.random((2, 2))-1/2
        if(np.linalg.det(A) != 0):
            tildeB = np.dot(A, A.transpose())/np.abs(np.linalg.det(A))
            if(np.linalg.cond(tildeB) >= cond_num):
                B = tildeB*np.sqrt(DeT)
                break
    u = generate_u_triangle(num)
    n = np.random.multivariate_normal([0, 0], B, num).T
    data.append(u+n)

pUMatr, pdfs_of_subjects, positions, xmin, xmax, ymin, ymax, dimx, dimy = est_common_density2D(data, bw_method=h, outliers=0,
                                                                                               dimx=100, xmin=-50, xmax=50,
                                                                                               dimy=100, ymin=-50, ymax=50)
for j in range(0, vnum):
    counter_plt = j - 16*int(j/16)
    # --------------------------------------------------#
    row = counter_plt % 4
    col = int(counter_plt/4)
    if row == col == 0: f1, axarr1 = plt.subplots(4, 4, figsize=(12, 10))
    axarr1[row, col].imshow(pdfs_of_subjects[j], origin='lower', interpolation='nearest',
                            cmap=plt.cm.gist_earth_r,
                            extent=[xmin, xmax, ymin, ymax],
                            vmin=None, vmax=None)
    axarr1[row, col].plot(data[j][0], data[j][1], 'k.', markersize=0.03)
    axarr1[row, col].set_title(str(j))
    axarr1[row, col].set_xlim(xmin, xmax)
    axarr1[row, col].set_ylim(ymin, ymax)
    axarr1[row, col].set_xticks([xmin*2/3, 0, xmax*2/3])
    axarr1[row, col].set_yticks([ymin*2/3, 0, ymax*2/3])
    axarr1[row, col].tick_params(axis='both', which='major', labelsize=10)
    axarr1[row, col].set_axis_off()

x1, x2 = np.meshgrid(np.linspace(xmin, xmax, num=dimx, endpoint=True, retstep=False, dtype=None), np.linspace(ymin, ymax, num=dimy,endpoint=True))
pos = np.vstack([x1.ravel(), x2.ravel()])
values = np.hstack(data)
kernel = scipy.stats.gaussian_kde(values, bw_method=h)
pdfAll = np.reshape(kernel.evaluate(pos).T, x1.shape)

u = generate_u_triangle(num)
kernel2 = scipy.stats.gaussian_kde(u, bw_method=h)
pdfGround_truth = np.reshape(kernel2.evaluate(pos).T, x1.shape)

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
axs[0].imshow(pdfGround_truth, origin='lower', interpolation='nearest', cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax], vmin=None, vmax=None)
axs[0].set_axis_off()
axs[1].imshow(pdfAll, origin='lower', interpolation='nearest', cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax], vmin=None, vmax=None)
axs[1].set_axis_off()
axs[2].imshow(pUMatr, origin='lower', interpolation='nearest', cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax], vmin=None, vmax=None)
axs[2].set_axis_off()
plt.show()







