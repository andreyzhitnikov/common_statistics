import numpy as np
from common_statistics import est_common_density2D
import matplotlib.pyplot as plt
# ---create views---
vnum = 16
num = int(np.floor(5000/vnum)*vnum)


def generate_u_triangle(num):
    r = 15
    ss = r/2*np.random.random(int(num/3))-r/2
    tt = r/2*np.random.random(int(num/3))
    uu = r*np.random.random(int(num/3))-r/2
    u1 = np.concatenate((ss, tt, uu))
    u2 = np.concatenate((2*ss+r/2, -2*tt+r/2, -r/2*np.ones(int(num/3))))
    return np.stack([u1, u2], axis=0)


np.random.seed(8409)


DeT = 100
h = 0.7
cond_num = 2
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

plt.show()







