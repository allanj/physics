from numpy import zeros
import random as rd
from random import randint
from random import random
from numpy import int_
import numpy as np


# training data for the discrete problem
# dd is the range of the interaction
def input_output_prepare_FAEastModel_det_longrange_err(L=10, M=5000, dd=2, er=0.02):
    inpdata = zeros(shape=(M, L))
    outpdata = zeros(shape=(M, L))
    for isam in range(M):
        for ipos in range(L):
            inpdata[isam, ipos] = randint(0, 1)
            outpdata[isam, ipos] = inpdata[isam, ipos]
        for ipos in range(L - dd):
            if (1 - inpdata[isam, ipos + dd] > 0):
                outpdata[isam, ipos] = 1 - inpdata[isam, ipos]
        if random() < er:
            for ipos in range(L):
                outpdata[isam, ipos] = randint(0, 1)
    return int_(inpdata), int_(outpdata)


# exact evolution of discrete sequences of 0 and 1
# yinit is a list of 0s and 1s of length L
# dd is the range of the interaction
def FAEastModel_evolution_det_longrange(yinit, nsteps=5, L=10, dd=2):
    outpdata = zeros(shape=(nsteps, L))
    for ipos in range(L):
        outpdata[0, ipos] = yinit[ipos]
    for istep in range(nsteps - 1):
        for ipos in range(L):
            outpdata[istep + 1, ipos] = outpdata[istep, ipos]
        for ipos in range(L - dd):
            if (1 - outpdata[istep, ipos + dd] > 0):
                outpdata[istep + 1, ipos] = 1 - outpdata[istep, ipos]
    return int_(outpdata)


# generate training data for the continuous probabilita distribution
def input_output_prepare_diffusion_prob_nonlinear_lr_err(r=1, L=10, M=5000, m=3, m2=3, g=-0.1, g2=0.5, er=0.2):
    inpdata = np.zeros(shape=(M, L))
    outpdata = np.zeros(shape=(M, L))
    for isam in range(M):
        sinp = 0
        for ipos in range(L):
            inpdata[isam, ipos] = random()
            sinp += inpdata[isam, ipos]
        for jpos in range(L):
            inpdata[isam, jpos] /= sinp
        for jpos in range(L):
            jpos_l = jpos - 1
            jpos_r = jpos + 1
            jpos_ll = jpos - 2
            jpos_rr = jpos + 2
            if jpos_l < 0:
                jpos_l = L - 1
            if jpos_r > L - 1:
                jpos_r = 0
            if jpos_ll < 0:
                jpos_ll = L + jpos_ll
            if jpos_rr > L - 1:
                jpos_rr = jpos_rr - L
                # outpdata[isam,kpos+1] = r*inpdata[isam,kpos]**m + (1-r)*inpdata[isam,kpos+2]**m -inpdata[isam,kpos+1]**m +inpdata[isam,kpos+1]
            outpdata[isam, jpos] = g * (
            r * inpdata[isam, jpos_l] ** m + (1 - r) * inpdata[isam, jpos_r] ** m - inpdata[isam, jpos] ** m) + inpdata[
                                       isam, jpos]
            outpdata[isam, jpos] += g2 * (
            (1 - r) * inpdata[isam, jpos_ll] ** m2 + r * inpdata[isam, jpos_rr] ** m2 - inpdata[isam, jpos] ** m2)
        if random() < er:
            sout = 0.
            for jpos in range(L):
                outpdata[isam, jpos] = random()
                sout += outpdata[isam, jpos]
            for jpos in range(L):
                outpdata[isam, jpos] /= sout
    return inpdata,outpdata
                # evolve exactly the continuous probability distribution


# yinit is a list of L numbers between 0 and 1 such that their sum is equal to 1
def diffusion_prob_evolution_nonlinear_lr(yinit, r=0.5, nsteps=5000, L=10, m=3, m2=3, g=-0.1, g2=0.5):
    outpdata = zeros(shape=(nsteps, L))
    for ipos in range(L):
        outpdata[0, ipos] = yinit[ipos]
    for istep in range(nsteps - 1):
        for jpos in range(L):
            jpos_l = jpos - 1
            jpos_r = jpos + 1
            jpos_ll = jpos - 2
            jpos_rr = jpos + 2
            if jpos_l < 0:
                jpos_l = L - 1
            if jpos_r > L - 1:
                jpos_r = 0
            if jpos_ll < 0:
                jpos_ll = L + jpos_ll
            if jpos_rr > L - 1:
                jpos_rr = jpos_rr - L
            outpdata[istep + 1, jpos] = g * (
            r * outpdata[istep, jpos_l] ** m + (1 - r) * outpdata[istep, jpos_r] ** m - outpdata[istep, jpos] ** m) + \
                                        outpdata[istep, jpos]
            outpdata[istep + 1, jpos] += g2 * (
            (1 - r) * outpdata[istep, jpos_ll] ** m2 + r * outpdata[istep, jpos_rr] ** m2 - outpdata[istep, jpos] ** m2)

    return outpdata


def write_to_file(train_x, train_y, file_name):
    f = open(file_name, 'w')
    for (arr_x, arr_y) in zip(train_x, train_y):
        f.write(str(arr_x)[1:-1] + "\n")
        f.write(str(arr_y)[1:-1] + "\n")
        f.write("\n")
    f.close()

np.random.seed(9000)
rd.seed(1234)

[train_x, train_y] = input_output_prepare_FAEastModel_det_longrange_err()


print(str(train_x[1])[1:-1])
print(train_y.shape)

train_file = "train.txt"


write_to_file(train_x, train_y, train_file)


[test_x, test_y] = input_output_prepare_FAEastModel_det_longrange_err()
test_file = "test.txt"
write_to_file(test_x, test_y, test_file)


# print("######SEPARATOR######")
# [x,y] =  input_output_prepare_diffusion_prob_nonlinear_lr_err()
# print(x.shape)
# print(y.shape)
# print(np.sum(x))
# print(np.sum(y))


# y = diffusion_prob_evolution_nonlinear_lr()
# print(y.shape)
# print(np.sum(y))

