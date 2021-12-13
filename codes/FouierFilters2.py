import numpy as np
import matplotlib as plt
import numpy.linalg as LA
import scipy
import netket as nk
import jax
import jax.numpy as jnp
import netket.nn as nknn
from jax import grad, jit, vmap
from jax import random
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, eigvalsh
from sympy.combinatorics import Permutation as Perm
from sympy.interactive import init_printing
from jax.scipy.linalg import expm
#

import numpy
import torch
import cnine
import Snob2
from scipy.sparse import bsr_matrix



class FourierFilters:
    def __init__(self, J=None, lattice=None, Nsites=None, partit=None, p=1):
        self.J = J
        self.lattice = lattice
        self.Nsites = Nsites
        self.partit = partit
        self.dim = Snob2.SnIrrep(partit).get_dim()
        self.group = Snob2.Sn(self.Nsites)
        self.rep = Snob2.SnIrrep(partit)
        self.p = p
        YJMs_mat = np.zeros((self.Nsites, self.Nsites, self.dim))
        for i in range(self.Nsites):
            if i == self.Nsites -1:
                YJM = self.get_YJMs(i, i).astype('float64')
                YJMs_mat[i, i, :] = np.diag(YJM)
            for j in range(i, self.Nsites):
                YJM = self.get_YJMs(i,j).astype('float64')
                YJMs_mat[i,j,:] = np.diag(YJM)
        self.YJMs = jnp.asarray(YJMs_mat)
        self.Ham = jnp.asarray(self.Ham_rep().astype('float64'))



        # J ------------------ [J_1, J_2]

        # lattice ---------------------- nested list containing the tranpositions
        #                                   e.g. [[(1,2), (2,3), (3,4), (1,4)], (1,3), (2,4)]]

        # H -------------------- a Dict() object consisting of lattice and J


    def Ham_rep(self):
        # ls = list(H.keys())
        # orth = SymmetricGroupRepresentation(self.partit, 'orthogonal');
        # dim = len(np.array(orth.representation_matrix((1, 2))))

        #     print(ls)
        rep_mat0 = np.multiply(np.multiply(-1.0, len(self.lattice[0]) / 2, dtype='float64'),
                               np.diag(np.ones(self.dim)))
        for st in self.lattice[0]:
            #         print(H[ls[0]])
            # print(st)
            rep_st = self.rep.transp(st[0], st[1]).torch().numpy()
            #         print(rep_st)
            rep_mat0 = np.add(rep_mat0, rep_st)
        rep_mat0 = np.multiply(self.J[0] / 2, rep_mat0)
        if float(self.J[1]) == float(0):
            return rep_mat0
        else:
            rep_mat1 = np.multiply(np.multiply(-1.0, len(self.lattice[1]) / 2, dtype='float64'),
                                   np.diag(np.ones(self.dim)))
            for st in self.lattice[1]:
                rep_st = self.rep.transp(st[0], st[1]).torch().numpy()
                #         print(rep_st)
                rep_mat1 = np.add(rep_mat1 ,rep_st)
            rep_mat1 = np.multiply(self.J[1] / 2, rep_mat1)
            rep_mat_H = np.add(rep_mat0, rep_mat1)

            return rep_mat_H



    # def Compute_Eig(self, rep_mat_H):
    #     v, w = np.linalg.eig(rep_mat_H.astype('float64'))
    #     self.EDgs_energy = v[np.argmin(v)]
    #     self.EDgs = w[np.argmin(w)]

    def ED_Ham(self):
        Ham = self.Ham_rep()
        E_gs, V_gs = eigh(Ham.astype('float64'), subset_by_index=[0,1])
        return E_gs[0], V_gs[:,0]



    def get_YJMs(self, k, l):
        # compute X_k X_l for the YJM elements and by default X_1 = e
        Xkl= np.zeros((self.dim, self.dim))
        if k == l == 1:
            return np.diag(np.ones(self.dim))
        for i in range(1, max(k, l)):
            # pi = self.group('({}, {})'.format(i, max(k, l)))
            pi = self.rep.transp(i, max(k, l)).torch().numpy()
            if min(k, l) == 1:
                Xkl = np.add(Xkl, pi)
            else:
                for j in range(1, min(k, l)):
                    pj = self.rep.transp(j, min(k,l)).torch().numpy()
                    pij = np.matmul(pi, pj)
                    Xkl = np.add(Xkl, pij)
        return Xkl

    def YJM_Conv2d(self, YJMparams):
        # params for the YJM Hamiltonian ------ n(n+1)/2 params per layer of YJM
        if jnp.iscomplex(YJMparams) is False:
            print('the parameters are not complex valued ')
            raise ValueError
        YJM_conv = jnp.ones(self.dim)
        YJMparams = jnp.multiply(1j, YJMparams)
        for i in range(1, self.Nsites + 1):
            if i == self.Nsites:
                # YJM_rep = self.get_YJMs(int(i), int(i))
                YJM_rep = self.YJMs.at[i,i,:].get()
                # YJM_rep = jnp.diag(jnp.asarray(YJM_rep.astype('float64')))
                # print(YJM_rep)
                # YJM_rep = jnp.multiply(jnp.asarray(1j, dtype='complex128'), YJM_rep)

                # YJMparams = YJMparams.at[int(i), int(i)].multiply(jnp.asarray(1j, dtype='complex128'))
                # print(YJMparams.at[int(i), int(j)].get())
                exp_YJM_rep = jnp.exp(jnp.multiply(YJMparams.at[int(i), int(i)].get(), YJM_rep))
                # exp_YJM_rep = jnp.diag(exp_YJM_rep)
                # exp_YJM_rep = exp_YJM_rep / jnp.linalg.norm(exp_YJM_rep, ord='fro')
                YJM_conv = jnp.multiply(YJM_conv, exp_YJM_rep)
            for j in range(i, self.Nsites + 1):
                # YJM_rep = self.get_YJMs(int(i), int(j))
                # YJM_rep = jnp.diag(jnp.asarray(YJM_rep.astype('float64')))
                YJM_rep = self.YJMs.at[i,j,:].get()
                # YJMparams = YJMparams.at[int(i), int(j)].multiply(jnp.asarray(1j, dtype='complex128'))
                # print(YJMparams.at[int(i), int(j)].get())
                # YJM_rep = jnp.multiply(jnp.asarray(1j, dtype='complex128'),YJM_rep)
                exp_YJM_rep = jnp.exp(jnp.multiply(YJMparams.at[int(i), int(j)].get(), YJM_rep))
                # exp_YJM_rep = jnp.diagexp_YJM_rep)
                # exp_YJM_rep = exp_YJM_rep / jnp.linalg.norm(exp_YJM_rep, ord ='fro')
                # print(exp_YJM_rep)
                # print(YJM_conv.shape)
                # print(exp_YJM_rep.shape)
                YJM_conv = jnp.multiply(YJM_conv, exp_YJM_rep)

               # return a diagonal matrix
        YJM_conv =YJM_conv / jnp.linalg.norm(YJM_conv, ord= int(2))
        return jnp.diag(YJM_conv)

    def Heis_Conv2d(self, Hparam):
        # rep_mat_H = self.Ham_rep()
        Hparam = jnp.multiply(Hparam, 1j)
        # Heis_conv = expm(jnp.multiply(Hparam, jnp.asarray(rep_mat_H.astype('complex128'))))
        Heis_conv = expm(jnp.multiply(Hparam, self.Ham))
        Heis_conv = Heis_conv / jnp.linalg.norm(Heis_conv, ord=int(2))
        return Heis_conv

    def CSn_Ansazte(self, YJMparams, Hparams):
        # rep_mat_H = self.Ham_rep()
        ansatze = jnp.diag(jnp.ones(self.dim))
        for i in range(self.p):
            Heis_conv = self.Heis_Conv2d(Hparams.at[i].get())
            YJM_conv = self.YJM_Conv2d(YJMparams.at[:, :, i].get())
            ansatz = jnp.matmul(YJM_conv, Heis_conv)
            ansatze = jnp.matmul(ansatz, ansatze)
            ansatze = ansatze / jnp.linalg.norm(ansatze, ord=int(2))
        return ansatze






















