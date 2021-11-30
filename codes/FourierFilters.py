import numpy as np
import matplotlib as plt
import numpy.linalg as LA
from sage.combinat.symmetric_group_representations import SymmetricGroupRepresentation
from sage.combinat.permutation import *
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


class FourierFilters:
    def __init__(self, J=None, lattice=None, Nsites=None, partit=None, p=1):
        self.J = J
        self.lattice = lattice
        self.Nsites = Nsites
        self.partit = partit
        self.dim = np.array(SymmetricGroupRepresentation(self.partit, 'orthogonal').
                            representation_matrix((1, 2))).shape[0]
        self.group = SymmetricGroup(self.Nsites)
        self.p = p
        # J ------------------ [J_1, J_2]

        # lattice ---------------------- nested list containing the tranpositions
        #                                   e.g. [[(1,2), (2,3), (3,4), (1,4)], (1,3), (2,4)]]

        # H -------------------- a Dict() object consisting of lattice and J

    def rep_matrix(self, st):
        orth = SymmetricGroupRepresentation(self.partit, 'orthogonal')
        #         print(st)
        #         print(orth.representation_matrix(st))
        mat = np.array(orth.representation_matrix(st))
        return mat

    def Ham_rep(self):
        # ls = list(H.keys())
        # orth = SymmetricGroupRepresentation(self.partit, 'orthogonal');
        # dim = len(np.array(orth.representation_matrix((1, 2))))

        #     print(ls)
        rep_mat0 = np.multiply(np.multiply(-1.0, len(self.lattice[0]) / 2, dtype='complex128'),
                               np.diag(np.ones(self.dim)))
        for st in self.lattice[0]:
            #         print(H[ls[0]])
            # print(st)
            rep_st = self.rep_matrix(st)
            #         print(rep_st)
            rep_mat0 = np.add(rep_mat0, rep_st)
        rep_mat0 = np.multiply(self.J[0] / 2, rep_mat0)
        if float(self.J[1]) == float(0):
            return rep_mat0
        else:
            rep_mat1 = np.multiply(np.multiply(-1.0, len(self.lattice[1]) / 2, dtype='complex128'),
                                   np.diag(np.ones(self.dim)))
            for st in self.lattice[1]:
                rep_st = self.rep_matrix(st)
                #         print(rep_st)
                rep_mat1 = rep_mat1 + rep_st
            rep_mat1 = np.multiply(self.J[1] / 2, rep_mat1)
            rep_mat_H = np.add(rep_mat0, rep_mat1)
            return rep_mat_H

    # def Compute_Eig(self, rep_mat_H):
    #     v, w = np.linalg.eig(rep_mat_H.astype('float64'))
    #     self.EDgs_energy = v[np.argmin(v)]
    #     self.EDgs = w[np.argmin(w)]

    # def ED_Ham(self, rep_mat_H):
    #     E_gs, V_gs= eigvalsh(rep_mat_H.astype('float64'), subset_by_index=[0, 1])
    #     return E_gs, V_gs

    def get_YJMs(self, k, l, opt= 'rep'):
        # compute X_k X_l for the YJM elements and by default X_1 = e
        Xkl = []
        if k == l == 1:
            return jnp.diag(jnp.ones(self.dim))
        for i in range(1, max(k, l)):
            pi = self.group('({}, {})'.format(i, max(k, l)))

            if min(k, l) == 1:
                Xkl.append(pi)
            for j in range(i, min(k, l)):
                # print(j)
                pj = self.group('({}, {})'.format(j, min(k, l)))
                # print(pj)
                Xkl.append(pi * pj)
        if opt =='exact':
            return Xkl
        elif opt == 'rep':
            YJM_rep = jnp.zeros((self.dim, self.dim))
            # print(Xkl)
            for i in range(len(Xkl)):
                # print(Xkl[i])
                rep_m = self.rep_matrix(Xkl[i])
                YJM_rep = np.add(YJM_rep, rep_m)
                # print('YJM_rep at iteration {}: --- {}'.format(i, YJM_rep))
            return YJM_rep

    def YJM_Conv2d(self, YJMparams):
        # params for the YJM Hamiltonian ------ n(n+1)/2 params per layer of YJM
        if jnp.iscomplex(YJMparams) is False:
            print('the parameters are not complex valued ')
            raise ValueError
        YJM_conv = jnp.ones(self.dim)
        for i in range(1, self.Nsites + 1):
            if i == self.Nsites:
                YJM_rep = self.get_YJMs(int(i), int(i))
                # print(YJM_rep)
                YJM_rep = jnp.multiply(jnp.asarray(1j, dtype='complex128'), jnp.asarray(YJM_rep.astype('complex128')))
                exp_YJM_rep = jnp.exp(jnp.multiply(YJMparams.at[int(i), int(i)].get(), YJM_rep))
                # exp_YJM_rep = jnp.diag(exp_YJM_rep)
                # exp_YJM_rep = exp_YJM_rep / jnp.linalg.norm(exp_YJM_rep, ord='fro')
                YJM_conv = jnp.matmul(YJM_conv, exp_YJM_rep)
            for j in range(i, self.Nsites + 1):
                # print('i = {}, j = {}'.format(i,j))
                YJM_rep = jnp.asarray(self.get_YJMs(int(i), int(j)).astype('complex128'))
                YJM_rep = jnp.multiply(jnp.asarray(1j, dtype='complex128'),YJM_rep)
                exp_YJM_rep = jnp.exp(jnp.multiply(YJMparams.at[int(i), int(j)].get(), YJM_rep))
                # exp_YJM_rep = jnp.diagexp_YJM_rep)
                # exp_YJM_rep = exp_YJM_rep / jnp.linalg.norm(exp_YJM_rep, ord ='fro')
                # print(exp_YJM_rep)
                YJM_conv = jnp.matmul(YJM_conv, exp_YJM_rep)
        return jnp.diag(YJM_conv / jnp.linalg.norm(YJM_conv))

    def Heis_Conv2d(self, Hparam, rep_mat_H):
        Heis_conv = jnp.exp(jnp.multiply(Hparam, jnp.asarray(rep_mat_H.astype('complex128'))))
        Heis_conv = jnp.multiply(jnp.asarray(1j, dtype='complex128'), Heis_conv)

        return Heis_conv / jnp.linalg.norm(Heis_conv, ord='fro')

    def CSn_Ansazte(self, YJMparams, Hparams):
        rep_mat_H = self.Ham_rep()
        ansatze = jnp.diag(jnp.ones(self.dim))
        for i in range(self.p):
            Heis_conv = self.Heis_Conv2d(Hparams.at[i].get(), rep_mat_H)
            YJM_conv = self.YJM_Conv2d(YJMparams.at[:, :, i].get())
            ansatz = jnp.matmul(YJM_conv, Heis_conv)
            ansatze = jnp.matmul(ansatz, ansatze)
            ansatze = ansatze / jnp.linalg.norm(ansatze)
        return ansatze






















