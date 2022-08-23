import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import logging
from FouierFilters2 import FourierFilters
from jax import random
from jax import grad, jit, vmap
# import tensorflow_quatum as tfq
from jax import custom_jvp
import optax
import pandas as pd
import datetime
import os

'''
using optax meta learning 
'''
from typing import Callable, Iterator, Tuple
import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

class CSnGradient(FourierFilters):
    '''
    Standard SGD method implemented using JAX autodifferentiation megthd

    Unsurpervised method for the minilization of the GS energy for each Sn irrep

    Wirtten in the language environment SAGE9.2, supported SageMath environment

    '''

    def __init__(self, lr=float(2e-3), max_iter=int(1001), gamma=float(0.95),meta_lr = 0.03 ,num_samples=None, quantumnoise = True ,**kwargs):
        super(CSnGradient, self).__init__(**kwargs)
        self.lr = jnp.array(lr) 
        self.meta_lr = jnp.array(meta_lr)
        self.max_iter = max_iter
        self.gamma = gamma
        self.sampling = np.random.randint(0, high=self.dim, size=num_samples)
        num_trans = jnp.add(len(self.lattice[0]), len(self.lattice[1]))
        # trans_const = jnp.multiply(num_trans, jnp.ones(self.dim), dtype='complex128')
        self.num_trans = num_trans
        self.quantumnoise = quantumnoise
        self.logging = {'energy': [], 'iteration':[]}
        self.logging2 = {'CQAGstate': [], 'iteration':[]}



    def random_params(self, scale=float(1e-2)):
        length = jnp.add(jnp.multiply(jnp.multiply(self.Nsites, self.Nsites), self.p), self.p)
        # print(length)
        return jnp.multiply(scale, random.normal(random.PRNGKey(self.p), (length,)))

    def random_params2(self, scale=float(1e-1)):
        YJMparams = jnp.zeros((self.Nsites, self.Nsites, self.p))
        Hparams = jnp.zeros((self.p))
        for i in range(self.p):
            w_key, b_key = random.split(random.PRNGKey(int(i)))
            YJMparams = YJMparams.at[:, :, i].set(jnp.multiply(scale, random.normal(w_key, (self.Nsites, self.Nsites))))
            Hparams = Hparams.at[i].set(jnp.multiply(scale, random.normal(b_key)))
        return YJMparams, Hparams

    '''
    -----------------------------------------------------------------------
    
    Custom derivative method (used for exact gradient at the level of energy functional) 
    
    Needed to update also the reverse mode auto-diff 
    
    ----------------------------------------------------------------------
    '''

    # @custom_jvp
    def CSn_VStates(self,YJMparams, Hparams):
        '''
        disgard the complex components for better numerical stability
        :param YJMparams:
        :param Hparams:
        :return:
        '''
        #         YJMparams = Params[0]
        #         Hparams = Params[1]
        # split = jnp.multiply(jnp.multiply(self.Nsites, self.Nsites), self.p)
        # YJMparams = jnp.reshape(Params.at[int(0):split].get(), newshape=(self.Nsites, self.Nsites, self.p))
        # Hparams = Params.at[split:int(-1)].get()
        ansazte = self.CSn_Ansazte(YJMparams, Hparams)
        GSket = jnp.zeros(self.dim)
        for i in range(len(self.sampling)):
            basis = jnp.zeros(self.dim)
            basis = basis.at[i].set(self.sampling[i])
            GSket = jnp.add(jnp.matmul(ansazte, basis), GSket)
        GSket = jnp.real(GSket)
        return GSket / jnp.linalg.norm(GSket)

    
    def Expect_braket(self, YJMparams, Hparams, scale = 1e-3):
        groundstate = self.CSn_VStates(YJMparams, Hparams)
        rep_H = self.Ham_rep().astype('float64')
        rep_H = jnp.asarray(rep_H)
        # if self.quantumnoise:
        #     noise = jax.random.normal(random.PRNGKey(int(24)), jnp.shape(rep_H)) * scale
        #     rep_H = noise + rep_H


        rep_H += jnp.add(rep_H, jnp.multiply(self.num_trans, jnp.diag(jnp.ones(self.dim))))
        # make sure the Hamiltonian is positive-definite by adding scalar multiple of identitties
        expectation = jnp.matmul(jnp.conjugate(groundstate), jnp.matmul(rep_H, groundstate)) / jnp.linalg.norm(groundstate)
        # self.logging['energy'].append(expectation)
        return expectation
    
    def Expect_braket_energy(self,YJMparams, Hparams, scale=1e-3):
        # split = jnp.multiply(jnp.multiply(self.Nsites, self.Nsites), self.p)
        # YJMparams = jnp.reshape(Params.at[int(0):split].get(), newshape=(self.Nsites, self.Nsites, self.p))
        # Hparams = Params.at[split:int(-1)].get()
        groundstate = self.CSn_VStates(YJMparams, Hparams)
        rep_H = self.Ham_rep().astype('float64')
        rep_H = jnp.asarray(rep_H)
        # if self.quantumnoise:
        #     noise = jax.random.normal(random.PRNGKey(int(24)), jnp.shape(rep_H)) * scale
        #     rep_H = noise + rep_H
        expectation =jnp.matmul(jnp.conjugate(groundstate), jnp.matmul(rep_H, groundstate)) / jnp.linalg.norm(groundstate)
        # self.logging['energy'].append(expectation)
        return expectation

    '''
    meta leanring using optax phase 
    '''
    def step(self, Hparams, YJMparams):
        grad_yjm = jax.grad(self.Expect_braket, argnums=0)(YJMparams, Hparams)
        grad_h = jax.grad(self.Expect_braket, argnums=1)(YJMparams, Hparams)
        