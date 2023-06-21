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
from scipy.linalg import eigh, eigvalsh

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
from tqdm import tqdm 


# import argparse
# parser = argparse.ArgumentParser()
# # parser.add_argument('--noise_scale', type=float, required=True)
# parser.add_argument('--lr', type=float, required=True)
# parser.add_argument('--J2', type=float, required=True)
# parser.add_argument('--p', type=int, default=4)
# parser.add_argument('--iterations', type=int, default=10)
# args = parser.parse_args()


class CSnGradient(FourierFilters):
    '''
    Standard SGD method implemented using JAX autodifferentiation megthd

    Unsurpervised method for the minilization of the GS energy for each Sn irrep

    Wirtten in the language environment SAGE9.2, supported SageMath environment

    '''

    def __init__(self, lr=float(2e-3), max_iter=int(1001), gamma=float(0.95), sigma=0.01 ,num_samples=None, quantumnoise = False, seed=0,**kwargs):
        super(CSnGradient, self).__init__(**kwargs)
        self.lr = jnp.array(lr) 
        # self.meta_lr = jnp.array(meta_lr)
        self.sigma = sigma
        self.max_iter = max_iter
        self.gamma = gamma
        self.sampling = np.random.randint(0, high=self.dim, size=num_samples)
        # num_trans = jnp.add(len(self.lattice[0]), len(self.lattice[1]))
        # trans_const = jnp.multiply(num_trans, jnp.ones(self.dim), dtype='complex128')
        # self.num_trans = num_trans
        self.quantumnoise = quantumnoise
        self.logging = {'energy': [], 'iteration':[]}
        self.logging2 = {'CQAGstate': [], 'iteration':[]}
        self.seed = seed



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
        params_dict = {'YJM': YJMparams, 'Heis': Hparams} 
        return params_dict
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
    
    def Expect_braket_energy(self,params_dict: dict, scale=1e-3):
        # split = jnp.multiply(jnp.multiply(self.Nsites, self.Nsites), self.p)
        # YJMparams = jnp.reshape(Params.at[int(0):split].get(), newshape=(self.Nsites, self.Nsites, self.p))
        # Hparams = Params.at[split:int(-1)].get()
        groundstate = self.CSn_VStates(params_dict['YJM'], params_dict['Heis'])
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
    # @jax.jit
    # def step(self, Hparams, YJMparams):
    #     grad_yjm = jax.grad(self.Expect_braket, argnums=0)(YJMparams, Hparams)
    #     grad_h = jax.grad(self.Expect_braket, argnums=1)(YJMparams, Hparams)
    

    def train(self, optimizer:optax.GradientTransformation): 
        params_dict_init = self.random_params2()
        opt_state = optimizer.init(params_dict_init)
        loss_history, param_history = [], [params_dict_init]
        @jax.jit
        def step(params_dict, opt_state):
            loss, grads = jax.value_and_grad(self.Expect_braket_energy, argnums=0)(params_dict)
            updates, opt_state = optimizer.update(grads, opt_state, params_dict)
            # print(opt_state.hyperparams.keys())
            params_dict = optax.apply_updates(params_dict, updates)
            return params_dict, opt_state, loss

        for it in tqdm(range(1, self.max_iter + 1)): 
            p_dict = param_history[-1]
            # print('p_dict | YJM: {} | Heis: {}'.format(p_dict['YJM'].shape, p_dict['Heis']))
            updated_p_dict, opt_state, loss = step(p_dict, opt_state)
            if self.quantumnoise: 
                w_key, b_key = random.split(random.PRNGKey(self.seed)) 
                w_noise = opt_state.hyperparams['learning_rate'] * self.sigma *jax.random.normal(w_key, updated_p_dict['Heis'].shape)
                b_noise = opt_state.hyperparams['learning_rate'] * self.sigma * jax.random.normal(b_key, updated_p_dict['YJM'].shape)
                updated_p_dict['Heis'] += w_noise 
                updated_p_dict['YJM'] += b_noise
                
            # print(it)
            # if it % 20 ==0:
            #     if opt_state.hyperparams['learning_rate'] >= 1e-4:
            #         opt_state.hyperparams['learning_rate'] = opt_state.hyperparams['learning_rate'] / 2
            #     print('learning rate now: {}'.format(opt_state.hyperparams['learning_rate']))
            # print("Step {:3d}   Cost_L = {:9.7f}".format(it, loss))
            # updated_p_dict = {'YJM': p_dict['YJM'] - args.lr * gradient['YJM'],
            #                 'Heis': p_dict['Heis'] - args.lr * gradient['Heis']}
            param_history.append(updated_p_dict)
            loss_history.append(loss)
        return loss_history, param_history


# def main():
#     lattice4 =[[(1,2), (1,8), (2,3), (2,7), (3,4), (3,6), (4,5), (5,6), (5, 12),
#             (6,7), (6, 11), (7, 10), (7, 8), (8,9), (9, 10), (10, 11), (11, 12)],

#            [(1,3), (1,9), (1,7), (2,4), (2,8), (2, 10), (2, 6), (3, 11), (3,5), (3,7),
#             (4, 12), (4, 6), (5, 7), (5, 11), (6, 8), (6, 10), (6, 12),  (7, 9), (7, 11), (8, 10),
#             (9, 11), (10, 12)]]
#     partit = [int(6),int(6)]
#     Nsites = int( 12)    

#     CsnFourier = CSnGradient(J= [1.0, args.J2], lattice = lattice4, Nsites=Nsites,
#                     partit=partit,p=args.p, num_samples =int(1000), max_iter = args.iterations, lr=args.lr)


#     Ham_rep = CsnFourier.Ham_rep()

#     # print(CsnFilters.rep_mat_H)
#     E_gs, V_gs = eigh(Ham_rep.astype('float64'), subset_by_index=[0,1])
#     V_gs = V_gs[:,0]
#     E_gs = E_gs[0]
#     V_gs = jnp.asarray(V_gs)
#     print('True Ground state Energy via ED for partition {}:--- ({}) '.format(partit, E_gs))
#     # print('True Ground State wavefuncion in Sn irrep basis for partition {}:--- {}'.format(partit, V_gs))

#     print('Irrep Dims for {}: --- {}'.format(partit, CsnFourier.dim))

#     print('now the gradient phase')
    
#     scheduled_adam = optax.inject_hyperparams(optax.adamw)(learning_rate=args.lr)
#     # optimizer = optax.adamw(learning_rate=args.lr)
#     loss_history, param_history =CsnFourier.train(scheduled_adam)
#     plt.style.use("seaborn")
#     plt.plot(loss_history, "g", label='Sn-CQA Ansatz')
#     plt.axhline(E_gs, color='r', linestyle='-', label='ED energy: {:.4f}'.format(E_gs))
#     plt.ylabel("Cost function")
#     plt.xlabel("Optimization steps")
#     plt.legend(loc="upper right")
#     plt.title(f'CQA Training with layers {args.p} with lattice size: {Nsites}') 
#     plt.show()
#     plt.savefig(f'Figures/CQA_p{args.p}_lattice{Nsites}')  


# if __name__ =='__main__':
#     main()