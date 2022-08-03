import torch
import cnine
import Snob2
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import scipy
import netket as nk
import jax
import jax.numpy as jnp
import netket.nn as nknn
from jax import grad, jit, vmap, vjp
from jax import random
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, eigvalsh
from sympy.combinatorics import Permutation as Perm
from sympy.interactive import init_printing
import json
import networkx as nx
from jax import random
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import netket.nn as nknn
import flax.linen as nn
import time
import json
from FouierFilters2 import FourierFilters
from sgd import CSnGradient
from jax import random

# rho = Snob2.SnIrrep([6,6])
#
# x = rho.transp(3,12).torch().numpy()
#
# print(x.shape)
#
# from FouierFilters2 import FourierFilters
# lattice = [[(1,2), (4,7)],[(2,8)]]
# FFilters = FourierFilters(J=[1,0], lattice=lattice,partit=[9,3],Nsites=12, p=1 )
#
# print(np.linalg.norm(np.diag(FFilters.get_YJMs(3,12))))
#
# print(np.linalg.norm(FFilters.get_YJMs(3,12), ord='fro'))
#
# print(FFilters.get_YJMs(3,12).shape)


'''
------------------------------------------------------

KetNet 3 x 4 Rectangular lattices 

----------------------------------------------------- 

'''


J = [1, 0.5]
# graph = nk.graph.Grid(extent= [2,4], pbc=False)
# edges = graph.edges
# nx.draw(graph.to_networkx(), with_labels=True, font_weight='bold')
edge_colors = [[0, 1, 1], [0,7, 1], [1, 2,1], [1,6, 1], [2,3,1],
               [2,5,1], [3,4,1], [4,5,1], [4,11,1], [5,6,1], [5, 10, 1],
              [6,9,1], [6,7,1],[7,8,1], [8,9,1], [9,10, 1], [10, 11, 1],
               # J2 terms now for the frustration
               [0,2,2], [0,8,2], [0,6,2], [1,3,2], [1,7,2], [1,9,2],
               [1,5,2], [2,10,2], [2,4,2], [2,6,2], [3,11,2], [3,5,2],
               [4,6,2], [4,10,2], [5,7,2], [5,9,2], [6,8,2], [6, 10, 2],
               [7,9,2], [8, 10, 2], [9, 11, 2], [5, 11, 2]]

#Sigma^z*Sigma^z interactions
sigmaz = [[1, 0], [0, -1]]
mszsz = (np.kron(sigmaz, sigmaz))

#Exchange interactions
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

bond_operator = [
    (J[0] * mszsz).tolist(),
    (J[1] * mszsz).tolist(),
    (J[0] * exchange).tolist(),
    (J[1] * exchange).tolist(),
]

bond_color = [1, 2, 1, 2]

# graph = nk.graph.Grid(extent= [2,4], pbc=False, edge_colors = edge_colors)
g = nk.graph.Graph(edges=edge_colors)
# nx.draw(g.to_networkx(), with_labels=True, font_weight='bold')
hi = nk.hilbert.Spin(s=float(0.5), total_sz=float(0.0), N=g.n_nodes)
# print(g.edge_colors)
ha = nk.operator.GraphOperator(hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)
evals = nk.exact.lanczos_ed(ha, compute_eigenvectors=False)
exact_gs_energy1 = evals[0]
print('The exact ground-state energy from computational basis for J2 = {} is-- ({}) '.format(J[1], exact_gs_energy1/float(4)))









'''
----------------------------------------------------------

test on 3 x 4 squares  

---------------------------------------------------------
'''


# J = [1, 0] # unfrustrated system for now
# lattice2 = [[(1,2), (2,3), (3,4),(4,5), (5,6),(1,6), (2,5)],[(1,5), (2,6), (2,4), (3,5),(1,3), (4,6)]]
# lattice3 = [[(1,2), (1,5), (2,3), (3,4), (3,6),
#              (5,7), (7, 9), (7,8), (6,10), (9,10), (10,11), (10,12)],
#             [(2,5), (4,6), (6,9), (8,9), (11,12), (1,3),
#             (2,4), (3,10), (2,6), (6,11), (6,12), (9,11),
#             (9, 12), (5, 8), (1,7), (7, 10), (5,9)]]
# J = [1,0]
lattice4 =[[(1,2), (1,8), (2,3), (2,7), (3,4), (3,6), (4,5), (5,6), (5, 12),
            (6,7), (6, 11), (7, 10), (7, 8), (8,9), (9, 10), (10, 11), (11, 12)],

           [(1,3), (1,9), (1,7), (2,4), (2,8), (2, 10), (2, 6), (3, 11), (3,5), (3,7),
            (4, 12), (4, 6), (5, 7), (5, 11), (6, 8), (6, 10), (6, 12),  (7, 9), (7, 11), (8, 10),
            (9, 11), (10, 12)]]
partit = [int(6),int(6)]
Nsites = int( 12)


CsnFourier = CSnGradient(J= J, lattice = lattice4, Nsites=Nsites,
                    partit=partit,p=int(4), num_samples =int(1000), max_iter = int(5001), lr=1e-3)


Ham_rep = CsnFourier.Ham_rep()

# print(CsnFilters.rep_mat_H)
E_gs, V_gs = eigh(Ham_rep.astype('float64'), subset_by_index=[0,1])
V_gs = V_gs[:,0]
E_gs = E_gs[0]
V_gs = jnp.asarray(V_gs)
print('True Ground state Energy via ED for partition {}:--- ({}) '.format(partit, E_gs))
# print('True Ground State wavefuncion in Sn irrep basis for partition {}:--- {}'.format(partit, V_gs))

print('Irrep Dims for {}: --- {}'.format(partit, CsnFourier.dim))


'''
-----------------------------------------------------------------------------

Sn-CQA Ansatze testing phase 

-----------------------------------------------------------------------------
'''

J = CsnFourier.Expect_braket
opt_YJM, opt_H = CsnFourier.CSn_nadam(J, scale=float(1e-2))


O_gs = CsnFourier.Groundstate(opt_YJM, opt_H)
optimized_energy = CsnFourier.Expect_braket_energy(opt_YJM, opt_H)
CsnFourier.logging['EDGstate'] = np.asarray(V_gs)
CsnFourier.logging['CQAGstate'] = np.asarray(O_gs)
CsnFourier.logging['overlap'] = np.dot(np.array(V_gs), np.array(O_gs))
CsnFourier.logging['precision'] = np.abs(E_gs - np.asarray(optimized_energy)) / np.abs(E_gs)

print('the optimized ground state: {}'.format(O_gs))
print('------------------------------------')
print('Optimized lowest energy: {}'.format(optimized_energy))
print('-------------------------------------')
print('True Ground State wavefuncion in Sn irrep basis: {}'.format(V_gs))
print('-------------------------------------')
print('The overlap between the optimized state and the ground state: {}'.format(jnp.dot(O_gs,
                                                                               V_gs)))


import pandas as pd
import datetime
snapshotdate = datetime.datetime.now().strftime('%m-%d_%H-%M')
df = pd.DataFrame.from_dict(CsnFourier.logging )
df.to_csv('../data/'+ snapshotdate +  '/CQA_J08_6square2.csv')
# print('The distance between the optimized state and the ground state: {}'.format(jnp.linalg.norm(jnp.subtract(V_gs, O_gs))))

"""
J2 = 0.5
The best run with n_samples = 1000, learning rates = 0.002, p = 4, with the overlap = 0.9973676

J2 = 0.8 

The best run with n_samples = 1000, learning rates = 0.001, p=4, with the overlap = 0.9996319. saved in CQA_J08_6squares2.csv

"""

