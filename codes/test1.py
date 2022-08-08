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

KetNet Kagomme Lattice 

----------------------------------------------------- 

'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--noise_scale', type=float, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--J2', type=float, required=True)
args = parser.parse_args()

# J = [1, 0.8]
# graph = nk.graph.Grid(extent= [2,4], pbc=False)
# edges = graph.edges
# nx.draw(graph.to_networkx(), with_labels=True, font_weight='bold')
edge_colors = [[0, 1, 1], [0, 4, 1], [1, 2,1],
               [2, 3, 1], [2,5,1], [4,6,1], [6,8,1], [6,7,1],
              [5,9,1], [8,9,1],[9,10,1], [9,11,1],
               # J2 terms now for the frustration
               [1,4,2], [3,5,2], [5,8,2], [7,8,2], [10,11,2], [0,2,2],
               [1,3,2], [2,9,2], [1,5,2], [5,10,2], [5,11,2], [8,10,2],
               [8,11,2], [4,7,2], [0,6,2], [6,9,2], [4,8,2]]

#Sigma^z*Sigma^z interactions
sigmaz = [[1, 0], [0, -1]]
mszsz = (np.kron(sigmaz, sigmaz))

#Exchange interactions
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

bond_operator = [
    (1.0 * mszsz).tolist(),
    (args.J2 * mszsz).tolist(),
    (1.0 * exchange).tolist(),
    (args.J2 * exchange).tolist(),
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
print('The exact ground-state energy from computational basis for J2 = {} is-- ({}) '.format(args.J2, exact_gs_energy1/float(4)))









'''
----------------------------------------------------------

test on CQA-Kagomme lattice 

---------------------------------------------------------
'''


# J = [1, 0] # unfrustrated system for now
# lattice2 = [[(1,2), (2,3), (3,4),(4,5), (5,6),(1,6), (2,5)],[(1,5), (2,6), (2,4), (3,5),(1,3), (4,6)]]
lattice3 = [[(1,2), (1,5), (2,3), (3,4), (3,6),
             (5,7), (7, 9), (7,8), (6,10), (9,10), (10,11), (10,12)],
            [(2,5), (4,6), (6,9), (8,9), (11,12), (1,3),
            (2,4), (3,10), (2,6), (6,11), (6,12), (9,11),
            (9, 12), (5, 8), (1,7), (7, 10), (5,9)]]
partit = [int(6),int(6)]
Nsites = int( 12)


CsnFourier = CSnGradient(J= [1.0, args.J2], lattice = lattice3, Nsites=Nsites,
                    partit=partit,p=int(6), num_samples =int(1000), max_iter = int(5001), lr=args.lr)


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

L = CsnFourier.Expect_braket
# opt_YJM, opt_H = CsnFourier.CSn_nadam(L, scale=float(1e-2))
optimized_energy, O_gs= CsnFourier.CSn_nadam(L, 'Kagome_12spin', scale=args.noise_scale, use_hessian=False)

# import pandas as pd

# df = pd.DataFrame(opt_energy_list)
# df.to_csv('../data/CQA_J05_Kagome_t.csv')

# O_gs = CsnFourier.Groundstate(opt_YJM, opt_H)
# optimized_energy = CsnFourier.Expect_braket_energy(opt_YJM, opt_H)

print('the optimized ground state: {}'.format(O_gs))
print('------------------------------------')
print('Optimized lowest energy: {}'.format(optimized_energy))
print('-------------------------------------')
print('True Ground State wavefuncion in Sn irrep basis: {}'.format(V_gs))
print('-------------------------------------')
print('The overlap between the optimized state and the ground state: {}'.format(jnp.dot(O_gs,
                                                                               V_gs)))
# print('The distance between the optimized state and the ground state: {}'.format(jnp.linalg.norm(jnp.subtract(V_gs, O_gs))))

'''
The best run for the J2 = 0.5 with learning rate self.rt = 1e-3 and num_samples = 5000 and the overlap with exact =  0.9521776

(p=6) J2 = 0.5 with self.rt = 1e-3 and num_samples = 1000. The overlap with exact = 0.9852500


The best run for the J2 = 0.8 with learning rate self.rt = 2e-3 and num_samples = 1000, with the overlap with the exact = 0.9964648

(p=6) the best run for the J2=0.8 with learning rate self.rt = 2e-3 and num_samples = 1000, the overlap with exact = 0.9967997
'''