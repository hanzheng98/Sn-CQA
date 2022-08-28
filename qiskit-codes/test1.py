import qiskit 
import numpy as np 
import matplotlib.pyplot as plt

from typing import Optional, Union, List, Callable, Tuple
from qiskit.circuit import QuantumCircuit, QuantumRegister, parameter
from qiskit import transpile
from qiskit.circuit.library.basis_change import QFT
from qiskit.quantum_info import SparsePauliOp, PauliList, Pauli
from qiskit.opflow import X, Y, Z, I, PauliTrotterEvolution
from qiskit.opflow import PauliOp, PauliSumOp
from cqa_compbasis import CQA
import networkx as nx

'''
create some Hamilonnian
'''



def getJ1J2_Ham(J: list, J1_edges: list, J2_edges:list, num_sites: int) -> PauliSumOp:
    xstring = np.zeros(num_sites)
    zstring = np.zeros(num_sites)
    J1_lst = []
    for J1pauli in J1_edges:  
        J1pauli_lst = []
        J1xstring = np.zeros(num_sites)
        J1xstring[J1pauli[0]] = 1
        J1xstring[J1pauli[1]] = 1 
        J1zstring = np.zeros(num_sites)
        J1zstring[J1pauli[0]] = 1
        J1zstring[J1pauli[1]] = 1 
        J1pauli_lst.append(Pauli(zstring, J1xstring))
        J1pauli_lst.append(Pauli(J1zstring, xstring))
        J1pauli_lst.append(Pauli(J1zstring, J1zstring))
        J1_lst.append(PauliSumOp(SparsePauliOp(J1pauli_lst, coeffs=np.array([J[0], J[0], J[0]]))))
    J2_lst = []
    for J2pauli in J2_edges:  
        J2pauli_lst = []
        J2xstring = np.zeros(num_sites)
        J2xstring[J2pauli[0]] = 1
        J2xstring[J2pauli[1]] = 1 
        J2zstring = np.zeros(num_sites)
        J2zstring[J2pauli[0]] = 1
        J2zstring[J2pauli[1]] = 1 
        J2pauli_lst.append(Pauli(zstring, J2xstring))
        J2pauli_lst.append(Pauli(J2zstring, xstring))
        J2pauli_lst.append(Pauli(J2zstring, J2zstring))
        J2_lst.append(PauliSumOp(SparsePauliOp(J2pauli_lst, coeffs=np.array([J[1], J[1], J[1]]))))
    heisenberg = sum(J1_lst) + sum(J2_lst)
    return heisenberg




# graph = nx.generators.lattice.grid_2d_graph(3,3)
# graph = nx.relabel.convert_node_labels_to_integers(graph)
# obs = []
# coeffs = []
# # for edge in graph.edges():
# #     coeffs.extend([1.0, 1.0, 1.0])
# #     obs.extend([X(edge[0]) @ X(edge[1]),
# #                         Y(edge[0]) @ Y(edge[1]),
# #                         (edge[0]) @ Z(edge[1])])


# print(graph.edges)
# nx.draw(graph)
# print(coeffs)
# print(obs)



# hamiltonian_heisenberg = qml.Hamiltonian(coeffs, obs)

if __name__ == "__main__":
    edge_colors = [[[0, 1, 1], [0, 4, 1], [1, 2,1],
               [2, 3, 1], [2,5,1], [4,6,1], [6,8,1], [6,7,1],
              [5,9,1], [8,9,1],[9,10,1], [9,11,1]],
               # J2 terms now for the frustration
               [[1,4,2], [3,5,2], [5,8,2], [7,8,2], [10,11,2], [0,2,2],
               [1,3,2], [2,9,2], [1,5,2], [5,10,2], [5,11,2], [8,10,2],
               [8,11,2], [4,7,2], [0,6,2], [6,9,2], [4,8,2]]]
    # print(edge_colors[0])
    # print(edge_colors[1])
    # lattice = nx.graph()
    # for edgeJ1 in edge_colors[0]:
    #     lattice.add_edge(edgeJ1[0], edgeJ1[1])
    num_sites = 12
    Heisenberg = getJ1J2_Ham([1.0, 0.5], edge_colors[0], edge_colors[1], num_sites)
    print(Heisenberg)
    
    p = 2 # num of alternating layers
    YJMparams = np.random.randn(p, num_sites, num_sites)
    Heisparams = np.random.randn(p)
    irrep = [6, 6]
    Sn_cqa = CQA(num_sites, p, Heisenberg, irrep, YJMparams, Heisparams, debug=False)
    print(Sn_cqa.decompose())
    Sn_cqa_gates = transpile(Sn_cqa, basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'cy', 'cry', 'h', 'ry'])
    print('depth of the sn_cqa_gates: {}'.format(Sn_cqa_gates.depth()))