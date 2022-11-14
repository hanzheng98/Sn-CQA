from unicodedata import name
import qiskit 
import numpy as np 
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Callable, Tuple
import pennylane as qml 
import argparse
import networkx as nx
from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel
from tqdm import tqdm
import jax 
import jax.numpy as jnp
# import pennylane.numpy as pnp 

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--p', type=int, default=2)
parser.add_argument('--irrep', type=list, default=[6,6])
parser.add_argument('--num_qubits', type=int, default=12)
parser.add_argument('--num_yjms', type=int, default=1)
parser.add_argument('--trotter_slice', type=int, default=1)
parser.add_argument('--lattice_size', type=list, default=[3,4])
parser.add_argument('--J', type=list, default=[1.0, 1.0])
parser.add_argument('--iterations', type=int, default=10)
parser.add_argument('--TOKEN', type=str, default="4c84e8146c0def626bb384424f78598ee055148df74863d4cbad29fb99d5e9cd4808077115561b6490f6dc4a0f8014b24fe5985003cdbb88c17a51c80c55279a")
parser.add_argument('--backend', type=str, default='ibm_perth')
parser.add_argument('--device', type=int, default=0) # 0 for classical simulation; 1 for ibm_qasm 2 for real quantum machine
parser.add_argument('--num_layer', type=int, default=1)
args = parser.parse_args()
print(args)

if args.device ==0:
    provider = qiskit.IBMQ.enable_account(args.TOKEN, hub='ibm-q-startup', group='qbraid', project='main')
    backend = provider.get_backend(args.backend)
    noise_model = NoiseModel.from_backend(backend)  
    dev_mu = qml.device('qiskit.aer', wires = args.num_qubits, noise_model=noise_model)
else:
    provider = qiskit.IBMQ.enable_account(args.TOKEN, hub='ibm-q-startup', group='qbraid', project='main')

    if args.device ==1:
        # backend = provider.get_backend(args.backend)
        # noise_model = NoiseModel.from_backend(backend) 
        dev_mu = qml.device('qiskit.ibmq', wires=args.num_qubits, backend='ibmq_qasm_simulator', provider=provider,
                        shots=2000)
    elif args.device ==2:
        dev_mu = qml.device('qiskit.ibmq', wires=args.num_qubits, backend=args.backend, provider=provider,
                        shots=2000) 



'''
------------------------------------------
Defining the Lattice Hamiltonian 
------------------------------------------
'''

def getHam_square(lattice_size: Union[int, list], J: Optional[Union[list, int]]):
    '''
    Build the Heisenberg Hamiltonian 
    '''
    graph = nx.generators.lattice.grid_2d_graph(lattice_size[0],lattice_size[1])
    graph = nx.relabel.convert_node_labels_to_integers(graph)
    obs = []
    coeffs = []
    for edge in graph.edges():
        coeffs.extend([1.0, 1.0, 1.0])
        obs.extend([qml.PauliX(edge[0]) @ qml.PauliX(edge[1]),
                            qml.PauliY(edge[0]) @ qml.PauliY(edge[1]),
                            qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])])
    hamiltonian_heisenberg = qml.Hamiltonian(coeffs, obs) * J[0]
    return hamiltonian_heisenberg




'''
------------------------------------------
Defining the variational circuit 

The Special class of CQA: Using radnomly picked YJMs and coxeters
------------------------------------------
'''

# state preparation:

def state_init(irrep: list):

    num_bell = 2*  (irrep[0] - abs(irrep[0] - irrep[1]))
    # num_comp = abs(irrep[0] - irrep[1])
    for i in range(0, num_bell-1):
        if i %2 ==0:
            qml.Hadamard(wires=i)
            qml.CNOT(wires=[i,i+1]) 

# Coxeter generators 

def coxeters(heis_params: jnp.array, layer:int): 
    '''
    if p is even, return evens 
    if p is odd, return odds
    '''
    if layer % 2 == 0: 
        for j, i  in enumerate(range(args.num_qubits)):
            if i % 2 ==0:
                # print(f'j, i: {j, i}')
                swap = _swap2pauli(i, i+1, mode='ham')
                qml.ApproxTimeEvolution(swap, heis_params[int(j/2)], args.trotter_slice)
    elif layer % 2==1:
        for j, i in enumerate(range(args.num_qubits)):
            if i % 2 ==1:
                swap = _swap2pauli(i, i+1, mode='ham')
                qml.ApproxTimeEvolution(swap, heis_params[int((j-1)/2)], args.trotter_slice)


# Do the YJM coexter(only first order so far on Pennylane)

def _swap2pauli(i: int, j:int, mode='ham'): 
    '''
    using the formula: SWAP(i j) = Si . Sj + 1/2 I (without I ) 
    '''
    if mode=='ham':
        coeff = np.ones(3)
        hamiltonian = [qml.PauliX(wires=i) @qml.PauliX(wires=j), 
                    qml.PauliY(wires=i) @qml.PauliY(wires=j),
                    qml.PauliZ(wires=i) @qml.PauliZ(wires=j) ]
        return qml.Hamiltonian(coeff, hamiltonian)
    elif mode=='list':
        hamiltonian = [qml.PauliX(wires=i) @qml.PauliX(wires=j), 
                    qml.PauliY(wires=i) @qml.PauliY(wires=j),
                    qml.PauliZ(wires=i) @qml.PauliZ(wires=j) ]
        return hamiltonian
        


def _get_YJM(idx:int):
    '''
    get YJM elements for a given index
    '''
    if idx == 0:
        print('too trivial choice')
        raise NotImplementedError
    elif idx ==1: 
        YJM = _swap2pauli(0,1, mode='ham')
        # print('---------')
        # print(YJM)
        return YJM
    else:
        swaps = []
        # yjm_lst = _swap2pauli(0, 1, mode='list')
        for i in range(idx):
            swaps.append(_swap2pauli(i, idx, mode='list'))
            # YJM += qml.SWAP(wires=[i, idx])
        # print(swaps)
        flat_yjm_lst = [item for sublist in swaps for item in sublist]
        # print(qml.Hamiltonian([1.0], [swaps[0]]))
        return qml.Hamiltonian(np.ones(len(flat_yjm_lst)), flat_yjm_lst)


def yjm_gates(yjm_params: jnp.array):
    # num_yjms = int(np.floor(num_qubits /3))
    selection = np.random.randint(1, args.num_qubits, args.num_yjms)
    for i, sel in enumerate(selection):
        YJM = _get_YJM(sel)
        qml.ApproxTimeEvolution(YJM, yjm_params[i], args.trotter_slice)



# Defining the CQA layer now 

def cqa_layers(params_dict:dict): 
    for layer in range(args.p):
        coxeters(params_dict['Heis'][layer], layer=layer)
        yjm_gates(params_dict['YJM'][layer])
    # hamiltonian = getHam_square(args.lattice_size, args.J)
    # return qml.expval(hamiltonian)






'''
------------------------------------------
Measuring wrt. the Heisenberg Hamiltonian 
------------------------------------------
'''
dev = qml.device("default.qubit", wires=args.num_qubits)

@qml.qnode(dev, interface='jax')
def cqa_circuit(params_dict: dict):
    state_init(args.irrep)
    cqa_layers(params_dict)
    hamiltonian = getHam_square(args.lattice_size, args.J)
    return qml.expval(hamiltonian)
    # return qml.expval(qml.PauliZ(wires=0))


def train(loss): 
    '''
    update the gradient via jax 
    '''

    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    params_dict_init = {'YJM': jax.random.uniform(key1, (args.p, args.num_yjms)),
                    'Heis': jax.random.uniform(key2, (args.p,int(np.ceil(args.num_qubits/2))))}
    loss_history, grad_history, param_history = [], [], [params_dict_init]
    # print('-------drawing the circuit--------')
    # drawer = qml.draw(cqa_circuit)
    # print(drawer(params_dict_init))
    # print(iteration)
    for it in tqdm(range(1, args.iterations + 1)): 
        p_dict = param_history[-1]
        print('p_dict | YJM: {} | Heis: {}'.format(p_dict['YJM'], p_dict['Heis']))
        loss, gradient = jax.value_and_grad(loss, argnums=0)(p_dict)
        print(f'gradient: {gradient}')
        print("Step {:3d}   Cost_L = {:9.7f}".format(it, loss))
        updated_p_dict = {'YJM': p_dict['YJM'] - args.lr * gradient['YJM'],
                        'Heis': p_dict['Heis'] - args.lr * gradient['Heis']}
        param_history.append(updated_p_dict)
        loss_history.append(loss)
        grad_history.append(gradient)
        if jnp.linalg.norm(gradient['YJM']) / gradient['YJM'].size < 1e-4:
            print(gradient['YJM'])
            break
    return loss_history, param_history, grad_history
            
    # loss = cqa_circuit(para)
    

def main():
    # edge_colors = [[[0, 1, 1], [0, 4, 1], [1, 2,1],
    #            [2, 3, 1], [2,5,1], [4,6,1], [6,8,1], [6,7,1],
    #           [5,9,1], [8,9,1],[9,10,1], [9,11,1]],
    #            # J2 terms now for the frustration
    #            [[1,4,2], [3,5,2], [5,8,2], [7,8,2], [10,11,2], [0,2,2],
    #            [1,3,2], [2,9,2], [1,5,2], [5,10,2], [5,11,2], [8,10,2],
    #            [8,11,2], [4,7,2], [0,6,2], [6,9,2], [4,8,2]]]
    # print(edge_colors[0])
    # print(edge_colors[1])
    # lattice = nx.graph()
    # for edgeJ1 in edge_colors[0]:
    #     lattice.add_edge(edgeJ1[0], edgeJ1[1])

    # Heisenberg = getJ1J2_Ham([1.0, 0.5], edge_colors[0], edge_colors[1], args.num_qubits)
    # print(Heisenberg)
    
    # p = 2 # num of alternating layers
    # YJMparams = np.random.randn(p, args.num_qubits, args.num_qubits)
    # Heisparams = np.random.randn(p)
    # irrep = [6, 6]
    # Sn_cqa = CQA(args.num_qubits, p, Heisenberg, irrep, YJMparams, Heisparams, debug=False, mode='random')
    # # print(Sn_cqa)
    # print(Sn_cqa.decompose())
    # Sn_cqa_gates = transpile(Sn_cqa, basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'cy', 'cry', 'h', 'ry'])
    # print('depth of the sn_cqa_gates: {}'.format(Sn_cqa_gates.depth()))
    # print('-----------------------')

    # params_dict = {'YJM': YJMparams, 'Heis': Heisparams}
    # # hamiltonian = getHam_square(args.lattice_size, args.J)
    # expectation = cqa_circuit(params_dict, Heisenberg )
    # print(f'expectation vlaue for the CQA in pennylane: {expectation}')
    print('---------variational phase-----------')
    loss_history, param_history, grad_history = train(cqa_circuit)
    plt.style.use("seaborn")
    plt.plot(loss_history, "g")
    plt.ylabel("Cost function")
    plt.xlabel("Optimization steps")
    plt.show()
    plt.savefig('CQA Training') 

if __name__ == '__main__':
    main()