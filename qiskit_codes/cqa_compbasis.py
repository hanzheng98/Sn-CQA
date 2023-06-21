from unicodedata import name
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


class CQA(QuantumCircuit):
    '''
    The circuit implementing CQA circuits on the computational 
    basis using the qiskit toolkit. 

 
    '''
    def __init__(
        self, 
        num_sites: int, 
        p: int, 
        irrep: list[int, int], 
        YJMparams: np.ndarray, 
        Heisparams: np.ndarray,
        heisenberg: Optional[Union[PauliOp, PauliSumOp, PauliList, SparsePauliOp]]=None, 
        num_time_slices: int=1,
        debug: bool=False,
        name: str = 'Sn-CQA',
        mode: Optional[str]='all', 
        method: Optional[str] ='Coxeter'
    ) -> QuantumCircuit: 
        self.num_sites = num_sites
        self.p = p 
        self.heisenberg = heisenberg
        self.debug = debug
        self.irrep = irrep
        self.YJMparams = YJMparams
        self.Heisparams = Heisparams
        self.num_time_slices = num_time_slices
        self.name = name
        self.mode = mode
        self.method = method
        num_bell = 2*  (self.irrep[0] - abs(self.irrep[0] - self.irrep[1]))
        num_comp = abs(self.irrep[0] - self.irrep[1])
        'computing the state initialization method'
        q_comp = QuantumRegister(num_comp, 'zeros states')
        q_bell = QuantumRegister(num_bell )
        circuit = QuantumCircuit(q_comp, q_bell, name=self.name)
        if self.debug: 
            print('number of bell states: {}'.format(num_bell))
            print('num of the standard zeros states: {}'.format(num_comp))
        for i in range(0, num_bell-1):
            if i %2 ==0:
                circuit.x(q_bell[i])
                circuit.x(q_bell[i+1])
                circuit.h(q_bell[i])
                circuit.cnot(q_bell[i], q_bell[i+1])
        'Get the CQA ansatze in the computational basis'
        if self.debug:
            print('printing the state initailization')
            print(circuit)
            print('----------------')
            print('print our problem hamiltonian: {}'.format(self.heisenberg))
        super().__init__(*circuit.qregs, name=circuit.name)
        for p in range(self.p): 
            CQA_layer_circ = self._getCQA_layer(self.YJMparams[p], self.Heisparams[p], p)
            circuit.append(CQA_layer_circ.to_instruction(), q_comp[:] + q_bell[:])
        self.compose(circuit, qubits=self.qubits, inplace=True)
        

    def _getYJMs(self, yjmcoeffs: np.ndarray) -> PauliSumOp:

        
        YJM_ham = self._getYJM(0)
        if self.mode == 'all':
            for k in range(self.num_sites):
                YJM_k = self._getYJM(k)
                if self.debug: 
                    print('--------')
                    assert (np.isreal(yjmcoeffs)).all()
                # assert YJM_k.is_hermitian() is True
                for l in range(self.num_sites):
                    if l >= k:
                        YJM_l = self._getYJM(l)
                        YJM_prod = YJM_k @ YJM_l
                        YJM_ham += 0.5* (YJM_prod.adjoint() + YJM_prod) * np.around(yjmcoeffs[k, l], decimals=10)
                        # if self.debug: 
                        #     print('-----({}, {})-----'.format(k, l))
                            # YJM_prod = YJM_k @ YJM_l 
                            # YJM_prod.coeffs = np.real(YJM_prod.coeffs)
                            # print(np.imag(YJM_k.coeffs))
                            # print(np.imag(YJM_l.coeffs))
                            # print(np.imag(YJM_ham.coeffs))
                            # assert (np.isreal(YJM_ham.coeffs)).all()
                            # print(np.imag((YJM_k @ YJM_l).coeffs))
                            # assert np.isreal((YJM_k @ YJM_l).coeffs).all()
                        #     print('----------------')
                        # assert YJM_l.is_hermitian() is True
                        # YJM_prod = YJM_k @ YJM_l
                        # YJM_ham += 0.5* (YJM_prod.adjoint() + YJM_prod) * yjmcoeffs[k, l]
            YJM_ham = YJM_ham - self._getYJM(0)
            if self.debug: 
                # coeffs = np.around(YJM_ham.coeffs, decimals=2)
                # YJM_ham = 0.5 * (YJM_ham.adjoint() + YJM_ham)
                print('first YJM elements: {}'.format(np.imag(YJM_ham.coeffs)))
                # assert (np.isreal(YJM_ham.coeffs).all())
                # assert YJM_ham.is_hermitian() is True
        elif self.mode =='first-order':
            yjmcoeffs =  yjmcoeffs.flatten()
            for k in range(self.num_sites):
                YJM_k = self._getYJM(k)
                YJM_ham += YJM_k * np.around(yjmcoeffs[k], decimals=10)
            YJM_ham = YJM_ham - self._getYJM(0)
        elif self.mode == 'random':
            yjmcoeffs =  yjmcoeffs.flatten()
            # selection = np.random.randint(0, self.num_sites, int(np.floor(self.num_sites / 3)))
            selection = [i for i in range(self.num_sites)]
            for k in range(len(selection)): 
                # print(f'i {i} num {num}')
                YJM_k = self._getYJM(selection[k])
                # print(f'YJM_k {YJM_k}')
                YJM_ham +=  YJM_k * np.around(yjmcoeffs[k], decimals=10)
            YJM_ham = YJM_ham - self._getYJM(0)
        else: 
            print('CQA module currently not implemented')
            raise NotImplementedError
        return YJM_ham.reduce()
    

    def _getYJM(self, k:int) -> PauliSumOp:
        if k == 0:
            YJM = self._transpo2pauli(0, 0)
            return YJM 
        elif k ==1: 
            YJM = self._transpo2pauli(0, 1)
            # print('---------')
            # print(YJM)
            return YJM

        elif k >1:
            YJM = self._transpo2pauli(0, k)
            for i in range(1, k): 
                YJM += self._transpo2pauli(i, k)
            return YJM


    def _transpo2pauli(self,i:int, j:int) -> PauliSumOp:
        '''
        implement the pauli representation of the transposition (i, j)

        return: Paulisumop 
        '''

        pauli_lst = []
        z0 = np.zeros(self.num_sites)
        x0 = np.zeros(self.num_sites)
        x = np.zeros(self.num_sites)
        z = np.zeros(self.num_sites)
        if i != j: 
            x[i] = 1
            x[j] = 1
            z[i] = 1
            z[j] = 1
            pauli_lst.append(Pauli((z0, x)))
            pauli_lst.append(Pauli((z, x0)))
            pauli_lst.append(Pauli((z, x)))
            pauli_lst.append(Pauli((z0, x0)))
            coeff = np.array([1.0, 1.0, 1.0, 0.5])
            transpo_pauli = PauliSumOp(SparsePauliOp(PauliList(pauli_lst), coeffs=coeff, ignore_pauli_phase=True))
            if self.debug:
                print('---transpo_pauli-----')
                print(np.imag(transpo_pauli.coeffs))
            # assert transpo_pauli.is_hermitian() is True
            return transpo_pauli
        else: 
            identity = PauliSumOp(SparsePauliOp(PauliList(Pauli((z0, x0))), ignore_pauli_phase=True))
            # identity = 0.5 * (identity.adjoint() + identity)
            # print('---------')
            # print(identity)
            # assert identity.is_hermitian() is True 
            return identity

    
        
    
    def _getCQA_layer(self, yjmparam, heisparam, layer=Optional[int]):
        # if self.debug: 
        #     print(yjmparam.shape)
        #     print(self.heisenberg.mul(heisparam))
        YJM_ham = self._getYJMs(yjmparam)
        YJM_evo = YJM_ham.exp_i()
        if self.method == 'Hamiltonian': 
            Heis_evo = (self.heisenberg.mul(heisparam[0])).exp_i()
            if self.debug: 
                print('coefficients for the YJM Hamilonian: {}'.format(np.imag(YJM_ham.coeffs)))
                # print('YJM Hamiltonian: {}'.format(YJM_ham))
                print('To check that if YJM Hamiltonian is hermitian: {}'.format(YJM_ham.is_hermitian()))
                print('To check that if Heisenberg Hamiltonian is hermitian: {}'.format(self.heisenberg.is_hermitian()))
                # YJM_ham.coeffs = np.real(YJM_ham.coeffs)
            # YJM_circuit = PauliTrotterEvolution(
            #         trotter_mode='trotter',
            #         reps=self.num_time_slices).convert(YJM_evo).to_circuit()
            # Heis_circuit = PauliTrotterEvolution(
            #         trotter_mode='trotter',
            #         reps=self.num_time_slices).convert(Heis_evo).to_circuit()
            # YJM_qc = QuantumCircuit(self.num_sites, name='exp(iYJM)')
            # YJM_qc.append(YJM_circuit.to_instruction(), YJM_qc.qubits)
            # Heis_qc = QuantumCircuit(self.num_sites, name='exp(iH_p)')
            # Heis_qc.append(Heis_circuit.to_instruction(), Heis_qc.qubits)
            # CQA_qc = QuantumCircuit(self.num_sites, name='CQA-Layer')
            # CQA_qc.append(YJM_qc.to_instruction(), CQA_qc.qubits)
            # CQA_qc.append(Heis_qc.to_instruction(), CQA_qc.qubits)
            # CQA_layer_circ = Heis_circuit.to_circuit(name='exp(iH_p)').compose(YJM_circuit.to_circuit(name='exp(iYJM)'))
        elif self.method == 'Coxeter': 
            coxeters = self._transpo2pauli(0, 0)
            if layer % 2 ==0: 
                for i in range(self.num_sites -1): 
                    if i % 2 ==0: 
                        coxeters += self._transpo2pauli(i, i+1).mul(heisparam[int(i/2)])
            if layer %2 ==1:
                for i in range(self.num_sites -1): 
                    if i % 2 ==1: 
                        coxeters += self._transpo2pauli(i, i+1).mul(heisparam[int((i-1)/2)])
            coxeters = coxeters - self._transpo2pauli(0, 0)
            Heis_evo = coxeters.exp_i()
        else: 
            raise NotImplementedError
        
        YJM_circuit = PauliTrotterEvolution(
                    trotter_mode='trotter',
                    reps=self.num_time_slices).convert(YJM_evo).to_circuit()
        Heis_circuit = PauliTrotterEvolution(
                trotter_mode='trotter',
                reps=self.num_time_slices).convert(Heis_evo).to_circuit()
        YJM_qc = QuantumCircuit(self.num_sites, name=f'mixer')
        YJM_qc.append(YJM_circuit.to_instruction(), YJM_qc.qubits)
        Heis_qc = QuantumCircuit(self.num_sites, name=f'Commuting SWAPs')
        Heis_qc.append(Heis_circuit.to_instruction(), Heis_qc.qubits)
        CQA_qc = QuantumCircuit(self.num_sites, name='CQA-Layer')
        CQA_qc.append(YJM_qc.to_instruction(), CQA_qc.qubits)
        CQA_qc.append(Heis_qc.to_instruction(), CQA_qc.qubits)

        return CQA_qc
    
        
        
    
        






        




