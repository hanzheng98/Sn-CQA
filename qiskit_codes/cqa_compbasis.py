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




class CQA(QuantumCircuit):
    '''
    The circuit implementing CQA circuits on the computational 
    basis using the qiskit toolkit. 


    
    '''
    def __init__(
        self, 
        num_sites: int, 
        p: int, 
        heisenberg: Union[PauliOp, PauliSumOp, PauliList, SparsePauliOp], 
        irrep: list[int, int], 
        YJMparams: np.ndarray, 
        Heisparams: np.ndarray,
        num_time_slices: int=1,
        debug: bool=False,
        name: str = 'Sn-CQA',
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
        num_bell = 2*  (self.irrep[0] - abs(self.irrep[0] - self.irrep[1]))
        num_comp = abs(self.irrep[0] - self.irrep[1])
        'computing the state initialization method'
        q_comp = QuantumRegister(num_comp, 'zeros states')
        q_bell = QuantumRegister(num_bell, 'bell' )
        circuit = QuantumCircuit(q_comp, q_bell, name=self.name)
        if self.debug: 
            print('number of bell states: {}'.format(num_bell))
            print('num of the standard zeros states: {}'.format(num_comp))
        for i in range(0, num_bell-1):
            if i %2 ==0:
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
            CQA_layer_circ = self._getCQA_layer(self.YJMparams[p], self.Heisparams[p])
            circuit.append(CQA_layer_circ.to_instruction(), q_comp[:] + q_bell[:])
        self.compose(circuit, qubits=self.qubits, inplace=True)
        

    def _getYJMs(self, yjmcoeffs: np.ndarray) -> PauliSumOp:
        
        YJM_ham = self._getYJM(0)
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
                    YJM_ham += 0.5* (YJM_prod.adjoint() + YJM_prod) * np.around(yjmcoeffs[k, l], decimals=2)
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
        return YJM_ham.reduce()
    

    def _getYJM(self, k:int) -> PauliSumOp:
        if k == 0:
            YJM = self._transpo2pauli(0, 0)
            return YJM 
        elif k ==1: 
            YJM = self._transpo2pauli(0, 1)
            print('---------')
            print(YJM)
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
            coeff = np.array([2.0, 2.0, 2.0, 0.5])
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

    
    def _getCQA_layer(self, yjmparam, heisparam):
        # if self.debug: 
        #     print(yjmparam.shape)
        #     print(self.heisenberg.mul(heisparam))
        YJM_ham = self._getYJMs(yjmparam)
        YJM_evo = YJM_ham.exp_i()
        Heis_evo = (self.heisenberg.mul(heisparam)).exp_i()
        if self.debug: 
            print('coefficients for the YJM Hamilonian: {}'.format(np.imag(YJM_ham.coeffs)))
            # print('YJM Hamiltonian: {}'.format(YJM_ham))
            print('To check that if YJM Hamiltonian is hermitian: {}'.format(YJM_ham.is_hermitian()))
            print('To check that if Heisenberg Hamiltonian is hermitian: {}'.format(self.heisenberg.is_hermitian()))
            # YJM_ham.coeffs = np.real(YJM_ham.coeffs)
        YJM_circuit = PauliTrotterEvolution(
                trotter_mode='trotter',
                reps=self.num_time_slices).convert(YJM_evo).to_circuit()
        Heis_circuit = PauliTrotterEvolution(
                trotter_mode='trotter',
                reps=self.num_time_slices).convert(Heis_evo).to_circuit()
        YJM_qc = QuantumCircuit(self.num_sites, name='exp(iYJM)')
        YJM_qc.append(YJM_circuit.to_instruction(), YJM_qc.qubits)
        Heis_qc = QuantumCircuit(self.num_sites, name='exp(iH_p)')
        Heis_qc.append(Heis_circuit.to_instruction(), Heis_qc.qubits)
        CQA_qc = QuantumCircuit(self.num_sites, name='CQA-Layer')
        CQA_qc.append(YJM_qc.to_instruction(), CQA_qc.qubits)
        CQA_qc.append(Heis_qc.to_instruction(), CQA_qc.qubits)
        # CQA_layer_circ = Heis_circuit.to_circuit(name='exp(iH_p)').compose(YJM_circuit.to_circuit(name='exp(iYJM)'))
        return CQA_qc
    
        
        
    
        






        



