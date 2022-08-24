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
    ): 
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
        q_comp = QuantumRegister(num_comp, '0 states')
        q_bell = QuantumRegister(num_bell, 'bell' )
        circuit = QuantumCircuit(q_comp, q_bell, name='init')
        for i in range(1, num_bell + 1):
            if i %2 ==1:
                circuit.h(q_bell[i])
                circuit.cnot(q_bell[i], q_bell[i+1])
        'Get the CQA ansatze in the computational basis'
        for p in range(self.p): 
            CQA_layer_circ = self._getCQA(self.YJMparams[p], self.Heisparams[p])
            circuit.compose(CQA_layer_circ.to_circuit(), qubits= self.qubits, inplace=True)
        super().__init__(*self.circuit.qregs, name=circuit.name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)
        

    def _getYJMs(self, yjmcoeffs: np.ndarray) -> PauliSumOp:
        
        YJM_ham = self._getYJM(0)
        for k in range(self.num_sites):
            YJM_k = self._getYJMs(k)
            for l in range(self.num_sites):
                YJM_l = self._getYJMs(l)
                YJM_ham += YJM_k @ YJM_l * yjmcoeffs[k, l]
        YJM_ham = YJM_ham - self._getYJM(0)
        return YJM_ham 
    

    def _getYJM(self, k:int) -> list:
        if k == 0:
            YJM = self._transpo2pauli(0, 0)
            return YJM 
        elif k ==1: 
            YJM = self._transpo2pauli(0, 1)
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
        x[i] = 1
        x[j] = 1
        z[i] = 1
        z[j] = 1
        pauli_lst.append(Pauli((z0, x)))
        pauli_lst.append(Pauli((z, x0)))
        pauli_lst.append(Pauli((z, x)))
        pauli_lst.append(Pauli((z0, x0)))
        coeff = np.array([2.0, 2.0, 2.0, 0.5])
        transpo_pauli = PauliSumOp(SparsePauliOp(pauli_lst, coeffs=coeff))
        return transpo_pauli

    
    def _getCQA_layer(self, yjmparam, heisparam):
        YJM_ham = self._getYJMs(yjmparam)
        YJM_evo = YJM_ham.exp_i()
        Heis_evo = (heisparam * self.heisenberg).exp_i()
        YJM_circuit = PauliTrotterEvolution(
                trotter_mode='trotter',
                reps=self.num_time_slices).convert(YJM_evo)
        Heis_circuit = PauliTrotterEvolution(
                trotter_mode='trotter',
                reps=self.num_time_slices).convert(Heis_evo)
        CQA_layer_circ = Heis_circuit.compose(YJM_circuit)
        return CQA_layer_circ
    
        
        
    
        






        






class PhaseEstimation(QuantumCircuit):
   

    def __init__(
        self,
        num_evaluation_qubits: int,
        unitary: Optional[QuantumCircuit]=None,
        iqft: Optional[QuantumCircuit] = None,
        name: str = "QPE",
        hamiltonian: Optional[Union[PauliOp, PauliSumOp, PauliList, SparsePauliOp]] = None,
        debug: bool= False,
        evolution_time: float=2 * np.pi,
        num_time_slices: int = 1,
    ) -> None:
        """
        Args:
            num_evaluation_qubits: The number of evaluation qubits.
            unitary: The unitary operation :math:`U` which will be repeated and controlled.
            iqft: A inverse Quantum Fourier Transform, per default the inverse of
                :class:`~qiskit.circuit.library.QFT` is used. Note that the QFT should not include
                the usual swaps!
            name: The name of the circuit.

        .. note::

            The inverse QFT should not include a swap of the qubit order.

        Reference Circuit:
            .. jupyter-execute::
                :hide-code:

                from qiskit.circuit import QuantumCircuit
                from qiskit.circuit.library import PhaseEstimation
                import qiskit.tools.jupyter
                unitary = QuantumCircuit(2)
                unitary.x(0)
                unitary.y(1)
                circuit = PhaseEstimation(3, unitary)
                %circuit_library_info circuit
        """
        # super(HHL, self).__init__(**kwargs)
        self.debug = debug
        self.evolution_time = evolution_time
        self.hamiltonian = hamiltonian
        self.num_time_slices = num_time_slices
        if self.hamiltonian is None:
            qr_eval = QuantumRegister(num_evaluation_qubits, "eval")
            qr_state = QuantumRegister(unitary.num_qubits, "q")
            circuit = QuantumCircuit(qr_eval, qr_state, name=name)
        else:
            qr_eval = QuantumRegister(num_evaluation_qubits, "eval")
            qr_state = QuantumRegister(self.hamiltonian.num_qubits, "q")
            circuit = QuantumCircuit(qr_eval, qr_state, name=name)
        if self.debug:
            copy_circuit = circuit.copy(name='copy')
            # copy_circuit.append(.to_instruction(), qr_state)
            print('circuit depth without the control: {}'.format(
                transpile(copy_circuit, basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'cy', 'cry', 'h', 'ry']).depth()))

        if iqft is None:
            iqft = QFT(num_evaluation_qubits, inverse=True, do_swaps=False).reverse_bits()
            if self.debug:
                decomp_iqft = transpile(iqft, basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'cy', 'cnot', 'cry', 'h'])
                print('circuit depth for the inverse QFT: {}'.format(decomp_iqft.depth()))
        circuit.h(qr_eval)  # hadamards on evaluation qubits
        if self.hamiltonian is None:
            for j in range(num_evaluation_qubits):  # controlled powers
                # evo_time = unitary.evolution_time * (2**j)
                unitary.evolution_time = self.evolution_time * (2**j)
                circuit.compose(unitary.control(), qubits=[j] + qr_state[:], inplace=True)
                if self.debug:
                    print('check the power if correct: {}'.format(unitary.evolution_time / (2**j)))
                    decomp_qc = transpile(circuit, basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'cy', 'cnot', 'cry', 'h'])
                    print('circuit depth after applying controlled unitary on {}th-qubit: {}'.format(j, decomp_qc.depth()))
        else:
            # use Hamiltonian instead of
            for j in range(num_evaluation_qubits):
                evo_time = self.evolution_time * (2**j)
                unitary = self.lcu_evo2(evo_time, debug=True)
                circuit.compose(unitary.control(), qubits=[j] + qr_state[:], inplace=True)
                if self.debug:
                    print('check the power if correct: {}'.format(evo_time / (2**j)))
                    decomp_qc = transpile(circuit, basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'cy', 'cnot', 'cry', 'h'])
                    print('circuit depth after applying controlled unitary after {}th-qubit: {}'.format(j, decomp_qc.depth()))

        circuit.compose(iqft, qubits=qr_eval[:], inplace=True)  # final QFT

        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)

    def lcu_evo2(self, evolution_time, debug=False, trotter=True):
        lcu_ham = PauliSumOp(self.hamiltonian)
        evolution_op = (evolution_time * lcu_ham).exp_i()
        if trotter:
            matrix_circuit = PauliTrotterEvolution(
                trotter_mode='trotter',
                reps=self.num_time_slices).convert(evolution_op)
        else:
            raise NotImplementedError
        if debug:
            # print('evolution_time: {}'.format(matrix_circuit.to_circuit().evolution_time))
            matrix_circuit = matrix_circuit.to_circuit()
            decomp_qc = transpile(matrix_circuit,
                                  basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'cy', 'cry', 'h', 'crz', 'crx', 'ch', 'cp'])
            print('Hamiltonian Simulation depth: {}'.format(decomp_qc.depth()))
            return matrix_circuit
        else:
            return matrix_circuit.to_circuit()