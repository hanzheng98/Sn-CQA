import torch
import cnine
import Snob2 
from torch.nn import Linear, ModuleList, ModuleDict 

'''

----------------Sn-CQA in Fourier space------------

'''

def get_basis(dim, sample_size, debug=False, state=None):
    '''
    getting the basis from sampling
    '''
    if debug is False: 
        sampling = torch.randint(0, dim, (sample_size, ))
        basis = torch.zeros(dim)
        for sample in sampling:
            sample_basis = torch.zeros(dim)
            sample_basis[sample] = 1.0
            basis += sample_basis
        basis = torch.view_as_complex(torch.stack([basis, torch.zeros_like(basis)], dim=-1))
        return basis / torch.norm(basis, p=2)
    elif debug is True: 
        basis = torch.view_as_complex(torch.stack([state, torch.zeros_like(state)], dim=-1)) 
        return basis 






'''
---------------CQA ansatze---------------

'''

class CQAFourier(torch.nn.Module):

    def __init__(self,J: list[int, int],num_sites:int, p:int, 
                            irrep: list[int, int], lattice: list, ham_scale: float ,device='cpu', debug: bool=False):
        super(CQAFourier, self).__init__()
        self.num_sites = num_sites
        self.p = p 
        self.irrep = irrep 
        self.J = J
        self.dim = Snob2.SnIrrep(self.irrep).get_dim()
        self.group = Snob2.Sn(self.num_sites)
        self.rep = Snob2.SnIrrep(self.irrep)
        self.lattice = lattice
        self.device = device
        self.debug = debug
        self.ham_scale = ham_scale
        YJMs_mat = torch.zeros((self.dim, self.num_sites, self.num_sites), device=self.device)
        for i in range(self.num_sites):
            if i == self.num_sites -1:
                YJM = self._get_YJMs(i, i)
                YJMs_mat[:,i, i] = torch.diag(YJM)
            for j in range(i, self.num_sites):
                YJM = self._get_YJMs(i,j)
                if self.debug:
                    assert torch.allclose(YJM, torch.diag(torch.diag(YJM)), atol=5e-5)
                    # print('--------')
                YJMs_mat[:,i,j] = torch.diag(YJM)
        self.YJMs = YJMs_mat.view(self.dim, self.num_sites * self.num_sites)
        self.Heisenberg = self._Ham_rep()
        self.YJMparams = ModuleList([Linear(self.num_sites * self.num_sites, 
                                self.num_sites * self.num_sites, bias=False) for i in range(self.p)])
        self.Heisparams = ModuleList([Linear(1, 1, bias=False) for i in range(self.p)])


    def forward(self, x: torch.tensor):

        '''
        x is the CQA state -- initialized to be a random Schur states
        '''
        cqa_mat = torch.diag(torch.ones(self.dim)).to(device=self.device)
        cqa_mat = torch.view_as_complex(torch.stack([cqa_mat, torch.zeros_like(cqa_mat)], dim=-1)) #converting to the complex type 
        
        
        # cqa_mat_layer = torch.diag(torch.ones(self.dim)).to(device=self.device)
        # cqa_mat_layer = torch.stack([cqa_mat_layer, torch.zeros_like(cqa_mat_layer)], dim=-1) 
        for YJMneural, Heisneural in zip(self.YJMparams, self.Heisparams):
            YJM_ham = YJMneural(self.YJMs)
            YJM_ham = (YJM_ham / torch.norm(YJM_ham,p=2, dim=0)).sum(dim=-1)
            YJM_ham = torch.view_as_complex(torch.stack([torch.zeros_like(YJM_ham), YJM_ham], dim=-1)).to(device=self.device)
            YJM_evo = torch.diag(torch.exp(YJM_ham * self.ham_scale)) ## YJM_ham.shape (dim, numterms)
            YJM_evo = YJM_evo / torch.norm(YJM_evo, p=2)
            if self.debug: 
                # print('norm of the YJM unitary: {}'.format(torch.norm(YJM_evo, p=2)))
                if torch.allclose(torch.tensor(1.0), torch.norm(YJM_evo, p=2), atol=1e-5) is False:
                    print('norm of unitary YJM: {}'.format(torch.norm(YJM_evo, p=2)))
                assert torch.allclose(torch.real(YJM_ham), torch.zeros(YJM_ham.shape))
            # YJM_evo = torch.matrix_exp(YJM_ham)
            Heis_ham = Heisneural(self.Heisenberg.unsqueeze(dim=-1)).squeeze() # heisenberg.shape 
            Heis_ham = torch.view_as_complex(torch.stack([torch.zeros_like(Heis_ham), Heis_ham], dim=-1)).to(device=self.device)
                # print('the shape of the Heisenberg hamiltonian: {}'.format(Heis_ham.shape))

                # print(Heis_ham.shape)
            Heis_evo = torch.matrix_exp(Heis_ham * self.ham_scale)
            Heis_evo = Heis_evo / torch.norm(Heis_evo, p=2)
            if self.debug:
                # print('norm of the heisenberg unitary: {}'.format(torch.norm(Heis_evo, p=2)))
                if torch.allclose(torch.tensor(1.0), torch.norm(Heis_evo, p=2), atol=1e-5) is False: 
                    print('the norm of the unitary heisenberg: {}'.format(torch.norm(Heis_evo))) 
                # print('the shape of the Heisenberg hamiltonian: {}'.format(Heis_ham.shape))
                assert torch.allclose(torch.real(Heis_ham), torch.zeros(Heis_ham.shape))
                # print('norm of the imaginary part: {}'.format(torch.norm(torch.imag(Heis_evo))))
            cqa_mat_layer = torch.matmul(Heis_evo, YJM_evo)
        cqa_mat = torch.matmul(cqa_mat_layer, cqa_mat)
        
        x = torch.matmul(cqa_mat, x) # return a approximate states for energy minimization
        return x /torch.norm(x, p=2)



    def _get_YJMs(self, k, l):
        # compute X_k X_l for the YJM elements and by default X_1 = e
        Xkl= torch.zeros((self.dim, self.dim), device=self.device)
        if k == l == 1:
            return torch.diag(torch.ones(self.dim))
        for i in range(1, max(k, l)):
            # pi = self.group('({}, {})'.format(i, max(k, l)))
            pi = self.rep.transp(i, max(k, l)).torch()
            if min(k, l) == 1:
                Xkl += pi
            else:
                for j in range(1, min(k, l)):
                    pj = self.rep.transp(j, min(k,l)).torch()
                    pij = torch.matmul(pi, pj)
                    Xkl = Xkl+  pij
        return Xkl
    

    def _Ham_rep(self):
        rep_mat0 = ((-1.0 * len(self.lattice[0])) / 2) * torch.diag(torch.ones(self.dim)) 
        for st in self.lattice[0]:
            #         print(H[ls[0]])
            # print(st)
            rep_st = self.rep.transp(st[0], st[1]).torch()
            #         print(rep_st)
            rep_mat0 = rep_mat0 + rep_st
        rep_mat0 = torch.mul(self.J[0] / 2, rep_mat0)
        if float(self.J[1]) == float(0):
            return rep_mat0
        else:
            # rep_mat1 = np.multiply(np.multiply(-1.0, len(self.lattice[1]) / 2, dtype='float64'),
            #                        np.diag(np.ones(self.dim)))
            rep_mat1 = ((-1.0 * len(self.lattice[1])) / 2) * torch.diag(torch.ones(self.dim))  
            for st in self.lattice[1]:
                rep_st = self.rep.transp(st[0], st[1]).torch()
                #         print(rep_st)
                rep_mat1 += rep_st
            rep_mat1 = torch.mul(self.J[1] / 2, rep_mat1)
            rep_mat_H = rep_mat0 + rep_mat1
            return rep_mat_H

    

if __name__ == '__main__':

    lattice = [[(1,2), (1,5), (2,3), (3,4), (3,6),
             (5,7), (7, 9), (7,8), (6,10), (9,10), (10,11), (10,12)],
            [(2,5), (4,6), (6,9), (8,9), (11,12), (1,3),
            (2,4), (3,10), (2,6), (6,11), (6,12), (9,11),
            (9, 12), (5, 8), (1,7), (7, 10), (5,9)]]
    J = [1.0, 0.5]
    num_sites = 12
    irrep = [6, 6]
    p = 4
    CQA = CQAFourier(J, num_sites, p, irrep, lattice, debug=True)
    print('----get the heisneberg hamiltonian: {}'.format(CQA.Heisenberg.shape))
    EDvalues, EDvectors = torch.linalg.eig(CQA.Heisenberg)
    EDvector = EDvectors[torch.argmin(torch.real(EDvalues))]
    EDvalue = EDvalues[torch.argmin(torch.real(EDvalues))]
    print('the exact ground state energy: {}'.format(EDvalue))
    print('-----')

    init_basis = get_basis(CQA.dim, 200)
    print('---the initialization basis state: {}'.format(init_basis.shape))
    print('--------------------')
    trial_state = CQA(init_basis)
    print('-----the returned trial state: {}'.format(trial_state.shape))


    