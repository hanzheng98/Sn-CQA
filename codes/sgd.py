import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('JAX SGD')
from FouierFilters2 import FourierFilters
from jax import random
from jax import grad, jit, vmap
# import tensorflow_quatum as tfq
from jax import custom_jvp

class CSnGradient(FourierFilters):
    '''
    Standard SGD method implemented using JAX autodifferentiation megthd

    Unsurpervised method for the minilization of the GS energy for each Sn irrep

    Wirtten in the language environment SAGE9.2, supported SageMath environment

    '''

    def __init__(self, lr=float(2e-3), max_iter=int(1001), gamma=float(0.95), num_samples=None, quantumnoise = False, **kwargs):
        super(CSnGradient, self).__init__(**kwargs)
        self.lr = lr
        self.max_iter = max_iter
        self.gamma = gamma
        self.sampling = np.random.randint(0, high=self.dim, size=num_samples)
        num_trans = jnp.add(len(self.lattice[0]), len(self.lattice[1]))
        # trans_const = jnp.multiply(num_trans, jnp.ones(self.dim), dtype='complex128')
        self.num_trans = num_trans
        self.quantumnoise = quantumnoise



    def random_params(self, scale=float(1e-2)):
        length = jnp.add(jnp.multiply(jnp.multiply(self.Nsites, self.Nsites), self.p), self.p)
        # print(length)
        return jnp.multiply(scale, random.normal(random.PRNGKey(self.p), (length,)))

    def random_params2(self, scale=float(1e-2)):
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

    # @CSn_VStates.defjvp
    # def CSn_Ansazte_jvp(self,primals, tangents):
    #     '''
    #
    #     Not for current use as still working on modify the custom derivative rules
    #
    #     :param primals:
    #     :param tangents:
    #     :return:
    #     '''
    #     YJMparams, Hparams = primals
    #     YJMparams_dot, Hparams_dot = tangents
    #     primal_out = self.CSn_VStates(YJMparams, Hparams)
    #     tangent_out = jnp.zeros(self.dim)
    #     for i in range(self.p):
    #         tangent_out += self.CSn_Ansazte_jvp_aux([int(1), int(1), i], i, YJMparams, Hparams, opt='H')
    #         for k, l in zip(range(1,self.Nsites+1), range(1,self.Nsites+1)):
    #             if k > l:
    #                 pass
    #             else:
    #                 tangent_out += self.CSn_Ansazte_jvp_aux([k,l,i], i, YJMparams, Hparams, opt='YJM')
    #
    #     return primal_out, tangent_out

    def CSn_Ansazte_jvp_aux(self,YJM_ind, H_ind, YJMparams, Hparams, opt='YJM'):
        '''

        :param YJM_ind: [k, l, p] a list consisting of the index to differentiate the YJMs
        :param H_ind: int(p) index for the Heisenberg
        :return: a sum of two case
        '''

        diff_YJM = float(1.0)
        Ham_rep = self.Ham_rep()
        Ham_rep = jnp.asarray(Ham_rep.astype('float64'))
        GSket = jnp.zeros(self.dim)
        for i in range(len(self.sampling)):
            basis = jnp.zeros(self.dim)
            basis = basis.at[i].set(self.sampling[i])
            GSket = jnp.add(GSket, basis)
        GSket = GSket / jnp.linalg.norm(GSket)
        if opt == 'YJM':
            if H_ind == int(0):
                ansatze2 = float(1.0)
                H_evo = self.Heis_Conv2d(Hparams.at[H_ind].get())
                YJM_evo = self.YJM_Conv2d(YJMparams.at[:, :, H_ind].get())
                YJM = self.get_YJMs(YJM_ind[int(0)], YJM_ind[int(1)])
                YJM = jnp.asarray(YJM.astype('float64'))
                YJM = jnp.multiply(complex(1j), YJM)
                diff_YJM = jnp.multiply(H_evo, diff_YJM)
                diff_YJM = jnp.matmul(YJM_evo, diff_YJM)
                diff_YJM = jnp.matmul(YJM, diff_YJM)
                for i in range(1, self.p):
                    ansatze = jnp.matmul(self.YJM_Conv2d(YJMparams.at[:, :, i].get()),
                                         self.Heis_Conv2d(Hparams.at[i].get()))
                    ansatze2 = jnp.multiply(ansatze, ansatze2)
                diff_YJM = jnp.matmul(ansatze2, diff_YJM)
            else:
                ansatze1 = float(1.0)
                ansatze2 = float(1.0)
                for i in range(H_ind):
                    ansatze = jnp.matmul(self.YJM_Conv2d(YJMparams.at[:, :, i].get()),
                                         self.Heis_Conv2d(Hparams.at[i].get()))
                    ansatze1 = jnp.multiply(ansatze, ansatze1)

                for i in range(H_ind, self.p+1):
                    if i == H_ind:
                        H_evo = self.Heis_Conv2d(Hparams.at[H_ind].get())
                        YJM_evo = self.YJM_Conv2d(YJMparams.at[:, :, H_ind].get())
                        YJM = self.get_YJMs(YJM_ind[int(0)], YJM_ind[int(1)])
                        YJM = jnp.asarray(YJM.astype('float64'))
                        YJM = jnp.multiply(complex(1j), YJM)
                        diff_YJM = jnp.multiply(H_evo, diff_YJM)
                        diff_YJM = jnp.matmul(YJM_evo, diff_YJM)
                        diff_YJM = jnp.matmul(YJM, diff_YJM)
                        diff_YJM = jnp.matmul(diff_YJM, ansatze1)
                    elif i == self.p:
                        pass

                    else:
                        ansatze = jnp.matmul(self.YJM_Conv2d(YJMparams.at[:, :, i].get()),
                                             self.Heis_Conv2d(Hparams.at[i].get()))
                        ansatze2 = jnp.multiply(ansatze, ansatze2)
                diff_YJM = jnp.matmul(ansatze2, diff_YJM)
            return jnp.matmul(diff_YJM, GSket)

        elif opt == 'H':
            diff_H = float(1.0)
            # Ham_rep = self.Ham_rep()
            # Ham_rep = jnp.asarray(Ham_rep.astype('float64'))
            Ham_rep = jnp.multiply(complex(1j), Ham_rep)
            # ansatze1 = jnp.diag(jnp.ones(self.dim))
            # ansatze2 = jnp.diag(jnp.ones(self.dim))
            if H_ind == int(0):
                ansatze2 = float(1.0)
                H_evo = self.Heis_Conv2d(Hparams.at[H_ind].get())
                YJM_evo = self.YJM_Conv2d(YJMparams.at[:, :, H_ind].get())
                diff_H = jnp.multiply(H_evo, diff_H)
                diff_H = jnp.matmul(Ham_rep, diff_H)
                diff_H = jnp.matmul(YJM_evo, diff_H)
                # ansatze1 = jnp.matmul(diff_H, ansatze1)
                for i in range(1, self.p):
                    ansatze = jnp.matmul(self.YJM_Conv2d(YJMparams.at[:, :, i].get()),
                                         self.Heis_Conv2d(Hparams.at[i].get()))
                    print('norm of intermeidate ansatze: {}'.format(jnp.linalg.norm(ansatze)))
                    ansatze2 = jnp.multiply(ansatze, ansatze2)
                print('norm of ansatze: {}'.format(jnp.linalg.norm(ansatze2)))
                diff_H = jnp.matmul(ansatze2, diff_H)
            else:
                ansatze1 = float(1.0)
                ansatze2 = float(1.0)
                for i in range(H_ind):
                    ansatze = jnp.matmul(self.YJM_Conv2d(YJMparams.at[:, :, i].get()),
                                         self.Heis_Conv2d(Hparams.at[i].get()))
                    ansatze1 = jnp.multiply(ansatze, ansatze1)

                for i in range(H_ind, self.p + 1):
                    if i == H_ind:
                        H_evo = self.Heis_Conv2d(Hparams.at[H_ind].get())
                        YJM_evo = self.YJM_Conv2d(YJMparams.at[:, :, H_ind].get())
                        diff_H = jnp.multiply(H_evo, diff_H)
                        diff_H = jnp.matmul(Ham_rep, diff_H)
                        diff_H = jnp.matmul(YJM_evo, diff_H)
                        diff_H = jnp.matmul(diff_H, ansatze1)

                    elif i == self.p:
                        pass

                    else:
                        ansatze = jnp.matmul(self.YJM_Conv2d(YJMparams.at[:, :, i].get()),
                                             self.Heis_Conv2d(Hparams.at[i].get()))
                        ansatze2 = jnp.multiply(ansatze, ansatze2)

                diff_H = jnp.matmul(ansatze2, diff_H)
            return jnp.matmul(diff_H, GSket)





    def energy_jvp_aux(self, YJM_ind, H_ind, YJMparams, Hparams, opt):

        '''
        Debugging  why it returns a vector instead of scalar

        :param YJM_ind:
        :param H_ind:
        :param YJMparams:
        :param Hparams:
        :param opt:
        :return:
        '''
        Ham_rep = jnp.asarray(self.Ham_rep().astype('float64'))
        Ham_rep = jnp.add(Ham_rep, jnp.multiply(self.num_trans, jnp.diag(jnp.ones(self.dim))))
        Psi = self.CSn_VStates(YJMparams, Hparams)
        Psi_tilde = jnp.add(Psi, jnp.conjugate(Psi))
        Psi_tilde = jnp.real(Psi_tilde)
        dev_Psi_tilde = jnp.add(self.CSn_Ansazte_jvp_aux(YJM_ind, H_ind, YJMparams, Hparams, opt=opt),
                                jnp.conjugate(self.CSn_Ansazte_jvp_aux(YJM_ind, H_ind, YJMparams, Hparams, opt=opt)))

        dev_energy_var = jnp.dot(jnp.conjugate(dev_Psi_tilde), jnp.matmul(Ham_rep, Psi_tilde))
        dev_energy_var = (jnp.inner(Psi_tilde, jnp.matmul(Ham_rep, Psi_tilde))
                          + dev_energy_var)/ jnp.inner(Psi_tilde, Psi_tilde)
        temp = jnp.dot(jnp.conjugate(dev_Psi_tilde), Psi_tilde) + jnp.dot(Psi_tilde, dev_Psi_tilde)
        dev_energy_var -= jnp.multiply(self.Expect_braket_energy(YJMparams, Hparams),
                                       temp / jnp.dot(Psi_tilde, Psi_tilde))

        return dev_energy_var



    def energy_jvp(self, YJMparams, Hparams):
        grad_yjm = jnp.zeros(YJMparams.shape)
        grad_h = jnp.zeros(Hparams.shape)
        for i in range(self.p):
            grad_h = grad_h.at[i].set(self.energy_jvp_aux([int(1), int(1), i], i, YJMparams, Hparams, 'H'))

            for k, l in zip(range(1,self.Nsites+1), range(1,self.Nsites+1)):
                if k > l:
                    pass
                else:
                    print(self.CSn_Ansazte_jvp_aux([k,l,i], i, YJMparams, Hparams, 'YJM'))
                    grad_yjm = grad_yjm.at[k, l, i].set(self.CSn_Ansazte_jvp_aux([k,l,i], i, YJMparams, Hparams, 'YJM'))

        return grad_yjm, grad_h

    '''
    -------------------------------------------------------------------------------------------------------

    Loss function Designs 

    ------------------------------------------------------------------------------------------------------
    '''

    def Expect_braket(self, YJMparams, Hparams, scale = 1e-3):
        groundstate = self.CSn_VStates(YJMparams, Hparams)
        rep_H = self.Ham_rep().astype('float64')
        rep_H = jnp.asarray(rep_H)
        if self.quantumnoise:
            noise = jax.random.normal(random.PRNGKey(int(24)), jnp.shape(rep_H)) * scale
            rep_H = noise + rep_H


        rep_H += jnp.add(rep_H, jnp.multiply(self.num_trans, jnp.diag(jnp.ones(self.dim))))
        # make sure the Hamiltonian is positive-definite by adding scalar multiple of identitties
        return jnp.matmul(jnp.conjugate(groundstate), jnp.matmul(rep_H, groundstate)) / jnp.linalg.norm(groundstate)

    def Expect_braket_energy(self,YJMparams, Hparams):
        # split = jnp.multiply(jnp.multiply(self.Nsites, self.Nsites), self.p)
        # YJMparams = jnp.reshape(Params.at[int(0):split].get(), newshape=(self.Nsites, self.Nsites, self.p))
        # Hparams = Params.at[split:int(-1)].get()
        groundstate = self.CSn_VStates(YJMparams, Hparams)
        rep_H = self.Ham_rep().astype('float64')
        rep_H = jnp.asarray(rep_H)
        return jnp.matmul(jnp.conjugate(groundstate), jnp.matmul(rep_H, groundstate)) / jnp.linalg.norm(groundstate)





    def Expect_OverlapE(self, YJMparams, Hparams):
        '''
        :param v_gs: the true ground state (should be real-valued)
        :param YJMparams:
        :param Hparams:
        :return:
        '''
        E_gs, V_gs = self.ED_Ham()
        V_gs = jnp.asarray(V_gs)
        trial_energy = self.Expect_braket_energy(YJMparams, Hparams)
        # dist = jnp.linalg.norm(jnp.subtract(v_gs, groundstate))
        return jnp.linalg.norm(jnp.subtract(E_gs, trial_energy))


    def Expect_OverlapS(self, YJMparams, Hparams):
        E_gs, V_gs = self.ED_Ham()
        V_gs = jnp.asarray(V_gs)
        trial_state = self.CSn_VStates(YJMparams, Hparams)
        # print('norm of the ground state: {}'.format(jnp.linalg.norm(V_gs)))
        print('probability of cross section: {}'.format(jnp.power(jnp.dot(V_gs, trial_state), int(2))))
        return jnp.power(jnp.subtract(jnp.power(jnp.dot(V_gs, trial_state), int(2)), float(1.0)), int(2))

    '''
    ---------------------------------------------------------------------

    Gradient Descent Mathods: 

    (1) Gradient with momentum 

    (2) Nadam 

    (3) Stochastic Reconfigration  (Natural Gradient) 


    ---------------------------------------------------------------------
    '''




    def GD_momentum(self, scale=float(1e0), Params = None, mode = 'jax', opt='expectation'):

        """Performs linear regression using batch gradient descent + momentum

        Returns:
            params: the weights and bias after performing the optimization
        """

        # Some configurations
        LOG = True

        if Params is None:
            YJMparams, Hparams = self.random_params2(scale = scale )
        else:
            YJMparams = Params[0]
            Hparams = Params[1]

        # To keep track of velocity parameters
        params_v = {
            'YJM': jnp.zeros((self.Nsites, self.Nsites, self.p)),
            'H': jnp.zeros(self.p)
        }
         # print('shape of the params_v[YJM]: {}'.format(params_v['YJM'].shape))

        # Define the gradient function w.r.t w and b
        if mode == 'jax':
            if opt=='expectation':
                grad_YJM = jax.jit(jax.grad(self.Expect_braket, argnums=int(0))) # argnums indicates which variable to differentiate with from the parameters list passed to the function
                grad_H = jax.jit(jax.grad(self.Expect_braket, argnums=int(1)))

            elif opt == 'overlap_energy':
                grad_YJM = jax.jit(jax.grad(self.Expect_OverlapE, argnums=int(0)))  # argnums indicates which variable to differentiate with from the parameters list passed to the function
                grad_H = jax.jit(jax.grad(self.Expect_OverlapE, argnums=int(1)))

            elif opt == 'overlap_state':
                grad_YJM = jax.jit(jax.grad(self.Expect_OverlapS, argnums=int(0))) # argnums indicates which variable to differentiate with from the parameters list passed to the function
                grad_H = jax.jit(jax.grad(self.Expect_OverlapS, argnums=int(1)))

            else:
                pass


            # Run once to compile JIT (Just In Time). The next uses of grad_W and grad_B will now be fast


            grad_YJM(YJMparams, Hparams)
            grad_H(YJMparams, Hparams)

        for i in range(self.max_iter):
            # Gradient w.r.t. argumnet index 1 i.e., w
            #             grad_yjm = grad_YJM(YJMparams, Hparams)

            if mode == 'jax':
                grad_yjm = grad_YJM(YJMparams, Hparams)
                grad_h = grad_H(YJMparams, Hparams)
            elif mode == 'exact':
                grad_yjm = self.energy_jvp(YJMparams, Hparams)
                grad_h = self.energy_jvp(YJMparams, Hparams)
            else:
                raise ValueError

            if grad_yjm.shape != params_v['YJM'].shape:
                print('dim for the grad_yjm: {}'.format(grad_yjm.shape))
                raise ValueError
            elif grad_yjm.shape == params_v['YJM'].shape:
                params_v['YJM'] = jnp.add(jnp.multiply(self.gamma, params_v['YJM']), grad_yjm)

            # Gradient w.r.t. argumnet index 2 i.e., b

            elif grad_h.shape != params_v['H'].shape:
               # grad_h = jnp.broadcast_to(grad_h, params_v['H'].shape
                temp = jnp.zeros(params_v['H'].shape)
                temp = temp.at[int(0): grad_h.shape].add(grad_h)
                params_v['H'] = jnp.add(jnp.multiply(self.gamma, params_v['H']), temp)
            elif grad_h.shape == params_v['H'].shape:
                params_v['H'] = jnp.add(jnp.multiply(self.gamma, params_v['H']), grad_h)


            if i % int(50) == int(0):

                print('shape of updated params_v[H]: {}'.format(params_v['H'].shape))
                print('shape of updated params_v[YJM]: {}'.format(params_v['YJM'].shape))


            # Parameter update
            YJMparams -= jnp.multiply(self.lr, params_v['YJM'])

            Hparams -= jnp.multiply(self.lr, params_v['H'])

            if LOG and i % int(100) == int(0):
                if opt == 'overlap_state':
                    print('energy expectation at iteration {}: --- ({})'.format(i,
                                                                            self.Expect_OverlapS(YJMparams, Hparams)))
                else:
                    print('energy expectation at iteration {}: --- ({})'.format(i,
                                                                                self.Expect_braket_energy(YJMparams,
                                                                                                     Hparams)))

        return YJMparams, Hparams



    def CSn_SR(self, loss,  scale = float(1e-2)):
        pass


    def CSn_nadam(self,J,  Params = None, delta1=float(0.99), delta2=float(0.999), scale = float(1.0), mode = 'jax'):

        """
        Using Nadam to accelerate the gradient descent

        Args:
            J: cost function  like self.Expect_braket or self.Expect_overlapS
            delta1: decay parameter 1
            delta2: decay parameter 2

        Returns:
            params: the optimized YJMparams, Hparams
        """

        '''
        ----------------------------------------------------
        The best learning rate for the Expect_overlapS is self.lr = float(2e-3)
        
        
        ---------------------------------------------------
        
        '''
        # Some configurations
        LOG = True
        # lr = float(0.5)  # Learning rate
        e = float(1e-7)  # Epsilon value to prevent the fractions going to infinity when denominator is zero

        if Params is None:
            YJMparams, Hparams = self.random_params2(scale=scale)
        else:
            YJMparams = Params[0]
            Hparams = Params[1]


        # To keep track of velocity parameters
        params_v = {
            'YJM': jnp.zeros(YJMparams.shape),
            'H': jnp.zeros(Hparams.shape)
        }

        # To keep running sum of squares of gradients with decay
        squared_grad = {
            'YJM': jnp.zeros(YJMparams.shape),
            'H': jnp.zeros(Hparams.shape)
        }

        # print('squared_grad: {}'.format(squared_grad['H'].shape))

        # Define the gradient function w.r.t yjm and b
        grad_YJM= jax.jit(jax.grad(J,
                                  argnums=int(0)))  # argnums indicates which variable to differentiate with from the parameters list passed to the function
        grad_H = jax.jit(jax.grad(J, argnums=int(1)))

        # Run once to compile JIT (Just In Time). The next uses of grad_yjm and grad_h will now be fast
        grad_YJM(YJMparams, Hparams)
        grad_H(YJMparams, Hparams)
        energy_list = []
        for i in range(self.max_iter):
            # Gradient w.r.t. argumnet index 1 i.e., YJMparams
            grad_yjm= grad_YJM(YJMparams, Hparams)

            # Gradient w.r.t. argumnet index 2 i.e., Hparams
            grad_h = grad_H(YJMparams, Hparams)

            # Momements update
            params_v['YJM'] = jnp.add(jnp.multiply(delta1 , params_v['YJM']) ,
                                      jnp.multiply(jnp.subtract(float(1.0) , delta1) , grad_yjm ))

            params_v['H'] = jnp.add(jnp.multiply(delta1 , params_v['H']) ,
                                      jnp.multiply(jnp.subtract(float(1.0) , delta1) , grad_h ))

            squared_grad['YJM'] = jnp.add(jnp.multiply(delta2 , squared_grad['YJM'] ),
                                          jnp.multiply(jnp.subtract(float(1.0), delta2) ,
                                                       (jnp.multiply(grad_yjm , grad_yjm))))

            squared_grad['H'] =jnp.add(jnp.multiply(delta2 , squared_grad['H'] ),
                                          jnp.multiply(jnp.subtract(float(1.0), delta2) ,
                                                       (jnp.multiply(grad_h , grad_h))))

            # Bias correction
            moment_yjm = params_v['YJM'] / jnp.subtract(float(1.0) , jnp.power(delta1, (jnp.add(i , int(1)))))
            moment_h= params_v['H'] / jnp.subtract(float(1.0) , jnp.power(delta1, (jnp.add(i , int(1)))))

            moment_squared_yjm = squared_grad['YJM'] / jnp.subtract(float(1.0) , jnp.power(delta2, (jnp.add(i , int(1)))))
            moment_squared_h = squared_grad['H'] / jnp.subtract(float(1.0) , jnp.power(delta2, (jnp.add(i , int(1)))))

            # Parameter update
            YJMparams -= jnp.multiply((self.lr / jnp.add(jnp.sqrt(moment_squared_yjm) , e)) ,
                        jnp.add(jnp.multiply(delta1 , moment_yjm) , jnp.multiply(jnp.subtract(float(1) , delta1) ,grad_yjm )
                        / jnp.subtract(float(1) , jnp.power(delta2, (jnp.add(i , int(1)))))))


            Hparams -= jnp.multiply((self.lr / jnp.add(jnp.sqrt(moment_squared_h) , e)) ,
                        jnp.add(jnp.multiply(delta1 , moment_h) , jnp.multiply(jnp.subtract(float(1) , delta1) ,grad_h )
                        / jnp.subtract(float(1) , jnp.power(delta2, (jnp.add(i , int(1)))))))

            # loss_energy = self.Expect_braket_energy(YJMparams, Hparams)
            # energy_list.append(loss_energy)
            if LOG and i % 5 == 0:
                print('updated gradient squared: {}---{}'.format(squared_grad['YJM'].shape, squared_grad['H'].shape))
                print('updated bia correction have the shape: {}--{}'.format(moment_squared_yjm.shape, moment_squared_h.shape))
                print('updated YJMparams, Hparams have the shape: {}, {}'.format(YJMparams.shape, Hparams.shape))
                # loss = J(YJMparams, Hparams)
                loss_energy = self.Expect_braket_energy(YJMparams, Hparams)
                print('energy expectation at iteration {}: --- ({})'.format(i, loss_energy))
                energy_list.append(loss_energy)
                # if loss < float(1e-6):
                #     print('--------------------------------------------')
                #     print('finding the optimized parameters for the energy expectation: {}'.format(loss))
                #     return YJMparams, Hparams


        return YJMparams, Hparams, energy_list


    def CQA_BFGS(self, J,  Params = None, scale=float(1e-1)):
        if Params is None:
            YJMparams, Hparams = self.random_params2(scale=scale)
        else:
            YJMparams = Params[0]
            Hparams = Params[1]




    def Groundstate(self, YJMparams, Hparams):
        # Params = self.SGD_momentum()
        return self.CSn_VStates(YJMparams, Hparams)