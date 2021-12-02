import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('JAX SGD')
from FourierFilters import FourierFilters
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

    def __init__(self, lr=float(0.05), max_iter=int(501), gamma=float(0.95), num_samples=None, **kwargs):
        super(CSnGradient, self).__init__(**kwargs)
        self.lr = lr
        self.max_iter = max_iter
        self.gamma = gamma
        self.sampling = np.random.randint(0, high=self.dim, size=num_samples)
        num_trans = jnp.add(len(self.lattice[0]), len(self.lattice[1]))
        # trans_const = jnp.multiply(num_trans, jnp.ones(self.dim), dtype='complex128')
        self.num_trans = num_trans



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
    def CSn_Ansazte_jvp(self,primals, tangents):
        '''

        Not for current use as still working on modify the custom derivative rules

        :param primals:
        :param tangents:
        :return:
        '''
        YJMparams, Hparams = primals
        YJMparams_dot, Hparams_dot = tangents
        primal_out = self.CSn_VStates(YJMparams, Hparams)
        tangent_out = jnp.zeros(self.dim)
        for i in range(self.p):
            tangent_out += self.CSn_Ansazte_jvp_aux([int(1), int(1), i], i, YJMparams, Hparams, opt='H')
            for k, l in zip(range(1,self.Nsites+1), range(1,self.Nsites+1)):
                if k > l:
                    pass
                else:
                    tangent_out += self.CSn_Ansazte_jvp_aux([k,l,i], i, YJMparams, Hparams, opt='YJM')

        return primal_out, tangent_out

    def CSn_Ansazte_jvp_aux(self,YJM_ind, H_ind, YJMparams, Hparams, opt='YJM'):
        '''

        :param YJM_ind: [k, l, p] a list consisting of the index to differentiate the YJMs
        :param H_ind: int(p) index for the Heisenberg
        :return: a sum of two case
        '''

        diff_YJM = jnp.diag(jnp.ones(self.dim))
        Ham_rep = self.Ham_rep()
        Ham_rep = jnp.asarray(Ham_rep.astype('float64'))
        GSket = jnp.zeros(self.dim)
        for i in range(len(self.sampling)):
            basis = jnp.zeros(self.dim)
            basis = basis.at[i].set(self.sampling[i])
            GSket = jnp.add(GSket, basis)
        GSket = GSket / jnp.linalg.norm(GSket)
        if opt == 'YJM':
            for i in range(self.p):
                if H_ind == i:
                    H_evo = self.Heis_Conv2d(Hparams.at[H_ind].get(), Ham_rep)
                    YJM_evo = self.YJM_Conv2d(YJMparams.at[:, :, H_ind].get())
                    YJM = self.get_YJMs(YJM_ind[int(0)], YJM_ind[int(0)])
                    YJM = jnp.asarray(YJM.astype('float64'))
                    YJM = jnp.matmul(jnp.multiply(complex(1j), YJM))
                    diff_YJM = jnp.matmul(H_evo, diff_YJM)
                    diff_YJM = jnp.matmul(YJM_evo, diff_YJM)
                    diff_YJM = jnp.matmul(YJM, diff_YJM)

                ansatze = jnp.matmul(self.YJM_Conv2d(YJMparams.at[:, :, i].get()),
                                     self.Heis_Conv2d(Hparams.at[i].get(), Ham_rep))
                ansatze = ansatze / jnp.linalg.norm(ansatze)
                diff_YJM = jnp.matmul(ansatze, diff_YJM)
            return jnp.matmul(diff_YJM, GSket)

        elif opt == 'H':
            diff_H = jnp.diag(jnp.ones(self.dim))
            # Ham_rep = self.Ham_rep()
            # Ham_rep = jnp.asarray(Ham_rep.astype('float64'))
            Ham_rep = jnp.matmul(complex(1j), Ham_rep)
            for i in range(self.p):
                if H_ind == i:
                    H_evo = self.Heis_Conv2d(Hparams.at[H_ind].get(), Ham_rep)
                    YJM_evo = self.YJM_Conv2d(YJMparams.at[:, :, H_ind].get())
                    diff_H = jnp.matmul(H_evo, diff_H)
                    diff_H = jnp.matmul(Ham_rep, diff_H)
                    diff_H = jnp.matmul(YJM_evo, diff_H)

                ansatze = jnp.matmul(self.YJM_Conv2d(YJMparams.at[:, :, i].get()),
                                     self.Heis_Conv2d(Hparams.at[i].get(), Ham_rep))
                ansatze = ansatze / jnp.linalg.norm(ansatze)
                diff_H = jnp.matmul(ansatze, diff_H)

            return jnp.matmul(diff_YJM, GSket)

    def energy_jvp_aux(self, YJM_ind, H_ind, YJMparams, Hparams, opt):

        Ham_rep = jnp.asarray(self.Ham_rep().astype('float64'))
        Ham_rep = jnp.add(Ham_rep, jnp.multiply(self.num_trans, jnp.diag(jnp.ones(self.dim))))

        Psi_tilde = jnp.add(self.CSn_Ansazte(YJMparams, Hparams))
        Psi_tilde = jnp.real(Psi_tilde)
        dev_Psi_tilde = jnp.add(self.CSn_Ansazte_jvp_aux(YJM_ind, H_ind, YJMparams, Hparams, opt=opt),
                                jnp.conjugate(self.CSn_Ansazte_jvp_aux(YJM_ind, H_ind, YJMparams, Hparams, opt=opt)))

        dev_energy_var = jnp.dot(jnp.conjugate(dev_Psi_tilde), jnp.matmul(Ham_rep, Psi_tilde))
        dev_energy_var = (jnp.dot(Psi_tilde, jnp.matmul(Ham_rep, Psi_tilde))
                          + dev_energy_var)/ jnp.dot(Psi_tilde, Psi_tilde)
        temp = jnp.dot(jnp.conjugate(dev_Psi_tilde), Psi_tilde) + jnp.dot(Psi_tilde, dev_Psi_tilde)
        dev_energy_var -= jnp.multiply(self.Expect_braket_energy(YJMparams, Hparams),
                                       temp /jnp.dot(Psi_tilde, Psi_tilde))

        return dev_energy_var













    def Expect_braket(self, YJMparams, Hparams):
        groundstate = self.CSn_VStates(YJMparams, Hparams)
        rep_H = self.Ham_rep().astype('float64')
        rep_H = jnp.asarray(rep_H)
        rep_H += jnp.add(rep_H, jnp.multiply(self.num_trans, jnp.diag(jnp.ones(self.dim))))
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
        return jnp.subtract(float(1.0), jnp.abs(jnp.dot(V_gs, trial_state)))




    def GD_momentum(self, scale=float(1e0), opt='expectation'):

        """Performs linear regression using batch gradient descent + momentum

        Returns:
            params: the weights and bias after performing the optimization
        """

        # Some configurations
        LOG = True

        YJMparams, Hparams = self.random_params2(scale = scale )

        # To keep track of velocity parameters
        params_v = {
            'YJM': jnp.zeros((self.Nsites, self.Nsites, self.p)),
            'H': jnp.zeros(self.p)
        }
         # print('shape of the params_v[YJM]: {}'.format(params_v['YJM'].shape))

        # Define the gradient function w.r.t w and b
        if opt=='expectation':
            grad_YJM = jax.jit(jax.grad(self.Expect_braket, argnums=int(0)))# argnums indicates which variable to differentiate with from the parameters list passed to the function
            grad_H = jax.jit(jax.grad(self.Expect_braket, argnums=int(1)))

        elif opt == 'overlap_energy':
            grad_YJM = jax.jit(jax.grad(self.Expect_OverlapE, argnums=int(
                0)))  # argnums indicates which variable to differentiate with from the parameters list passed to the function
            grad_H = jax.jit(jax.grad(self.Expect_OverlapE, argnums=int(1)))

        elif opt == 'overlap_state':
            grad_YJM = jax.jit(jax.grad(self.Expect_OverlapS, argnums=int(
                0)))  # argnums indicates which variable to differentiate with from the parameters list passed to the function
            grad_H = jax.jit(jax.grad(self.Expect_OverlapS, argnums=int(1)))

        else:
            raise ValueError

        # Run once to compile JIT (Just In Time). The next uses of grad_W and grad_B will now be fast

        # vgrad(self.Expect_braket, Params)
        #         grad_H(YJMparams, Hparams)
        grad_YJM(YJMparams, Hparams)
        grad_H(YJMparams, Hparams)

        for i in range(self.max_iter):
            # Gradient w.r.t. argumnet index 1 i.e., w
            #             grad_yjm = grad_YJM(YJMparams, Hparams)

            grad_yjm = grad_YJM(YJMparams, Hparams)
            grad_h = grad_H(YJMparams, Hparams)


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


            if i % int(20) == int(0):

                print('shape of updated params_v[H]: {}'.format(params_v['H'].shape))
                print('shape of updated params_v[YJM]: {}'.format(params_v['YJM'].shape))


            # Parameter update
            YJMparams -= jnp.multiply(self.lr, params_v['YJM'])

            Hparams -= jnp.multiply(self.lr, params_v['H'])

            if LOG and i % int(50) == int(0):
                print('energy expectation at iteration {}: --- ({})'.format(i,
                                                                            self.Expect_braket_energy(YJMparams, Hparams)))

        return YJMparams, Hparams



    def CSn_SR(self, loss,  scale = float(1e-2)):
        pass


    def nadam(self,J,  delta1=float(0.9), delta2=float(0.999), scale = float(1.0)):
        """Performs linear regression using nadam

        Args:
            J: cost function  like self.Expect_braket or self.Expect_overlapS
            delta1: decay parameter 1
            delta2: decay parameter 2

        Returns:
            params: the weights and bias after performing the optimization
        """
        # Some configurations
        LOG = False
        # lr = float(0.5)  # Learning rate
        e = float(1e-7)  # Epsilon value to prevent the fractions going to infinity when denominator is zero

        YJMparams, Hparams = self.random_params2(scale=scale)


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

        # Define the gradient function w.r.t w and b
        grad_YJM= jax.jit(jax.grad(J,
                                  argnums=int(0)))  # argnums indicates which variable to differentiate with from the parameters list passed to the function
        grad_H = jax.jit(jax.grad(J, argnums=int(0)))

        # Run once to compile JIT (Just In Time). The next uses of grad_W and grad_B will now be fast
        grad_YJM(YJMparams, Hparams)
        grad_H(YJMparams, Hparams)

        for i in range(int(1000)):
            # Gradient w.r.t. argumnet index 1 i.e., w
            grad_yjm= grad_YJM(YJMparams, Hparams)
            # Gradient w.r.t. argumnet index 2 i.e., b
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

            if LOG and i % 50 == 0:
                print('energy expectation at iteration {}: --- ({})'.format(i, self.Expect_braket_energy(YJMparams, Hparams)))

        return YJMparams, Hparams



    def Groundstate(self, YJMparams, Hparams):
        # Params = self.SGD_momentum()
        return self.CSn_VStates(YJMparams, Hparams)