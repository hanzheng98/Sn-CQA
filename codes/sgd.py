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

class CSnSGD(FourierFilters):
    '''
    Standard SGD method implemented using JAX autodifferentiation megthd

    Unsurpervised method for the minilization of the GS energy for each Sn irrep

    Wirtten in the language environment SAGE9.2, supported SageMath environment

    '''

    def __init__(self, lr=float(0.05), max_iter=int(1e3), gamma=float(0.95), num_samples=None, **kwargs):
        super(CSnSGD, self).__init__(**kwargs)
        self.lr = lr
        self.max_iter = max_iter
        self.gamma = gamma
        self.sampling = np.random.randint(0, high=self.dim, size=num_samples)

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

    def CSn_VStates(self, YJMparams, Hparams):
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
        return GSket / jnp.linalg.norm(GSket)

    def Expect_braket(self, YJMparams, Hparams):
        # split = jnp.multiply(jnp.multiply(self.Nsites, self.Nsites), self.p)
        # YJMparams = jnp.reshape(Params.at[int(0):split].get(), newshape=(self.Nsites, self.Nsites, self.p))
        # Hparams = Params.at[split:int(-1)].get()
        groundstate = self.CSn_VStates(YJMparams, Hparams)
        rep_H = self.Ham_rep()
        rep_H = jnp.asarray(rep_H.astype('complex128'))
        return jnp.matmul(jnp.conjugate(groundstate), jnp.matmul(rep_H, groundstate)) / jnp.linalg.norm(groundstate)

    def SGD_momentum(self, scale=float(1e-2)):

        """Performs linear regression using batch gradient descent + momentum

        Returns:
            params: the weights and bias after performing the optimization
        """

        # def vgrad(f, x):
        #     '''
        #
        #     :param f:  Expectation function self.Expect_braket
        #     :param x:  Params 1-d jnp array with parameters for both YJM and H
        #     :return: The gradient in dtype= complex128
        #     '''
        #     y, vjp_fn = jax.vjp(f, x)
        #     print(y)
        #     print(yjp_fn)
        #     return vjp_fn(jnp.ones(y.shape, dtype='complex128'))[int(0)]

        # Some configurations
        LOG = True
        # Params = self.random_params(scale=scale)
        # split = jnp.multiply(jnp.multiply(self.Nsites, self.Nsites), self.p)
        # YJMparams = jnp.reshape(Params.at[int(0):split].get(), newshape=(self.Nsites, self.Nsites, self.p))
        YJMparams, Hparams = self.random_params2(scale = scale )
        YJMparams = jnp.asarray(YJMparams, dtype='complex128')
        # Hparams = Params.at[split:int(-1)].get()
        Hparams = jnp.asarray(Hparams, dtype='complex128')
        #         YJMparams = Params[0]
        #         Hparams = Params[1]

        # To keep track of velocity parameters
        params_v = {
            'YJM': jnp.zeros((self.Nsites, self.Nsites, self.p), dtype='complex128'),
            'H': jnp.zeros(self.p, dtype='complex128')
        }
         # print('shape of the params_v[YJM]: {}'.format(params_v['YJM'].shape))

        # Define the gradient function w.r.t w and b
        grad_YJM = jax.jit(jax.grad(self.Expect_braket, argnums=int(0), holomorphic=True))# argnums indicates which variable to differentiate with from the parameters list passed to the function
        grad_H = jax.jit(jax.grad(self.Expect_braket, argnums=int(1), holomorphic=True))

        # Run once to compile JIT (Just In Time). The next uses of grad_W and grad_B will now be fast

        # vgrad(self.Expect_braket, Params)
        #         grad_H(YJMparams, Hparams)
        grad_YJM(YJMparams, Hparams)
        grad_H(YJMparams, Hparams)

        for i in range(self.max_iter):
            # Gradient w.r.t. argumnet index 1 i.e., w
            #             grad_yjm = grad_YJM(YJMparams, Hparams)
            grad_yjm = grad_YJM(YJMparams, Hparams)
            # print('shape for grad_yjm: {}'.format(grad_yjm.shape))
            grad_h = grad_H(YJMparams, Hparams)
            # print('shape for grad_h: {}'.format(grad_h.shape))

            if grad_yjm.shape != params_v['YJM'].shape:
                # grad_yjm = jnp.broadcast_to(grad_yjm, params_v['YJM'].shape)
                # params_v['YJM'].at[int(0):grad_yjm.shape[int(0)],
                #                  int(0):grad_yjm.shape[int(1)], int(0):grad_yjm.shape[int(2)]] = jnp.multiply(self.gamma, params_v['YJM']).at[int(0):grad_yjm.shape[int(0)],
                #                  int(0):grad_yjm.shape[int(1)], int(0):grad_yjm.shape[int(2)]].add(grad_yjm)
                print('dim for the grad_yjm: {}'.format(grad_yjm.shape))
                raise ValueError
            elif grad_yjm.shape == params_v['YJM'].shape:
                params_v['YJM'] = jnp.add(jnp.multiply(self.gamma, params_v['YJM']), grad_yjm)



            # Gradient w.r.t. argumnet index 2 i.e., b

            elif grad_h.shape != params_v['H'].shape:
               # grad_h = jnp.broadcast_to(grad_h, params_v['H'].shape
                temp = jnp.zeros(params_v['H'].shape, dtype='complex128')
                temp = temp.at[int(0): grad_h.shape].add(grad_h)
                params_v['H'] = jnp.add(jnp.multiply(self.gamma, params_v['H']), temp)
            elif grad_h.shape == params_v['H'].shape:
                params_v['H'] = jnp.add(jnp.multiply(self.gamma, params_v['H']), grad_h)


            if i % int(50) == int(0):

                print('shape of updated params_v[H]: {}'.format(params_v['H'].shape))
                print('shape of updatged params_v[YJM]: {}'.format(params_v['YJM'].shape))
            # # Update velocity
            # params_v['YJM'] = jnp.add(jnp.multiply(self.gamma, params_v['YJM']), grad_yjm)
            #
            # params_v['H'] = jnp.add(jnp.multiply(self.gamma, params_v['H']), grad_h)

            # Parameter update
            YJMparams -= jnp.multiply(self.lr, params_v['YJM'])

            Hparams -= jnp.multiply(self.lr, params_v['H'])

            if LOG and i % int(50) == int(0):
                print(self.Expect_braket(YJMparams, Hparams))

        return YJMparams, Hparams



    def CSn_SR(self, loss,  scale = float(1e-2)):
        pass

    def Groundstate(self, YJMparams, Hparams):
        # Params = self.SGD_momentum()
        return self.CSn_VStates(YJMparams, Hparams)