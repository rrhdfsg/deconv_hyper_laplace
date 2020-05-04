from classes.data_handling.image import Image
from numpy import diag, ones, array, arange, exp, log, linspace, round
from numpy.linalg import inv
from scipy import sparse
from math import pi
import matplotlib.pyplot as plt
import copy

class DeconvAlgorithm():
    """
    A class that, when instantiated, uses an image path to deconvolve an image according to
    a hyper-laplacian prior

    """
    def __init__(self, args):

        self.image_path = args.image

        self.config = args.config
        self.image = Image(self.image_path)

        #dummy values
        self.w1 = ones((self.image.image_array.shape[0] * self.image.image_array.shape[1],1))
        self.w2 = ones((self.image.image_array.shape[0] * self.image.image_array.shape[1],1))
        self.lam = 1
        self.bet = 1

        self.calc_X()
        plt.figure()
        plt.imshow(self.image.image_array.sum(axis=2))
        plt.figure()
        plt.imshow(self.x.sum(axis=2))
        plt.show()

    def make_DFT(self, n):
        return exp(2 * pi * 1j * (arange(0, n, 1).reshape((-1, 1)) * arange(0, n, 1).reshape((-1, 1)).T) / n)

    def make_F1(self):
        I = sparse.eye(self.image.image_array_dims[0])
        dx = sparse.eye(self.image.image_array_dims[0], k=1) - sparse.eye(self.image.image_array_dims[0])
        self.F1 = sparse.kron(I, dx)

    def make_F2(self):
        I = sparse.eye(self.image.image_array_dims[1])
        dx = sparse.eye(self.image.image_array_dims[0], k=1) - sparse.eye(self.image.image_array_dims[0])
        self.F2 = sparse.kron(dx, I)

    def make_DFT(self, n):
        return exp(2 * pi * 1j * (arange(0, n, 1).reshape((-1, 1)) * arange(0, n, 1).reshape((-1, 1)).T) / n)

    def make_F1(self):
        I = sparse.eye(self.image.image_array_dims[1])
        dx = sparse.eye(self.image.image_array_dims[0], k=1) - sparse.eye(self.image.image_array_dims[0])
        self.F1 = sparse.kron(I, dx)

    def make_K(self, n):
        m = (log(n))**(1/2)
        x = linspace(-m, m, n-1)
        exp_K = exp(-x ** 2) * n
        k = (exp_K.reshape((-1, 1)) * exp_K.reshape((-1, 1)).T) / (exp_K.sum())**2
        j = 0
        self.K = sparse.eye(self.image.image_array_dims[0] * self.image.image_array_dims[1],self.image.image_array_dims[0] * self.image.image_array_dims[1],0) * 0
        for i in range(int(n/2), n):
            self.K += sparse.eye(self.image.image_array_dims[0] * self.image.image_array_dims[1], self.image.image_array_dims[0] * self.image.image_array_dims[1], j) * k[int(n / 2)-1, i-1]
            self.K += sparse.eye(self.image.image_array_dims[0] * self.image.image_array_dims[1], self.image.image_array_dims[0] * self.image.image_array_dims[1], -j) * k[int(n / 2)-1, i-1]
            j += 1


    def calc_X(self):
        self.make_F1()
        self.make_F2()
        self.make_K(6)
        vec_y = self.image.image_array[:, :, 0].reshape((-1, 1))
        x = copy.copy(self.image.image_array)
        F = self.make_DFT(vec_y.shape[0])


        # Didn't end up using these because it was unclear how to implement this in the frequency domain
        inv_F = inv(F)
        FF1 = F @ self.F1
        FF2 = F @ self.F2

        Fw1 = F @ self.w1
        Fw2 = F @ self.w2
        FK = F @ self.K

        for i in range(self.image.image_array.shape[2]):
            vec_y = self.image.image_array[:, :, i].reshape((-1, 1))
            H = (self.F1.T @ self.F1 + self.F2.T @ self.F2 + (self.lam / self.bet) * self.K.T @ self.K).toarray()
            vec_x = inv(H) @ self.K.T @ vec_y
            x[:,:,i] = vec_x.reshape(self.image.image_array[:, :, 0].shape)
        self.x = x




