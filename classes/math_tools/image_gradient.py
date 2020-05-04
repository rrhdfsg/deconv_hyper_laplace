from cv2 import Laplacian as gradient
from cv2 import CV_64F
from numpy import histogram, log2, array


class Gradient():
    """
    A class that, when instantiated, loads in and calculates quantities related to the gradient, gradient norm
    and grandient norm distribution of an array

    **kwargs can be defined in such a way as to pass arguments to the gradient calculator
    """
    def __init__(self, image_array, **kwargs):
        self.log2pdf = None
        self.gradient_vals = None
        self.image_array = image_array
        self.calc_gradient(**kwargs)
        self.make_gradient_vector()
        self.make_gradient_pdf()

    def calc_gradient(self, **kwargs):
        """
        Gradient calculation
        """
        self.image_gradient = gradient(self.image_array, CV_64F, **kwargs)

    def make_gradient_vector(self):
        self.gradient_vector = self.image_gradient.flatten()

    def make_gradient_pdf(self):
        h = 3.49 * self.gradient_vector.std() / (self.gradient_vector.shape[0]) ** (1 / 3)
        k = int((self.gradient_vector.max() - self.gradient_vector.min()) / h)
        P1, X1 = histogram(self.gradient_vector, bins=k, density=True)
        self.log2pdf = log2(P1[P1 != 0])
        self.gradient_vals = array([(X1[i] + X1[i + 1]) / 2 for i in range(X1.shape[0] - 1)])[P1 != 0]
