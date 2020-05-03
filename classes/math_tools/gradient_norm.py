from numpy import gradient, zeros
from numpy.linalg import norm


class GradientNorm():
    """
    A class that, when instantiated, loads in and calculates quantities related to the gradient, gradient norm
    and grandient norm distribution of an array

    **kwargs can be defined in such a way as to pass arguments to the gradient calculator
    """
    def __init__(self, array, **kwargs):
        self.array = array
        self.array_dims = array.shape[:-1]
        self.number_color_channels = array.shape[-1] - 1
        self.channel_gradient_list = None
        self.channel_gradient_norm = None
        self.calc_gradient_norm(**kwargs)

    def calc_gradient_norm(self, **kwargs):
        """
        Gradient norm calculation
        """

        self.channel_gradient_list = self.calc_array_gradient(self.array, **kwargs)
        self.calc_channel_gradient_norms()

    def calc_channel_gradient_norms(self):
        """
        Norm of gradient arrays in list (for all channels in the image)
        """
        self.channel_gradient_norm = zeros(self.array_dims + (self.number_color_channels,))
        for i in range(self.number_color_channels):
            self.channel_gradient_norm[:, :, i] = self.array_norm(self.channel_gradient_list[i])


    @staticmethod
    def calc_array_gradient(array, **kwargs):
        """
        Gradient calculation for multi-"channel" array
        """
        channel_gradient_list = []
        for i in range(array.shape[2]-1):
            channel_gradient = zeros(array.shape[0:2] + (2,))
            channel_gradient[:, :, 0] = gradient(array[:, :, i], axis=0, **kwargs)
            channel_gradient[:, :, 1] = gradient(array[:, :, i], axis=1, **kwargs)
            channel_gradient_list += [channel_gradient]

        return channel_gradient_list

    @staticmethod
    def array_norm(array):
        """
        Calculates the norm of an n x m x p array for each p-length vector at index i,j
        """
        return norm(array, axis=2)
