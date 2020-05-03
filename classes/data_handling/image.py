from classes.data_handling.image_loader import ImageLoader
from classes.math_tools.gradient_norm import GradientNorm

class Image():
    '''
    A class that, when instantiated, loads in and stores image information to be manipulated and
    have standard necessary calculations executed on it
    '''
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_array = None
        self.gradient = None
        self.image_gradient_norm = None
        self.image_loader = ImageLoader(self.image_path)
        self.image_array = self.image_loader.image_array

    def calculate_gradient_information(self):
        self.gradient = GradientNorm(self.image_array)
        self.image_gradient_norm = self.gradient.channel_gradient_norm


