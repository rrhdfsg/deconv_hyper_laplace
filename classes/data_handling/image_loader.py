from cv2 import imread

class ImageLoader():
    '''
    A class that, when instantiated, loads in an image from a path and converts it to a python array
    '''
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_array = None
        self.load_image()

    def load_image(self):
        self.image_array = imread(self.image_path)