from classes.data_handling.image import Image

img_path = '/home/ron/source/deconv_hyper_laplace/examples/example_images/ex.png'
image = Image()
image.calculate_gradient_information()

import numpy as np
import cv2
img = cv2.imread(img_path, 0)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
gradient_vals = laplacian.flatten()


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.plot(X2, np.gradient(F2) / np.gradient(X2))
plt.savefig('fig')