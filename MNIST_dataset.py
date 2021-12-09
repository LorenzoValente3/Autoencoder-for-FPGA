from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import scipy


# +------------------+
# | MNIST data class |
# +------------------+


class MNISTData:
    """MNIST data class. You can adjust the data_fraction to use when creating
    the data, according to your system capabilities."""

    def __init__(self, data_fraction = 1., size_initial = 20, size_final = 8, color_depth = 5, flat = True):
        data = mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data.load_data()

        self.get_subset_of_data(data_fraction)

        self.convert_label_to_categorical()
        
        #self.normalize_mnist_images()
        
        

        self.crop_and_interpolate(size_initial, size_final, color_depth)
          
        #self.reshape_to_color_channel()

        #self.crop_center( 22)
        
        #if flat is True:
        #    self.flatten_pictures()

    def get_subset_of_data(self, data_fraction):
        """Choosing a fraction of data according to the machine capabilities"""
        index = int(len(self.x_train) * data_fraction)
        self.x_train = self.x_train[:index]
        self.y_train = self.y_train[:index]
        index = int(len(self.x_test) * data_fraction)
        self.x_test = self.x_test[:index]
        self.y_test = self.y_test[:index]

    def convert_label_to_categorical(self):
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def crop_and_interpolate(self, size_initial, size_final, color_depth):
        #defisce gli indici del cropping definito da size_initial
        bordo = (28-size_initial)//2
        bordo_top = -bordo
        if bordo == 0:
            bordo_top = None

        X_train_flat_zoom = []
        X_test_flat_zoom = []
        
        X_train_flat_zoom_int = []
        X_test_flat_zoom_int = []

        for image in self.x_train:
            tmp = scipy.ndimage.zoom(image[bordo:bordo_top, bordo:bordo_top],
                                     size_final/size_initial).flatten()
            tmp = (tmp/(256//2**color_depth)).astype(int)
            X_train_flat_zoom.append(tmp/2**color_depth)
            X_train_flat_zoom_int.append(tmp)

        X_train_flat_zoom = np.array(X_train_flat_zoom)
        X_train_flat_zoom_int = np.array(X_train_flat_zoom_int)

        self.x_train=X_train_flat_zoom

        #processing Test Set
        for image in self.x_test:
            tmp = scipy.ndimage.zoom(image[bordo:bordo_top, bordo:bordo_top], 
                                        size_final/size_initial).flatten()
            tmp = (tmp/(256//2**color_depth)).astype(int)
            X_test_flat_zoom.append(tmp/2**color_depth)
            X_test_flat_zoom_int.append(tmp)

        X_test_flat_zoom = np.array(X_test_flat_zoom)
        X_test_flat_zoom_int = np.array(X_test_flat_zoom_int)
        self.x_test=X_test_flat_zoom





    def normalize_mnist_images(self):
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def reshape_to_color_channel(self):
        self.x_train = self.x_train[:, :, :, np.newaxis]
        self.x_test = self.x_test[:, :, :, np.newaxis]

    
    
    def flatten_pictures(self):
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)