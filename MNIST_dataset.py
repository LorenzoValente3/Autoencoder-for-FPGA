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

    def __init__(self, data_fraction=1., zoom_factor = None):
        data = mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data.load_data()

        self.get_subset_of_data(data_fraction)

        self.convert_label_to_categorical()
        
        self.normalize_mnist_images()
        
        if zoom_factor is not None:
            self.interpolate(zoom_factor)

        self.reshape_to_color_channel()

        self.flatten_pictures()
    
    def interpolate(self, zoom_factor):
        self.x_train =  scipy.ndimage.zoom(self.x_train, 
                                            (1, zoom_factor, zoom_factor))
        self.x_test =  scipy.ndimage.zoom(self.x_test,
                                        (1, zoom_factor, zoom_factor))

    def convert_label_to_categorical(self):
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def normalize_mnist_images(self):
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def reshape_to_color_channel(self):
        self.x_train = self.x_train[:, :, :, np.newaxis]
        self.x_test = self.x_test[:, :, :, np.newaxis]

    def get_subset_of_data(self, data_fraction):
        """Choosing a fraction of data according to the machine capabilities"""
        index = int(len(self.x_train) * data_fraction)
        self.x_train = self.x_train[:index]
        self.y_train = self.y_train[:index]
        index = int(len(self.x_test) * data_fraction)
        self.x_test = self.x_test[:index]
        self.y_test = self.y_test[:index]
    
    def flatten_pictures(self):
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)