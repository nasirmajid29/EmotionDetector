import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, preprocessing, models, optimizers, callbacks, utils
from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import adam_v2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
from IPython.display import SVG, Image
from livelossplot.inputs.tf_keras import PlotLossesCallback

print("Tensorflow version:", tf.__version__)
