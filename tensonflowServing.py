import os
import tensorflow as tf
import numpy as np
from keras.models import load_model

# Load the Keras model
model = load_model('models/model.h5')

version = 1
export_path = os.path.join('storage/mobilenet_model', str(version))
tf.saved_model.save(model, export_path)
