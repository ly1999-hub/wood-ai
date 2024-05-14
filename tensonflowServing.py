import os
import tensorflow as tf
import numpy as np
from keras.models import load_model

# Load the Keras model
model = load_model('models/model.h5')
version = 1
export_path = os.path.join('storage/mobilenet_model', str(version))
tf.saved_model.save(model, export_path)

model_prev = load_model('models/model_prev.h5')
version2=1
export_path2 = os.path.join('storage/mobilenet_model_1', str(version2))
tf.saved_model.save(model_prev, export_path2)

# docker run -d -p 8500:8500 --name=tensorflow_serving --mount type=bind,source=D:\ML\CNN\tsjs\storage\mobilenet_model,target=/models/mobilenet_model -e MODEL_NAME=wood --mount type=bind,source=D:\ML\CNN\tsjs\storage\mobilenet_model_1,target=/models/mobilenet_model_1  -e  MODEL_NAME=wood1 tensorflow/serving

# docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=D:\ML\CNN\woodAI\storage\mobilenet_model_1,target=/models/wood1 -e MODEL_NAME=wood1 --mount type=bind,source=D:\ML\CNN\woodAI\storage\mobilenet_model,target=/models/wood -e MODEL_NAME=wood -t tensorflow/serving                                                           

# docker run -p 8601:8601 -p 8600:8600 --mount type=bind,source=D:\ML\CNN\woodAI\storage\mobilenet_model_1,target=/models/wood1 -e MODEL_NAME=wood1 -t tensorflow/serving
# docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=D:\ML\CNN\woodAI\storage\mobilenet_model,target=/models/wood -e MODEL_NAME=wood -t tensorflow/serving