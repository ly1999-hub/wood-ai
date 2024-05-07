import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image

# Load the model from the .h5 file
model = load_model('./models/model2.h5')
img = Image.open("./small_set_mahogany/5.jpg")
img = img.resize((50, 50))
img_array = np.array(img) / 255.0  # Chuẩn hóa giá trị pixel từ 0-255 thành 0-1

# Reshape ảnh để phù hợp với input shape của mô hình
img_array = np.expand_dims(img_array, axis=0)
# Assume you have a sample input data for testing # Replace input_shape with the actual input shape of your model

# Perform a prediction using the loaded model
prediction = model.predict(img_array)

# Print the prediction result
print(prediction)