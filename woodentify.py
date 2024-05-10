import cv2
from matplotlib import pyplot as plt
import numpy as np
from img_preprocess import *
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
#from keras.layers.convolutional import Conv2D
from keras.layers import Conv2D

from keras.backend import image_data_format
from sklearn.model_selection import train_test_split
np.random.seed(1337)  # for reproducibility
import sys

if __name__ == '__main__':
    img_rows, img_cols = 50, 50 # input image dimensions
    folder_list = ["Go","GoBachDan","GoLim","GoSoi","GoThong","GoTrac","GoTram","GoXoan"]
    X, y = prep_total_pipeline(folder_list, img_rows, limit=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    batch_size = 100
    num_classes = len(folder_list)
    num_epochs = 10
    nb_filters = 32 # number of convolutional filters to use
    pool_size = (2, 2) # size of pooling area for max pooling
                       # decreases image size, and helps to avoid overfitting

    if image_data_format == 'th':
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[3], img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[3], img_rows, img_cols)
        input_shape = (X_test.shape[3], img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, X_train.shape[3])
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, X_test.shape[3])
        input_shape = (img_rows, img_cols, X_test.shape[3])

    # don't change conversion or normalization
    X_train = X_train.astype('float32') # data was uint8 [0-255]
    X_test = X_test.astype('float32')  # data was uint8 [0-255]
    X_train /= 255 # normalizing (scaling from 0 to 1)
    X_test /= 255  # normalizing (scaling from 0 to 1)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices (don't change)
    Y_train = to_categorical(y_train, num_classes) # cool
    Y_test = to_categorical(y_test, num_classes)   # cool * 2

    model = Sequential()
    model.add(Conv2D(nb_filters, (5, 5),
                        padding='valid',
                        input_shape=input_shape))
    model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers
    model.add(Conv2D(nb_filters, (3,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Flatten()) # necessary to flatten before going into conventional dense layer (keep layer)
    print('Model flattened out to ', model.output_shape)
    
    # now start a typical neural network
    model.add(Dense(128))
    model.add(Activation('tanh'))
    model.add(Dropout(0.15))
    model.add(Dense(num_classes)) # 10 final nodes (one for each class) (keep layer)
    model.add(Activation('softmax')) # keep softmax at end to pick between classes 0-9
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    checkpoit =ModelCheckpoint("models/best.hdf5",monitor='val_loss',save_best_only=True,mode='auto')
    # during fit process watch train and test error simultaneously
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs,
                verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1]) # this is the one we care about

    model.save('models/model.h5')

# /home/huuly/wood-ai/storage/mobilenet_model
# sudo docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=/home/huuly/wood-ai/storage/mobilenet_model,target=/models/wood -e MODEL_NAME=wood -t emacski/tensorflow-serving