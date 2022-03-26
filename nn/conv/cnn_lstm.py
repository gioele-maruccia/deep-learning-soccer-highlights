''' '''

# import the necessary packages
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Bidirectional
from keras.layers import LSTM

from keras import backend as K
from keras.layers import Input
from keras import layers
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.layers.wrappers import TimeDistributed

class cnn_lstm:

    @staticmethod
    def build(num_frames, num_input_tokens):

        video = Input(shape=(num_frames, 64, 64, 3))

        cnn_base = Sequential()
        cnn_base.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
        cnn_base.add(Activation('relu'))
        cnn_base.add(MaxPooling2D(pool_size=(2, 2)))

        cnn_base.add(Conv2D(32, (3, 3)))
        cnn_base.add(Activation('relu'))
        cnn_base.add(MaxPooling2D(pool_size=(2, 2)))

        cnn_base.add(Conv2D(64, (3, 3)))
        cnn_base.add(Activation('relu'))
        cnn_base.add(MaxPooling2D(pool_size=(2, 2)))

        cnn_out = Flatten()(cnn_base.output)
        cnn = Model(input=cnn_base.input, output=cnn_out)
        cnn.trainable = True

        encoded_frames = TimeDistributed(cnn)(video)
        encoded_sequence = LSTM(256, return_sequences=True)(encoded_frames)
        output = LSTM(units=100, return_sequences=False)(encoded_sequence)
        model = Model([video], output)

        #x = LSTM(units=500, return_sequences=True, input_shape=input_shape)
        #x = LSTM(units=200, return_sequences=True)(x)
        #x = LSTM(units=100, return_sequences=True)(x)
        #output = LSTM(units=100, return_sequences=False)(x)

        return model
