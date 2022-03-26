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
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.layers.wrappers import TimeDistributed

class vgg16_lstm:

    @staticmethod
    def build(num_frames, num_input_tokens):

        video = Input(shape=(num_frames, 224, 224, 3))

        cnn_base = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        cnn_out = GlobalAveragePooling2D()(cnn_base.output)
        cnn = Model(input=cnn_base.input, output=cnn_out)
        cnn.trainable = True
        set_trainable = False
        for layer in cnn_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        encoded_frames = TimeDistributed(cnn)(video)
        encoded_sequence = LSTM(256, return_sequences=True)(encoded_frames)
        output = LSTM(units=100, return_sequences=False)(encoded_sequence)
        model = Model([video], output)

        #x = LSTM(units=500, return_sequences=True, input_shape=input_shape)
        #x = LSTM(units=200, return_sequences=True)(x)
        #x = LSTM(units=100, return_sequences=True)(x)
        #output = LSTM(units=100, return_sequences=False)(x)

        return model
