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
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.layers.wrappers import TimeDistributed

class mobilenet2_lstm:

    @staticmethod
    def build(num_frames, num_input_tokens):

        video = Input(shape=(num_frames, 96, 96, 3))

        cnn_base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(96, 96, 3))
        cnn_out = GlobalAveragePooling2D()(cnn_base.output)
        cnn = Model(input=cnn_base.input, output=cnn_out)
        cnn.trainable = True
        for layer in cnn.layers[:-5]:
            layer.trainable = False

        encoded_frames = TimeDistributed(cnn)(video)
        encoded_sequence = LSTM(200, return_sequences=True)(encoded_frames)
        output1 = LSTM(units=100, return_sequences=True)(encoded_sequence)
        output = LSTM(units=100, return_sequences=False)(output1)
        model = Model([video], output)

        #x = LSTM(units=500, return_sequences=True, input_shape=input_shape)
        #x = LSTM(units=200, return_sequences=True)(x)
        #x = LSTM(units=100, return_sequences=True)(x)
        #output = LSTM(units=100, return_sequences=False)(x)

        return model
