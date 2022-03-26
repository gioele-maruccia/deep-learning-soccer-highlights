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


class lstm2:

    @staticmethod
    def build(num_frames, num_input_tokens):
        # initialize the model along with the input shape to be
        # "channels last"

        # if we are using "channels first", update the input shape
        input_shape = (num_frames, num_input_tokens)

        model = Sequential(layers=[
            LSTM(units=1, return_sequences=True, input_shape=input_shape)

        ])

        # return the constucted network architecture
        return model
