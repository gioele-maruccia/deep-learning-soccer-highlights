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
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input
from keras.models import Model

class lstm:

    @staticmethod
    def build(num_frames, num_input_tokens):
        # initialize the model along with the input shape to be
        # "channels last"

        # if we are using "channels first", update the input shape
        input_shape = (num_frames, num_input_tokens)

        # model = Sequential(layers=[
        #
        #     Dense(1000, input_shape=input_shape),
        #
        #     LSTM(units=200, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
        #
        #     LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
        #
        #     LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, return_sequences=False),
        #
        #     # Dense(100),
        #
        #
        #     #Flatten(),
        #
        #     #Dense(100, activation='tanh', kernel_initializer='random_uniform')
        #
        # ])

        video = Input(shape=(num_frames, 25088))

        #cnn = Dense(10000, input_shape=(1,25088)) # GIOX
        #cnn.trainable = True

        cnn = Flatten(input_shape=(1, 25088)) # GIOXX

        encoded_frames = TimeDistributed(cnn)(video)
        encoded_sequence = LSTM(500, dropout=0.3, return_sequences=True)(encoded_frames)
        encoded_sequence = LSTM(200, dropout=0.3, return_sequences=True)(encoded_frames)

        encoded_sequence2 = LSTM(100, dropout=0.3, return_sequences=True)(encoded_sequence)

        # output = Dense(100)(encoded_sequence2) # GIOX
        output = LSTM(100, dropout=0.3, return_sequences=False)(encoded_sequence2)

        model = Model([video], output)

        # return the constucted network architecture
        return model
