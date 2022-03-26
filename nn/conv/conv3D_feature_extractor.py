''' in order to use this nn it has to be well investigated how we can handle so much data in our model.
At this, it is by no means possible to train this specific neural network, althouth it has been proved to be the state-of-the-art condition to train
an action recognition model for videos'''

# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import MaxPooling3D

from keras import backend as K


class Conv3D_feature_extractor:

    @staticmethod
    def build(length, width, height, depth):
        # initialize the model along with the input shape to be
        # "channels last"

        # if we are using "channels first", update the input shape
        input_shape = (length, height, width, depth)

        if K.image_data_format() == "channels_first":
            input_shape = (length, depth, height, width)

        model = Sequential(layers=[

        Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',
                         input_shape=input_shape),

        MaxPooling3D(pool_size=(2, 2, 2)),

        #Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'),

        #MaxPooling3D(pool_size=(2, 2, 2)),

        #Conv3D(256, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'),

        #Conv3D(256, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'),

        #MaxPooling3D(pool_size=(2, 2, 2)),

        # Conv3D(512, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'),
        
        #Conv3D(512, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'),

        #MaxPooling3D(pool_size=(2, 2, 2)),
        
        #Conv3D(512, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'),

        #Conv3D(512, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'),

        #MaxPooling3D(pool_size=(2, 2, 2)),

        #Dense(4096, activation=None, kernel_initializer='glorot_uniform'),

        #Dense(4096, activation=None, kernel_initializer='glorot_uniform'),

        Flatten(),

        Dense(1, activation=None, kernel_initializer='glorot_uniform' )
        ])

        # return the constucted network architecture
        return model
