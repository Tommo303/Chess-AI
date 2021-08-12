from numpy import array
from copy import copy
import chess
from keras.layers import Conv2D, BatchNormalization, Dense, Input, Add, ReLU, Flatten
from keras.models import Model


def preprocess(data):
    piece_integer_values = {'.': 0, 
                            'P': 1, 
                            'p': 2,
                            'R': 3,
                            'r': 4,
                            'N': 5,
                            'n': 6,
                            'B': 7,
                            'b': 8,
                            'Q': 9,
                            'q': 10,
                            'K': 11,
                            'k': 12
                            }
    
    for i in range(len(data)):
        board = list(str(chess.Board()).replace(' ', '').replace('\n', ''))
    
        for j in range(len(board)):
            board[j] = piece_integer_values[board[j]]
            
        board = array(board).reshape((8, 8))
    
    return data

def convolutional_layer(y, num_filters, filter_size):
    # 256 Convolution Filters (3x3)
    y = Conv2D(num_filters, filter_size, input_shape=y.shape[1:], padding='same')(y)
    
    # Batch Normalisation
    y = BatchNormalization()(y)
    
    # Recitified Linear Unit (ReLU)
    y = ReLU()(y)
    
    return y

def residual_layer(y, num_filters, filter_size):
    skip = copy(y)
    
    # 256 Convolution Filters (3x3)
    y = Conv2D(num_filters, filter_size, input_shape=y.shape[1:], padding='same')(y)
    
    # Batch Normalisation
    y = BatchNormalization()(y)
    
    # Recitified Linear Unit (ReLU)
    y = ReLU()(y)
    
    # 256 Convolution Filters (3x3)
    y = Conv2D(num_filters, filter_size, input_shape=y.shape[1:], padding='same')(y)
    
    # Batch Normalisation
    y = BatchNormalization()(y)
    
    # Skip Connection
    if skip.shape == y.shape:
        y = Add()([y, skip])
        
    # Recitified Linear Unit (ReLU)
    y = ReLU()(y)
       
    return y

def value_head(y, num_filters, filter_size):
    # 1 Convolution Filters (1x1)
    y = Conv2D(num_filters, filter_size, input_shape=y.shape[1:], padding='valid')(y)
    
    # Batch Normalisation
    y = BatchNormalization()(y)
    
    # Flatten
    y = Flatten()(y)
    
    # Recitified Linear Unit (ReLU)
    y = ReLU()(y)
    
    # Fully Connected Layer
    y = Dense(64)(y)
    
    # Recitified Linear Unit (ReLU)
    y = ReLU()(y)
    
    # Fully Connected Layer
    y = Dense(64)(y) 
    
    # Fully Connected Layer - Tanh Activation
    y = Dense(1, activation='tanh')(y)
    
    return y

def policy_head(y, num_filters, filter_size, output_shape):
    # 2 Convolution Filters (1x1)
    y = Conv2D(num_filters, filter_size, input_shape=y.shape[1:], padding='valid')(y)
    
    # Batch Normalisation
    y = BatchNormalization()(y)
    
    # Recitified Linear Unit (ReLU)
    y = ReLU()(y)
    
    # Flatten
    y = Flatten()(y)
    
    # Fully Connected Layer - Softmax Activation
    y = Dense(output_shape, activation='softmax')(y)
    
    return y


x = Input((8, 8, 1))

y = convolutional_layer(x, 256, (3, 3))

for i in range(40):
    y = residual_layer(y, 256, (3, 3))

value = value_head(y, 1, (1, 1))
policy = policy_head(y, 2, (1, 1), 35)

    
model = Model(inputs=x, outputs=[value, policy])

# Compile model
model.compile(
  'SGD',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

