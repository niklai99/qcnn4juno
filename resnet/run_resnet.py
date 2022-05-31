import numpy as np
import tensorflow as tf
import tensorflow.keras as tk

from keras.layers import Dense, Dropout, Flatten, Conv2D, Input, Add, \
                         Activation, ZeroPadding2D, BatchNormalization, \
                         AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform


"""
First of all we define the two kind of blocks of the ResNet as functions.
"""
def conv1(X, filters = 32 , block="conv1", stage=1):

    """ 
    This is not a Residual block! 
    It's composed by two Conv2D layers with two different kernel size (6,3) , (3,3)
    Batch normalization and one MaxPooling at the end
    Arguments:
    - X = the output of the previous layer or the input-shape of the Net
    - filters = number of filters of the 2 convolutional blocks
    Returns: Keras layer 
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    X_input = X
    # First component of main path
    X = Conv2D(filters, kernel_size = (6,3), strides = (2,1),
            name = conv_name_base + '2a',
            )(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters, kernel_size = (3,3), strides = (2,2),
            name = conv_name_base + '2b',
            )(X)
    X = BatchNormalization(axis=1, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3,3), strides=(1, 1))(X)
    
    return X


def conv2x(X, stride, filters, block, stage=2):

    """ 
    It's a Residual block!
    The residual operation is make up of a stack of convolutions
    It's composed by 3 Conv2D layers 
    Arguments:
        - X = the output of the previous layer or the input-shape of the Net
        - filters = list of 2 numbers (int) of filters of the convolutional blocks
            --> the first 2 and the last 2 have the same f 
        - stride = the stride of the "skip path" and the second "Full path" Conv2D
        - block = the name of the Residual block
    Returns: Keras layer 
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
                
    f1,f2 = filters
    s = stride
    #skip path for the residual part ("skip path")
    X_shortcut = Conv2D(f2, kernel_size = (3,3), strides = (s,s),
            name = conv_name_base + '-shortcut',
            )(X)

    X_shortcut = BatchNormalization(axis=1, name = bn_name_base + '-shortcut')(X_shortcut)

    # First component of "Full path"
    X = Conv2D(f1, kernel_size = (1,1), strides = (1,1),
            name = conv_name_base + '2a',
            )(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of "Full path"
    X = Conv2D(f1, kernel_size = (3,3), strides = (s,s),
            name = conv_name_base + '2b',
            )(X)
    X = BatchNormalization(axis=1, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    # Third component of "Full path"
    X = Conv2D(f2, kernel_size = (1,1), strides = (1,1),
            name = conv_name_base + '2c',
            )(X)
    X = BatchNormalization(axis=1, name = bn_name_base + '2c')(X)
    X = Activation('relu')(X)

    X = Add()([X, X_shortcut])
    out = Activation('relu')(X)

    return out


def ResNetJ(feature, lr_power=-3.0, lr_decay=0.0):
    """
    Implementation of the ResNet with architecture:        
    Arguments:
        feature is a string which allow two input values:
        --> 'vertex' or 'energy'
    """
    
    # Define the input as a tensor with shape input_shape
    # X_input = Input(shape=(230,124,2))
    X_input = Input(shape=(219,122,2))
    
    # Stage 1
    X = conv1(X_input, block="conv1")
    # Stage2
    X = conv2x(X, stride=1, filters=[32,128], block="conv2x_1")
    X = conv2x(X, stride=1, filters=[32,128], block="conv2x_2")
    X = conv2x(X, stride=1, filters=[32,128], block="conv2x_3")

    # Stage 3
    # X = conv2x(X, stride=2, filters=[64,256], block="conv3x_1")
    X = conv2x(X, stride=1, filters=[64,256], block="conv3x_2")
    X = conv2x(X, stride=1, filters=[64,256], block="conv3x_3")
    X = conv2x(X, stride=1, filters=[64,256], block="conv3x_4")

    # Stage 4 
    # X = conv2x(X, stride=2, filters=[128,512], block="conv4x_1")
    X = conv2x(X, stride=1, filters=[128,512], block="conv4x_2")
    X = conv2x(X, stride=1, filters=[128,512], block="conv4x_3")
    X = conv2x(X, stride=1, filters=[128,512], block="conv4x_4")
    X = conv2x(X, stride=1, filters=[128,512], block="conv4x_5")
    X = conv2x(X, stride=1, filters=[128,512], block="conv4x_6")

    # Stage 5
    X = conv2x(X, stride=2, filters=[256,1024], block="conv5x_1")
    X = conv2x(X, stride=1, filters=[256,1024], block="conv5x_2")
    X = conv2x(X, stride=1, filters=[256,1024], block="conv5x_3")

    # AVGPOOL 
    X = AveragePooling2D((2,2), name='avg_pool')(X)
    # Flatten
    X = Flatten()(X)
    X = Dense(512, name='first_dense',  kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(100, name='second_dense', kernel_initializer=glorot_uniform(seed=0))(X)

    # Output 
    if(feature=="energy"):
        X = Dense(1, name='fc_outputs', kernel_initializer=glorot_uniform(seed=0))(X)
    elif(feature=="vertex"):
        X = Dense(3, name='fc_outputs', kernel_initializer=glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name = 'ResNetJ')
    
    # Compile model
    learning_rate = 10.0**(lr_power)
    opt = tk.optimizers.Adam(learning_rate=learning_rate, beta_1 = 0.9, beta_2 = 0.999 )

    model.compile(loss="mean_squared_error", optimizer=opt, metrics=['accuracy'])
    
    return model

from keras.callbacks import LearningRateScheduler

def step_decay_schedule(initial_lr=1e-8, decay_factor=0.75, step_size=10, BS=64, ndat = 5e6):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        if epoch==1:
            decay_factor = 10**(-3) * BS / ndat
            lr_sched = initial_lr * decay_factor     #(decay_factor ** np.floor(epoch/step_size))
            return lr_sched
        else:
            decay_factor = 0.1
            lr_sched = tk.optimizers.schedules.ExponentialDecay(initial_learning_rate=10**(-3) )
            return lr_sched
#        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr=1e-4)

res = ResNetJ(feature="energy")
res.summary()

BATCH_SIZE = 64
EPOCHS = 15
history = res.fit(x_train, y_train,
         batch_size=BATCH_SIZE,
         epochs=EPOCHS,
         callbacks=[lr_sched],
         validation_data=(x_val, y_val),
         shuffle=True)