import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tk

from keras.layers import Dense, Dropout, Flatten, Conv2D, Input, Add, \
                         Activation, ZeroPadding2D, BatchNormalization, \
                         AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
from keras import backend as K

import datetime

#*****INITIALIZE GPU****#
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""
Define costum loss
"""
def mean_L2_loss(y_pred, y_true):
    """
    Mean L2-norm regression loss
    
    Parameters
    ----------
    y_true : array-like of shape = (n_samples, vec_dim)
    y_pred : array-like of shape = (n_samples, vec_dim)
    Returns
    -------
    Loss : A positive floating point value, the best value is 0.0.
    """
    d = y_pred - y_true
    return tf.reduce_mean(tf.norm(d, axis=1))

"""
Define costum metric
"""
def rmse(y_pred, y_true):
    """
    Root-mean-square-error metrics
    
    Parameters
    ----------
    y_true : array-like of shape = (n_samples, vec_dim)
    y_pred : array-like of shape = (n_samples, vec_dim)
    Returns
    -------
    Metrics : A positive floating point value, the best value is 0.0.
    """
    d = y_pred - y_true
    return K.sqrt(K.mean(K.square(tf.norm(d, axis=1))))

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


def ResNetJ(feature, lr_power=-3.0, lr_decay=0.0, nepochs=15, ndat=5e6, BS=64):
    """
    Implementation of the ResNet with architecture:        
    Arguments:
        feature is a string which allow two input values:
        --> 'vertex' or 'energy'
    """
    
    # Define the input as a tensor with shape input_shape
    # X_input = Input(shape=(230,124,2))
    X_input = Input(shape=(230,124,2))
    
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
    decay_factor = 10**(-5*nepochs*ndat/BS)
    opt = tk.optimizers.Adam(learning_rate=learning_rate #tk.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                          #                          decay_rate=decay_factor, 
                                                           #                         decay_steps=1), 
                             ,beta_1 = 0.9, beta_2 = 0.999 )

    model.compile(loss="mse", optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model

from keras.callbacks import LearningRateScheduler
'''
def step_decay_schedule(BS=64, ndat = 5e6, nepochs=15):
    
    #Wrapper function to create a LearningRateScheduler with step decay schedule.
    
    def schedule(epoch, lr):
        if epoch==1:
            decay_factor = 10**(-3) * BS / ndat
            #lr_sched = lr * decay_factor     #(decay_factor ** np.floor(epoch/step_size))
            return lambda step: decay_factor*step
        else:
            decay_factor = 10**(-5*(nepochs-1)*ndat/BS)
            lr_sched = tk.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_rate=decay_factor, decay_steps=1)
            return lr_sched
#        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)
'''


#*******READ DATA*******#
print('Reading data...', end='')
filename  = '../../juno_data/data/projections/proj_raw_data_train_0.npz'
labelname = '../../juno_data/data/real/train/targets/targets_train_0.csv'
x_train = np.load(filename, allow_pickle=True)['arr_0']

# nan to zero 
x_train[np.isnan(x_train)] = 0

y_train = pd.read_csv(labelname)
y_train = y_train['edep'].to_numpy()
print(' Done')

#******DEFINE MODEL******#
BATCH_SIZE = 64
EPOCHS = 40
res = ResNetJ(feature="energy", BS=BATCH_SIZE, ndat=x_train.shape[0], nepochs=EPOCHS)
print(res.summary())

#******TENSORBOARD******#
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/" + date_time
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#******TRAIN*******#
print('Training...')
history = res.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          #callbacks=[tensorboard_callback],
          validation_split=0.3,
          shuffle=True)


#*****SAVE MODEL******#
import pickle


# save model weights
res.save_weights('models/res_weights_' + date_time + '.h5')
with open('models/res_history_' + date_time + '.pkl', 'wb') as f:
    pickle.dump(history, f)
    pickle.dump(res, f)