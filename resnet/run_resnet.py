import numpy as np
import pandas as pd
import math
import tensorflow as tf
import tensorflow.keras as tk

from keras.layers import Dense, Dropout, Flatten, Conv2D, Input, Add, \
                         Activation, ZeroPadding2D, BatchNormalization, \
                         AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
from keras import backend as K

import datetime
import pickle
import glob
import re

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

#******CUSTOM LEARNING RATE******#
class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate, epochs, steps_per_epoch):
    self.initial_learning_rate = initial_learning_rate
    self.steps_per_epoch = steps_per_epoch
    self.m = initial_learning_rate / steps_per_epoch
    self.decay_rate = (10**-8 / initial_learning_rate)**(((epochs - 1)*steps_per_epoch)**-1)
    print('decay_rate:', self.decay_rate)

  def __call__(self, step):
    result = tf.cond(tf.less(step, self.steps_per_epoch), 
                   lambda: self.m * (step+1),
                   lambda: self.initial_learning_rate * self.decay_rate**(step+1-self.steps_per_epoch))

    tf.print('lr at step', step, 'is', result, output_stream='file://learning_rates.txt')
    return result  


def ResNetJ(feature, epochs, steps_per_epoch, lr_power=-3.0):
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
    
    opt = tk.optimizers.Adam(learning_rate=MyLRSchedule(learning_rate, epochs, steps_per_epoch),
                             beta_1 = 0.9, beta_2 = 0.999 )

    model.compile(loss="mse", optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model



BATCH_SIZE = 64
EPOCHS = 15

#*******READ DATA*******#
ntrainfiles = 1

filelist = glob.glob('../../juno_data/data/projections/*.npz')
filelist = filelist[:ntrainfiles]
print(f'Creating tf.data.Dataset object reading {ntrainfiles} files...')
def get_data_from_filename(filename):
   labelfile = '../../juno_data/data/real/train/targets/targets_train_{}.csv'.format(re.findall('\d+', filename.decode())[0])
   labeldata = pd.read_csv(labelfile)
   labeldata = labeldata['edep'].to_numpy()

   npdata = np.load(filename, allow_pickle=True)['arr_0']
   npdata[tf.math.is_nan(npdata)] = 0.0

   return (npdata, labeldata)

def get_data_wrapper(filename):
   # Assuming here that both your data and label is double type
   features, labels = tf.numpy_function(
       get_data_from_filename, [filename], (tf.float64, tf.float64)) 
   return tf.data.Dataset.from_tensor_slices((features, labels))

# Create dataset of filenames.
ds = tf.data.Dataset.from_tensor_slices(filelist)
ds = ds.flat_map(get_data_wrapper).prefetch(tf.data.AUTOTUNE).batch(BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)


#******DEFINE MODEL******#
steps_per_epoch = int(math.ceil(5000*ntrainfiles / BATCH_SIZE))
print('Inferred steps per epoch:', steps_per_epoch)

res = ResNetJ(feature="energy", epochs=EPOCHS, steps_per_epoch=steps_per_epoch)
print(res.summary())

#******TENSORBOARD******#
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/" + date_time
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100)

#******TRAIN*******#
print('Training...')
history = res.fit(ds,
                  epochs=EPOCHS,
                  callbacks=[tensorboard_callback],
                  shuffle=True)


#*****SAVE MODEL******#
print('Saving fittet model, its weights and fit history...')
# save model weights
res.save_weights('models/res_weights_' + date_time + '.h5')
with open('models/res_mod-history_' + date_time + '.pkl', 'wb') as f:
    pickle.dump(history, f)
    pickle.dump(res, f)

print('ALL DONE!')