import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow import keras

import cirq
import sympy
import numpy as np
import collections
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

from matplotlib.ticker import ScalarFormatter

from scipy.optimize import curve_fit

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow_quantum as tfq

import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# train
## data
proj_train_path = "../../juno_data/data/projections/reduced/train"
train_files     = os.listdir(proj_train_path)
train_proj   = [file for file in train_files if "npy" in file]
train_proj.sort(key=natural_keys)

## labels
target_train_path = "../../juno_data/data/real/train/targets"
train_target_files = os.listdir(target_train_path)
train_target       = [file for file in train_target_files if "targets" in file]
train_target.sort(key=natural_keys)
train_target = train_target[:20]

# test
energies = ['0', '0.1', '0.3', '0.6'] + [str(i) for i in range(1, 11)]

## data & labels
proj_test_path = "../../juno_data/data/projections/reduced/test"
target_test_path = "../../juno_data/data/real/test/targets"

test_proj_dict   = { e:None for e in energies}
test_target_dict = { e:None for e in energies}
for i, e in enumerate(energies):
    proj_files_path   = proj_test_path + "/e+_" + e
    target_files_path = target_test_path + "/e+_" + e
    
    proj_files   = os.listdir(proj_files_path)
    target_files = os.listdir(target_files_path)
    
    test_proj_dict[e]    = [file for file in proj_files if "npy" in file] 
    test_target_dict[e]  = [file for file in target_files if "targets" in file]
    
    test_proj_dict[e].sort(key=natural_keys)
    test_target_dict[e].sort(key=natural_keys)
    
    test_target_dict[e] = test_target_dict[e][:50]

# train
## data

print("Loading train set ...")

#mat = np.empty(shape=(0, 57, 31, 2))
f=list()
for file in train_proj:
    f.append(np.load(proj_train_path + '/' + file))

mat = np.concatenate(f, axis=0)

train_images = mat

max_time = 0
max_charge = 0

for image in train_images:

    ev_max_time = np.max(image[:, :, 1])
    ev_max_charge = np.max(image[:, :, 0])
    if ev_max_time > max_time:
        max_time = ev_max_time
    if ev_max_charge > max_charge:
        max_charge = ev_max_charge

train_images[:, :, :, 0] = train_images[:, :, :, 0]/max_charge
train_images[:, :, :, 1] = train_images[:, :, :, 1]/max_time

## labels
train_labels = np.empty(shape = (0))
for i, file in enumerate(train_target):
    f = np.array(pd.read_csv(target_train_path + '/' + file)["edep"])
    train_labels = np.concatenate((train_labels, f), axis=0)

print("done!")

'''# test
## data

print("Loading test set ...")

test_images = {e:None for e in energies}

for e in energies:
    f = list()
    for file in test_proj_dict[e]:
        f.append(np.load(proj_test_path + "/e+_" + e + '/' + file))
    test_images[e] = np.concatenate(f, axis=0)
    del f

## labels
test_labels = {e:None for e in energies}
for e in energies:
    test_labels[e] = np.empty(shape = (0))
    for file in test_target_dict[e]:
        f = np.array(pd.read_csv(target_test_path + "/e+_" + e + '/' + file)["edep"])
        test_labels[e] = np.concatenate((test_labels[e], f), axis=0)


print("done!")'''

debug=False
class QConv(tf.keras.layers.Layer):
    def __init__(self, filter_size, depth, strides=(1, 1), activation=None, name=None, kernel_regularizer=None, **kwangs):
        super(QConv, self).__init__(name=name, **kwangs)
        self.filter_size = (filter_size, filter_size) if type(filter_size)==int else filter_size
        self.pixels = self.filter_size[0]*self.filter_size[1]
        self.depth = depth
        self.strides = strides
        self.learning_params = []
        self.QCNN_layer_gen()
        # self.circuit_tensor = tfq.convert_to_tensor([self.circuit])
        self.activation = tf.keras.layers.Activation(activation)
        self.kernel_regularizer = kernel_regularizer

    def _next_qubit_set(self, original_size, next_size, qubits):
        step = original_size // next_size
        qubit_list = []
        for i in range(0, original_size, step):
            for j in range(0, original_size, step):
                qubit_list.append(qubits[original_size*i + j])
        return qubit_list

    def _get_new_param(self):
        """
        return new learnable parameter
        all returned parameter saved in self.learning_params
        """
        new_param = sympy.symbols("p"+str(len(self.learning_params)))
        self.learning_params.append(new_param)
        return new_param
    
    def _QConv(self, step, target, qubits):
        """
        apply learnable gates each quantum convolutional layer level
        """
        yield cirq.CZPowGate(exponent=self._get_new_param())(qubits[target], qubits[target+step])
        yield cirq.CXPowGate(exponent=self._get_new_param())(qubits[target], qubits[target+step])
    
    def QCNN_layer_gen(self):
        """
        make quantum convolutional layer in QConv layer
        """
        cirq_qubits = cirq.GridQubit.rect(self.filter_size[0]*self.depth, self.filter_size[1])      # a circuit for each depth
        # mapping input data to circuit
        input_circuit = cirq.Circuit()
        input_params = [sympy.symbols('a%d' %i) for i in range(self.pixels)]
        for i, qubit in enumerate(cirq_qubits):
            input_circuit.append(cirq.rx(np.pi*input_params[i%self.pixels])(qubit))                                 # input params repeats for each depth
        # apply learnable gate set to QCNN circuit
        QCNN_circuit = cirq.Circuit()
        step_size = [2**i for i in range(np.log2(self.pixels).astype(np.int32))]                                    # add for in depth
        
        if np.log2(self.pixels) % 1 == 0:
            for i in range(self.depth):
                skip = i*self.pixels
                for step in step_size:
                    for target in range(0, self.pixels, 2*step):
                        QCNN_circuit.append(self._QConv(step, target+skip, cirq_qubits))                        # split the circuits
        else:
            for i in range(self.depth):
                skip = i*self.pixels
                for target in range(self.pixels-1):
                    QCNN_circuit.append(self._QConv(1, target+skip, cirq_qubits))                               # split the circuits
                QCNN_circuit.append(self._QConv(1-self.pixels, self.pixels-1+skip, cirq_qubits))
                
        # merge the circuits
        full_circuit = cirq.Circuit()
        full_circuit.append(input_circuit)
        full_circuit.append(QCNN_circuit)
        self.circuit = full_circuit # save circuit to the QCNN layer obj.
        
        self.params = input_params + self.learning_params
        self.op = [cirq.Z(cirq_qubits[(i)]) for i in range(len(cirq_qubits))]                      # measure
        
    def build(self, input_shape):
        self.width = input_shape[1]
        self.height = input_shape[2]
        self.channel = input_shape[3]
        self.num_x = (self.width - self.filter_size[0]) // self.strides[0] + 1
        self.num_y = (self.height - self.filter_size[1]) // self.strides[1] + 1
        if(((self.width - self.filter_size[0]) % self.strides[0]) or 
           ((self.height - self.filter_size[1]) % self.strides[1])):
            if(((self.width - self.filter_size[0]) % self.strides[0]) and 
               ((self.height - self.filter_size[1]) % self.strides[1])):
                print("WARNING: cutting image borders, consider changing filter size or stride on both dimensions")    
            elif((self.height - self.filter_size[1]) % self.strides[1]):
                print("WARNING: cutting image borders, consider changing filter size or stride on dimension 1")
            else:
                print("WARNING: cutting image borders, consider changing filter size or stride along dimension 0")
                
        self.kernel = self.add_weight(name="kenel",                                                                     # careful with self.learning_params
                                      shape=[self.channel, 
                                             len(self.learning_params)],
                                     initializer=tf.keras.initializers.glorot_normal(),
                                     regularizer=self.kernel_regularizer)
        self.circuit_tensor = tfq.convert_to_tensor([self.circuit] * self.num_x * self.num_y * self.channel)
        if debug:
            print('circuit_tensor:', self.circuit_tensor.shape)
        

    def call(self, inputs):
        # input shape: [N, width, height, channel]
        # slide and collect data
        stack_set = None
        for i in range(0, self.width-self.filter_size[0]+1, self.strides[0]):
            for j in range(0, self.height-self.filter_size[1]+1, self.strides[1]):
                slice_part = tf.slice(inputs, [0, i, j, 0], [-1, self.filter_size[0], self.filter_size[1], -1])
                slice_part = tf.reshape(slice_part, shape=[-1, 1, self.filter_size[0], self.filter_size[1], self.channel])
                if debug:
                    print('i, j:', i, j)
                    print('slice_part:', slice_part.shape)
                if stack_set == None:
                    stack_set = slice_part
                else:
                    stack_set = tf.concat([stack_set, slice_part], 1)
                del slice_part
        if debug:
            print('stack_set:', stack_set.shape) 
        # -> shape: [N, num_x*num_y, filter_size, filter_size, channel]
        stack_set = tf.transpose(stack_set, perm=[0, 1, 4, 2, 3])
        # -> shape: [N, num_x*num_y, channel, filter_size, fiter_size]
        stack_set = tf.reshape(stack_set, shape=[-1, self.filter_size[0]*self.filter_size[1]])
        # -> shape: [N*num_x*num_y*channel, filter_size^2]
        if debug:
            print(stack_set.shape, stack_set)
        # total input circuits: N * num_x * num_y * channel
        circuit_inputs = tf.tile([self.circuit_tensor], [tf.shape(inputs)[0], 1])
        circuit_inputs = tf.reshape(circuit_inputs, shape=[-1])
        #tf.fill([tf.shape(inputs)[0]*self.num_x*self.num_y, 1], 1)
        
        controller = tf.tile(self.kernel, [tf.shape(inputs)[0]*self.num_x*self.num_y, 1])
        outputs=self.single_depth_QCNN(stack_set, controller, circuit_inputs)
        # shape: [N, num_x, num_y, self.depth] 
        del stack_set
        del controller
        del circuit_inputs
            
        output_tensor = tf.math.acos(tf.clip_by_value(outputs, -1+1e-5, 1-1e-5)) / np.pi
        del outputs
        # output_tensor = tf.clip_by_value(tf.math.acos(output_tensor)/np.pi, -1, 1)
        return self.activation(output_tensor)
    
    def single_depth_QCNN(self, input_data, controller, circuit_inputs):
        """
        make QCNN for 1 channel only
        """
        # input shape: [N*num_x*num_y*channel, filter_size^2]
        # controller shape: [N*num_x*num_y*channel, len(learning_params)]
        input_data = tf.concat([input_data, controller], 1)
        # input_data shape: [N*num_x*num_y*channel, len(learning_params)]
        QCNN_output = tfq.layers.Expectation()(circuit_inputs, 
                                               symbol_names=self.params,
                                               symbol_values=input_data,
                                               operators=self.op)
        del input_data
        # QCNN_output shape: [N*num_x*num_y*channel]
        QCNN_output = tf.reshape(QCNN_output, shape=[-1, self.num_x, self.num_y, self.channel, self.depth*self.pixels])
        return tf.math.reduce_sum(QCNN_output, 3)

    @property
    def svg(self):
        circtoplot = self.circuit
        circtoplot.append(self.op)
        return SVGCircuit(circtoplot)

def hybrid_qcnn_model(
    img_size = (57, 31),
    channels = 2
):
    hybrid_qcnn_model = models.Sequential(
        [
            layers.Conv2D(
                filters     = 32, 
                kernel_size = 3, 
                activation  = 'relu',
                strides     = (2,2), 
                input_shape = (img_size[0], img_size[1], channels),
                data_format = 'channels_last'
            ),
            layers.Conv2D(
                filters     = 64, 
                kernel_size = 2, 
                activation  = 'relu',
                strides     = (2,2), 
                input_shape = (img_size[0], img_size[1], channels),
                data_format = 'channels_last'
            ),
            QConv(
                filter_size=2, 
                strides=(2,2), 
                depth=1, 
                activation='relu', 
                name='qconv'
            ),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='linear')
        ]
    )

    hybrid_qcnn_model.compile(
        optimizer='adam',
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    return hybrid_qcnn_model

fig_shape = (train_images.shape[1], train_images.shape[2])
channels = 2

print("Compiling model ...")

hybrid_qcnn = hybrid_qcnn_model(
    img_size = fig_shape,
    channels = channels
)

hybrid_qcnn.summary()

print("Starting training ...")

n_epochs   = 10
batch_size = 64

hybrid_qcnn_history = hybrid_qcnn.fit(
    train_images[:50000],
    train_labels[:50000],
    validation_split = 0.1,
    batch_size       = batch_size,
    epochs           = n_epochs
)

print("Saving trained model ...")

hybrid_qcnn.save('./models/hybrid_qcnn')
