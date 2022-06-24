import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow import keras

import cirq
import sympy
import numpy as np
import collections
import datetime
import pickle
import qsimcirq

# visualization tools
import matplotlib as mpl
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit


# set up gpu
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow_quantum as tfq


#*********LOAD DATASET********#
n_train = 20000    # Size of the train dataset
n_test  = 2000    # Size of the test dataset
ansatz = 1

mnist_dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# Reduce dataset size
train_images = train_images[:n_train]
train_labels = train_labels[:n_train]
test_images = test_images[:n_test]
test_labels = test_labels[:n_test]

# Normalize pixel values within 0 and 1
if ansatz == 1:
    train_images = train_images / 255
    test_images = test_images / 255
elif ansatz==2:
    train_images = train_images / 128 - 1
    test_images = test_images / 128 - 1
elif ansatz==3:
    print('non devo fare nulla miao\nChing Chong è un boss')

# Add extra dimension for convolution channels
train_images = np.array(train_images[..., tf.newaxis])
test_images = np.array(test_images[..., tf.newaxis])

# reduce image resolution
train_images = tf.image.resize(train_images[:], (14,14)).numpy()
test_images = tf.image.resize(test_images[:], (14,14)).numpy()


#*********DEFINE QCONV LAYER********#
debug=False
options = qsimcirq.QSimOptions(use_gpu=True, max_fused_gate_size=4, gpu_mode=1)
s = qsimcirq.QSimSimulator(options)
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
        self.op = [cirq.Z(cirq_qubits[(i+1)]) for i in range(0, len(cirq_qubits), self.pixels)]                      # measure
        
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
        if debug:
            print('stack_set:', stack_set.shape) 
        # -> shape: [N, num_x*num_y, filter_size, filter_size, channel]
        stack_set = tf.transpose(stack_set, perm=[0, 1, 4, 2, 3])
        # -> shape: [N, num_x*num_y, channel, filter_size, fiter_size]
        stack_set = tf.reshape(stack_set, shape=[-1, self.filter_size[0]*self.filter_size[1]])
        # -> shape: [N*num_x*num_y*channel, filter_size^2]
        if debug:
            print(stack_set.shape)
        # total input circuits: N * num_x * num_y * channel
        circuit_inputs = tf.tile([self.circuit_tensor], [tf.shape(inputs)[0], 1])
        circuit_inputs = tf.reshape(circuit_inputs, shape=[-1])
        #tf.fill([tf.shape(inputs)[0]*self.num_x*self.num_y, 1], 1)
        #outputs = []
        
        controller = tf.tile(self.kernel, [tf.shape(inputs)[0]*self.num_x*self.num_y, 1])
        outputs=self.single_depth_QCNN(stack_set, controller, circuit_inputs)
        # shape: [N, num_x, num_y, self.depth] 
            
        output_tensor = outputs #tf.stack(outputs, axis=3)
        output_tensor = tf.math.acos(tf.clip_by_value(output_tensor, -1+1e-5, 1-1e-5)) / np.pi
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
        QCNN_output = tfq.layers.Expectation(backend=s)(circuit_inputs, 
                                               symbol_names=self.params,
                                               symbol_values=input_data,
                                               operators=self.op)
        # QCNN_output shape: [N*num_x*num_y*channel]
        QCNN_output = tf.reshape(QCNN_output, shape=[-1, self.num_x, self.num_y, self.channel, self.depth])
        return tf.math.reduce_sum(QCNN_output, 3)

    @property
    def svg(self):
        circtoplot = self.circuit
        circtoplot.append(self.op)
        return SVGCircuit(circtoplot)




#******DEFINE MODEL********#
width = np.shape(train_images)[1]
height = np.shape(train_images)[2]

qcnn_model = models.Sequential()


qcnn_model.add(QConv(filter_size=2, strides=(2,2), depth=1, activation='relu', 
                     name='qconv1', input_shape=(width, height, 1)))
qcnn_model.add(QConv(filter_size=2, strides=(1,1), depth=2, activation='relu', 
                     name='qconv2', input_shape=(width, height, 1)))
qcnn_model.add(layers.Flatten())
qcnn_model.add(layers.Dense(32, activation='relu'))
qcnn_model.add(layers.Dense(10, activation='softmax'))

log_dir = "qcnn_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#*******TRAIN*********#
qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

qcnn_history = qcnn_model.fit(train_images, train_labels,
                        validation_split = 0.3, 
                        epochs=20, batch_size=64,
                        callbacks=[tensorboard_callback])


#******save model weights*****#
qcnn_model.save_weights('qcnn_weights_14x14_2x2_1_2x2_2_64.h5')
with open('qcnn_history_14x14_2x2_1_2x2_2_64.pkl', 'wb') as f:
    pickle.dump(qcnn_history, f)
    pickle.dump(qcnn_model, f)

