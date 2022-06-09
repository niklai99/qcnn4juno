import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow import keras

import numpy as np
import collections
import pandas as pd

import matplotlib.pyplot as plt

import pickle
import glob
import re
import os

from tqdm import tqdm
from scipy.optimize import curve_fit

#*****INITIALIZE GPU****#
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate, epochs, steps_per_epoch):
    self.initial_learning_rate = initial_learning_rate
    self.epochs = epochs
    self.steps_per_epoch = steps_per_epoch
    self.m = initial_learning_rate / steps_per_epoch
    self.decay_rate = tf.constant((10**-8 / initial_learning_rate)**(((epochs - 1)*steps_per_epoch)**-1), dtype=tf.float32)
    print('decay_rate:', self.decay_rate)

  def __call__(self, step):
    result = tf.cond(tf.less(step, self.steps_per_epoch), 
                   lambda: self.m * (step+1),
                   lambda: self.initial_learning_rate * self.decay_rate**tf.cast(step+1-self.steps_per_epoch, dtype=tf.float32))

    #tf.print('lr at step', step, 'is', result, output_stream='file://learning_rates.txt')
    return result  

  def get_config(self):
      return {
          "initial_learning_rate": self.initial_learning_rate,
          "epochs": self.epochs,
          "steps_per_epoch": self.steps_per_epoch
      }


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def gaus(x, a, mu, sigma):
    return a*np.exp( -(x-mu)**2 / (2*sigma**2) )


def get_data_from_filename(filename):
    # Read the corresponding label file

    npdata = np.load(filename, mmap_mode='r')

    return (npdata)

def get_data_wrapper(filename):
    # Assuming here that both your data and label is double type
    features = tf.numpy_function(
        get_data_from_filename, [filename], tf.float64) 
    return tf.data.Dataset.from_tensor_slices(features)


N_FILES = None


energy_list = [0, 0.1, 0.3, 0.6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
resolution_list = []
bias_list = []
err_resolution_list = []
err_bias_list = []
print("Loading ResNet...")
resnet_model = tf.keras.models.load_model("./models/20220607-093041/", custom_objects={'MyLRSchedule': MyLRSchedule})


for e in energy_list:
    
    print(f"\n\nUsing E = {e} MeV\n\n")

    data_path    = f"../../juno_data/data/projections_opt/test/e+_{e}/"
    target_path  = f"../../juno_data/data/real/test/targets/e+_{e}/"
    data_files   = os.listdir(data_path) 
    target_files = os.listdir(target_path) 

    test_proj    = [file for file in data_files    if "proj" in file]
    test_target  = [file for file in target_files  if "targets" in file]

    test_proj.sort(key=natural_keys)   
    test_target.sort(key=natural_keys) 
    
    test_proj   = test_proj[:N_FILES]
    test_target = test_target[:N_FILES]
    
    print("Reading data...")
    # Create dataset of filenames.
    ds = tf.data.Dataset.from_tensor_slices([data_path + file for file in tqdm(test_proj)])
    # Retrieve .npy files
    ds = ds.flat_map(get_data_wrapper)
    # Optimizations
    ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
    ds = ds.batch(64, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    print("Reading labels...")
    test_labels = np.concatenate([pd.read_csv(target_path + file)["edep"].to_numpy() for file in tqdm(test_target)])
         
    print("\nStarting prediction:")
    edep_pred = resnet_model.predict(ds, verbose=1)
    print(f"Predicted E = {e} MeV\n")
    edep_pred = edep_pred.reshape(edep_pred.shape[0],)
    
    res = (edep_pred-test_labels)/test_labels
    res = res[np.abs(res - res.mean()) < 3*res.std()]
    


    hist, edges = np.histogram(res, bins=100)
    bincenters = (edges[1:] + edges[:-1]) / 2

    xgrid = np.linspace(edges[0], edges[-1], 500)

    n     = res.shape[0]
    mean  = res.mean()
    sigma = res.std()
    popt, pcov = curve_fit(gaus, bincenters, hist, p0=[n, mean, sigma])

    n_events   = edep_pred.shape[0]
    resolution = popt[2]
    bias       = popt[1]
    
    err_resolution = pcov[2][2]**0.5
    err_bias       = pcov[1][1]**0.5
    
    resolution_list.append(resolution)
    bias_list.append(bias)
    err_resolution_list.append(err_resolution)
    err_bias_list.append(err_bias)

    fig = plt.figure(figsize=(12,7), constrained_layout=True)
    ax  = fig.add_subplot(111)

    ax.hist(bincenters, weights=hist, bins=100)
    ax.plot(xgrid, gaus(xgrid, *popt), lw=3)
    ax.text(0.65, 0.75, f"number of events = {n_events}", fontsize=18,transform=ax.transAxes)
    ax.text(0.65, 0.7, f"resolution = {resolution*100:.2f}%", fontsize=18,transform=ax.transAxes)
    ax.text(0.65, 0.65, f"bias = {abs(bias)*100:.2f}%", fontsize=18,transform=ax.transAxes)

    ax.set_title(f"E$_e$ = {e} MeV", fontsize=18)
    ax.set_xlabel("(E$_{pred}$ - E$_{true}$) / E$_{true}$", fontsize=16)
    ax.set_ylabel("counts", fontsize=16)

    ax.tick_params(axis="both", which="major", labelsize=14, length=5)
    
    print("Saving plot...")
    fig.savefig(f"./plots/e_{e}mev_hist_resnet1002.png", dpi=300, facecolor="white")
    

print("\n\n\nALL ENERGIES PREDICTED!!")
results = np.concatenate((resolution_list, err_resolution_list, bias_list, err_bias_list), axis = 1)
print("\n\nSAVING RESULTS!!")
np.savetxt("./results/results_resnet1002.txt", results, header="res err_res bias err_bias")