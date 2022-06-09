import numpy as np
import pandas as pd
import os
import re
import time
from multiprocessing import Pool

############################################# read files path #############################################
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

print("Reading training files path....")

# train
proj_train_path = "../../juno_data/data/projections_opt"
train_files     = os.listdir(proj_train_path)
train_proj   = [file for file in train_files if "proj" in file]
train_proj.sort(key=natural_keys)

print("-> completed\n")

print("Reading test files path....")

# test
energies = [str(i) for i in range(6, 11)]

proj_test_path = "../../juno_data/data/projections_opt/test"

test_proj_dict   = { e:None for e in energies}

for i, e in enumerate(energies):
    proj_files_path   = proj_test_path + "/e+_" + e
    
    proj_files   = os.listdir(proj_files_path)
    
    test_proj_dict[e]    = [file for file in proj_files if "proj" in file] 
    
    test_proj_dict[e].sort(key=natural_keys)

print("-> completed\n")
    
########################## reduce image resolution with a costum convolution ##########################

def conv_juno(in_matrix, kernel_size, input_shape=(230, 124)):
    """Convolves the input image to reduce size"""
    '''k_s_0 = input_shape[0] - out_shape[0]*stride[0] + 1*stride[0]
    k_s_1 = input_shape[1] - out_shape[1]*stride[1] + 1*stride[1]
    kernel_size = (k_s_0, k_s_1)'''
    stride = kernel_size

    out_shape = np.zeros(2).astype(int)
    out_shape[0] = int((input_shape[0]-kernel_size[0])/stride[0] + 1)
    out_shape[1] = int((input_shape[1]-kernel_size[1])/stride[1] + 1)
    
    out_matrix = np.zeros((out_shape[0], out_shape[1], 2))

    # Loop over the image
    for j in range(0, input_shape[0] - kernel_size[0] + 1, stride[0]):
        for k in range(0, input_shape[1] - kernel_size[1] + 1, stride[1]):
            # sum charge over the kernel
            out_matrix[j  // stride[0], k  // stride[1], 0] = np.sum(in_matrix[j:j+kernel_size[0], k:k+kernel_size[1], 0])
            # charge weighted sum of time over the kernel
            weights = in_matrix[j:j+kernel_size[0], k:k+kernel_size[1], 0]
            elements = in_matrix[j:j+kernel_size[0], k:k+kernel_size[1], 1]
            out_matrix[j  // stride[0], k  // stride[1], 1] = np.sum(np.multiply(elements, weights))/np.sum(weights) \
                                                              if np.sum(weights) != 0 else 0
    return out_matrix


# train files
n_train_files = 50

print('Reducing images..\n')

read_time = []
conv_time = []

#for j, file in enumerate(train_proj[:n_train_files]):
def initializer():
    def conv_juno(in_matrix, kernel_size, input_shape=(230, 124)):
        """Convolves the input image to reduce size"""
        '''k_s_0 = input_shape[0] - out_shape[0]*stride[0] + 1*stride[0]
        k_s_1 = input_shape[1] - out_shape[1]*stride[1] + 1*stride[1]
        kernel_size = (k_s_0, k_s_1)'''
        stride = kernel_size

        out_shape = np.zeros(2).astype(int)
        out_shape[0] = int((input_shape[0]-kernel_size[0])/stride[0] + 1)
        out_shape[1] = int((input_shape[1]-kernel_size[1])/stride[1] + 1)
        
        out_matrix = np.zeros((out_shape[0], out_shape[1], 2))

        # Loop over the image
        for j in range(0, input_shape[0] - kernel_size[0] + 1, stride[0]):
            for k in range(0, input_shape[1] - kernel_size[1] + 1, stride[1]):
                # sum charge over the kernel
                out_matrix[j  // stride[0], k  // stride[1], 0] = np.sum(in_matrix[j:j+kernel_size[0], k:k+kernel_size[1], 0])
                # charge weighted sum of time over the kernel
                weights = in_matrix[j:j+kernel_size[0], k:k+kernel_size[1], 0]
                elements = in_matrix[j:j+kernel_size[0], k:k+kernel_size[1], 1]
                out_matrix[j  // stride[0], k  // stride[1], 1] = np.sum(np.multiply(elements, weights))/np.sum(weights) \
                                                                if np.sum(weights) != 0 else 0
        return out_matrix

    proj_train_path = "../../juno_data/data/projections_opt"
    proj_test_path = "../../juno_data/data/projections_opt/test"
    out_path = "../../juno_data/data/projections_opt"

def f(file):
    #start_time = time.time()
    # load file
    f = np.load(proj_train_path + '/' + file, mmap_mode='r')
    #read_time.append(time.time() - start_time)
    # set kernel size
    kernel_size = (4, 4)
    # reduce images
    #start_time = time.time()
    f_r_shape = (int(f.shape[1]/kernel_size[0]), int(f.shape[2]/kernel_size[1]))
    f_r = np.zeros((f.shape[0], f_r_shape[0], f_r_shape[1], f.shape[3]))
    for i in range(f.shape[0]):
        f_r[i] = conv_juno(f[i], kernel_size)
    #conv_time.append(time.time() - start_time)
    # save reduced file
    np.save(proj_train_path + '/reduced/train/red_'+ file, f_r)
    #print(f'{(j+1)/n_train_files*100:.2f}% done \t avg reading time: {round(np.mean(read_time),2)}s, avg conv time: {round(np.mean(conv_time),2)}s', '\r', end='')

#print('Reducing files:', train_proj[:n_train_files], sep='\n')
#with Pool(processes=16, initializer=initializer) as pool:
#    result = pool.map(f, train_proj[:n_train_files]) 


# test files

out_path = "../../juno_data/data/projections_opt"
def f_test(file):
    #start_time = time.time()
    # load file
    f = np.load(proj_test_path + f"/e+_{e}/" + file, mmap_mode='r')
    #read_time.append(time.time() - start_time)
    # set kernel size
    kernel_size = (4, 4)
    # reduce images
    #start_time = time.time()
    f_r_shape = (int(f.shape[1]/kernel_size[0]), int(f.shape[2]/kernel_size[1]))
    f_r = np.zeros((f.shape[0], f_r_shape[0], f_r_shape[1], f.shape[3]))
    for i in range(f.shape[0]):
        f_r[i] = conv_juno(f[i], kernel_size)
    #conv_time.append(time.time() - start_time)
    # save reduced file
    np.save(out_path + f'/reduced/test/e+_{e}/red_'+ file, f_r)
    #print(f'{(j+1)/n_train_files*100:.2f}% done \t avg reading time: {round(np.mean(read_time),2)}s, avg conv time: {round(np.mean(conv_time),2)}s', '\r', end='')


for e in energies:
    n_train_files = 50
    
    test_proj = test_proj_dict[e][:n_train_files]

    print(f"Reducing energy: {e} MeV images\n")
    print('Reducing files:', test_proj, sep='\n')
    with Pool(processes=8, initializer=initializer) as pool:
        result = pool.map(f_test, test_proj) 

            
print('JOB FINISHED!')

