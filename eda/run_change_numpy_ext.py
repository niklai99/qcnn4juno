import numpy as np
import glob
from tqdm import tqdm
import os

filelist_test = glob.glob('../../juno_data/data/projections/test/*/*.npz')
print(f'converting {len(filelist_test)} files...')
listf = filelist_test
for i in tqdm(range(len(listf))):
    data = np.load(listf[i], allow_pickle=True)['arr_0']
    data[np.isnan(data)] = 0.0
    filename = listf[i].split('.npz')[0] + '.npy'
    split = filename.split('projections')
    np.save(split[0] + 'projections_opt' + split[1], data)
    del data
    os.remove(listf[i])

