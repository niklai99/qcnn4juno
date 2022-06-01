import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
from tqdm import tqdm


import modules
import modules.convert_pmt_ids

energies = [0.1, 0.3, 0.6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# PMT positions on the sphere
PMTPos_CD_LPMT    = pd.read_csv("../data/PMTPos_CD_LPMT.csv")
PMTPos_CD_LPMT.rename({"PMTID":"id"}, axis=1, inplace=True)
# PMT ID mapping
PMT_ID_conversion = pd.read_csv("../data/PMT_ID_conversion.csv")

for e in energies:
    
    datapath  = f'../../juno_data/data/real/test/data/e+_{e}/'
    outputpath = f'../../juno_data/data/projections/test/e+_{e}/'
    
    os.mkdir(outputpath)
    
    print(f'Reading test set for E =  {e} MeV...')

    for i in range(0, 100):

        #**********READING DATA*************#
        print(f'Reading datafile number {i}...')
        filename  = f'raw_data_test_{i}.npz'
        if i == 381:
            print('Skipping corrupted file' + filename + '\n-----------------') 
            continue
        datafile  = datapath + filename
        # EVENTS
        train_data        = np.load(datafile, allow_pickle=True)["a"]

        NEvents = train_data.shape[1]
        for i in tqdm(range(NEvents)):
            # find non-zero-time hits
            nonzeros_inds = np.where(train_data[2, i] != 0.0)[0]
            # PMT ID mapping
            train_data[0, i] = modules.convert_pmt_ids.convert_pmt_ids(train_data[0, i], PMT_ID_conversion)
            # keep only non-zero hits
            train_data[0, i] = train_data[0, i][nonzeros_inds]
            train_data[1, i] = train_data[1, i][nonzeros_inds]
            train_data[2, i] = train_data[2, i][nonzeros_inds]


        #********MAPPING MATRIX***********#
        print('Mapping...')
        pmt_pos      = PMTPos_CD_LPMT.copy()
        #pmt_pos["z"] = pmt_pos["z"].astype("int32")
        z_levels     = pmt_pos["z"].round().unique()       #np.linspace(pmt_pos['z'].min(), pmt_pos['z'].max(), 124) 

        mat    = np.empty(shape=(230, 124))
        mat[:] = np.nan

        N_max = 115

        for i, z in enumerate(z_levels):

            mask = (pmt_pos.z.round() == z)         #(np.abs(pmt_pos.z - z) < delta)
            masked = pmt_pos[mask]

            R = masked.R.mean()

            Neff = N_max * np.sqrt(R**2 - z**2) / R

            for index, row in masked.iterrows():
                ix = round( Neff * (np.arctan2(row.x , row.y) / np.pi) + (N_max / 2) ) + 57

                if ix >= 230:
                    ix = ix - 230

                if np.isnan(mat[ix, i+1]):
                    mat[ix, i+1] = row.id
                else:
                    mat[ix, 123 if i else i] = row.id


        m_dict = {key: None for key in range(17612)}

        for j, i in itertools.product(range(mat.shape[1]), range(mat.shape[0])):
            if np.isnan(mat[i,j]): continue
            m_dict[mat[i,j]] = (i, j)


        #*****DATA GENERATION****#
        print('Inserting data into mapped images...')
        la_cazzo_di_matrice = np.empty(shape=(NEvents, 230, 124, 2))
        la_cazzo_di_matrice[:] = np.nan

        # qui inizia il bello
        for ev in range(NEvents):  
            for i, pmt_active in enumerate(train_data[0, ev]):
                row, col = m_dict[pmt_active]
                la_cazzo_di_matrice[ev,row,col,0] = train_data[1,ev][i]
                la_cazzo_di_matrice[ev,row,col,1] = train_data[2,ev][i]

            print(f'{(ev+1)/NEvents*100:.2f}% done', '\r', end='')

        print('\nSaving mapped images\n')
        outputfile = outputpath + 'proj_' + filename

        np.savez(outputfile, la_cazzo_di_matrice)
        print('------------------------------')

print('JOB FINISHED!')