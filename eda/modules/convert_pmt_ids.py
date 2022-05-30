import numpy as np
import pandas as pd 

def convert_pmt_ids(input_ids, conversion_ids):
    cd_ids = np.array(conversion_ids['CdID'])
    pmt_ids = np.array(conversion_ids['PMTID'])
    mask = np.isin(cd_ids, input_ids)
    return pmt_ids[mask]
