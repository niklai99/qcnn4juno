import pandas as pd
import numpy as np

from tqdm import tqdm

def convert_pmt_ids(input_ids, conversion_ids):
    cd_ids = np.array(conversion_ids['CdID'])
    pmt_ids = np.array(conversion_ids['PMTID'])
    indices = np.where(np.in1d(cd_ids, input_ids))[0]
    return pmt_ids[indices]


def load_train_data(
    path:            str,
    training_data:   str,
    training_labels: str,
    pmt_mapping:     str,
    pmt_id_conv:     str
):
    # PMT positions on the sphere
    pmt_pos    = pd.read_csv(path+pmt_mapping)
    pmt_pos.rename({"PMTID":"id"}, axis=1, inplace=True)
    # PMT ID mapping
    pmt_id = pd.read_csv(path+pmt_id_conv)

    # EVENTS
    train_data        = np.load(path+training_data, allow_pickle=True)["a"]
    # TRUE ENERGY AND TRUE VERTEX OF THE EVENTS
    train_label       = pd.read_csv(path+training_labels)

    # COMPUTE RADIUS FROM TRUE VERTEX
    train_label["edepR"] = (train_label["edepX"]**2 + train_label["edepY"]**2 + train_label["edepZ"]**2)**0.5
    
    return train_data, train_label, pmt_pos, pmt_id


def load_train_df(
    path:            str,
    training_data:   str,
    training_labels: str,
    pmt_mapping:     str,
    pmt_id_conv:     str
):
    
    train_data, train_label, pmt_pos, pmt_id = load_train_data(
        path            = path,
        training_data   = training_data,
        training_labels = training_labels,
        pmt_mapping     = pmt_mapping,
        pmt_id_conv     = pmt_id_conv
    )

    NEvents = train_data.shape[1]
    for i in tqdm(range(NEvents)):
        # find non-zero-time hits
        nonzeros_inds = np.where(train_data[2, :][i] != 0.0)[0]
        # PMT ID mapping
        train_data[0, :][i] = convert_pmt_ids(train_data[0, :][i], pmt_id)
        # keep only non-zero hits
        train_data[0, :][i] = train_data[0, :][i][nonzeros_inds]
        train_data[1, :][i] = train_data[1, :][i][nonzeros_inds]
        train_data[2, :][i] = train_data[2, :][i][nonzeros_inds]

    df = pd.DataFrame(train_data.T, columns=("pmt_id", "charge", "hit_time"))
    df["event_id"] = np.arange(0, df.shape[0], 1)
    df = df[["event_id", "pmt_id", "charge", "hit_time"]]  

    event_list = []
    # for each event
    for ev in tqdm(df["event_id"]):

        # number of hits in the event
        ev_shape = df["pmt_id"][ev].shape[0]
        # store features
        pmt      = df["pmt_id"][ev]
        q        = df["charge"][ev]
        ht       = df["hit_time"][ev]
        # replicate the event id
        ev_id    = [ev] * ev_shape

        # build the dataframe
        event_df = pd.DataFrame(np.array([ev_id, pmt, q, ht]).T, columns = ("ev_id", "pmt_id", "charge", "hit_time"))

        # store the event
        event_list.append(event_df)

    # concatenate all events
    data = pd.concat(event_list, ignore_index=True)

    return pd.merge(data, pmt_pos, how="left", left_on="pmt_id", right_on="id").drop("id", axis=1), train_label


def load_pmt_positions(
    path:            str,
    pmt_mapping:     str,
):
    # PMT positions on the sphere
    pmt_pos    = pd.read_csv(path+pmt_mapping)
    pmt_pos.rename({"PMTID":"id"}, axis=1, inplace=True)
    return pmt_pos