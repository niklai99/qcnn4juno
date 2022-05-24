import numpy as np
import pandas as pd

def plane_mapping(data: pd.DataFrame):
    data["z"]   = data["z"].astype("int32")
    data["col"] = data.groupby("z").ngroup()
    
    tmp = data[["id", "col"]]
    df = (
        tmp
            .groupby("col", as_index=False)
            .apply(lambda x: x.reset_index(drop = True))
            .reset_index()
    )
    df.drop(["level_0", "id"], axis=1, inplace=True)
    df.rename({"level_1":"row"}, axis=1, inplace=True)
    df["id"] = df.index
    
    return df


def plane_projection(df: pd.DataFrame, mapping: pd.DataFrame, ev_id: np.int16):
    mask = (df["ev_id"] == ev_id)
    return pd.merge(
        mapping, 
        df[mask], 
        how      = "left", 
        left_on  = "id", 
        right_on = "pmt_id"
    ).drop("pmt_id", axis = 1)