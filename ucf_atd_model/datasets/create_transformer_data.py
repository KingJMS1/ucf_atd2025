import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import pyarrow.parquet as pq

from datetime import datetime
import time
import pickle as pkl
from typing import *
import os

from ..data import data_loc, new_data_loc
from .. import transformer_consts as const

import torch as pt

def initialize_preprocessors():
    df = get_data(0)
    _, scaler, discretizer = create_transformer_data(df)
    with open(new_data_loc("trans_preprocessors.pkl"), 'wb') as file:
        pkl.dump((scaler, discretizer), file)


def get_data(i):
    # Read in historical data
    with pq.ParquetFile(data_loc("historical.parquet")) as fulldata:
        rowgroup = fulldata.read_row_group(i)
        data: pd.DataFrame = rowgroup.to_pandas()
        data["time"] = data["time"].apply(lambda x: datetime.combine(datetime.fromtimestamp(0).date(), x))
    
    data["track_id_true"] = data["track_id"]
    return data


def create_transformer_data(df, scaler: StandardScaler = None , discretizer: KBinsDiscretizer = None) -> Tuple[pt.Tensor, pt.Tensor, StandardScaler, KBinsDiscretizer]:
    # Preprocess into a nice df
    ais_data = df.copy()
    ais_data = ais_data.sort_values(['track_id', 'time']).reset_index(drop=True)
    
    ais_data['course_rad'] = np.deg2rad(ais_data['course'])
    ais_data['course_x'] = np.sin(ais_data['course_rad'])
    ais_data['course_y'] = np.cos(ais_data['course_rad'])
    
    features_to_scale = ['lat', 'lon', 'speed', 'course_x', 'course_y']
    if scaler is None:
        scaler = StandardScaler()
        ais_data[features_to_scale] = scaler.fit_transform(ais_data[features_to_scale])
    else:
        ais_data[features_to_scale] = scaler.transform(ais_data[features_to_scale])
    
    # Discretize features
    bin_cols = [f'{feat}_bin' for feat in features_to_scale]
    if discretizer is None:
        discretizer = KBinsDiscretizer(n_bins=const.n_bins, encode='ordinal', strategy='uniform')
        ais_data[bin_cols] = discretizer.fit_transform(ais_data[features_to_scale])
    else:
        ais_data[bin_cols] = discretizer.transform(ais_data[features_to_scale])

    ais_data = ais_data.drop(columns=['course', 'course_rad'])

    # Break out out X, y
    features_for_trans = ['lat_bin', 'lon_bin', 'speed_bin', 'course_x_bin', 'course_y_bin']
    all_sequences = []
    target_sequences = []
    for track_id, group in ais_data.groupby('track_id'):
        data = group[features_for_trans].to_numpy()
        target_data = group[features_for_trans].to_numpy()
        
        seq_length = const.seq_length
        predict_steps = const.predict_steps
        
        if len(data) >= seq_length:
            for i in range(len(data) - seq_length + 1):
                # Create X
                all_sequences.append(data[i:i + seq_length])
                
                # Create y
                if i + seq_length + predict_steps <= len(data):
                    target_sequences.append(target_data[i + seq_length:i + seq_length + predict_steps])
                else:
                    pad_length = i + seq_length + predict_steps - len(data)
                    target = target_data[i + seq_length:] if i + seq_length < len(data) else target_data[-predict_steps:]
                    if pad_length > 0:
                        padding = np.repeat(target_data[-1:], pad_length, axis=0)
                        target = np.concatenate([target, padding], axis=0)[:predict_steps]
                    target_sequences.append(target)
        else:
            # Create X
            padded = np.zeros((seq_length, len(features_for_trans)))
            padded[:len(data)] = data
            padded[len(data):] = data[-1] if len(data) > 0 else 0
            all_sequences.append(padded)
            
            # Create y
            target = np.zeros((predict_steps, len(features_for_trans)))
            target[:min(len(data), predict_steps)] = target_data[-min(len(data), predict_steps):]
            if len(data) < predict_steps:
                target[len(data):] = target_data[-1] if len(data) > 0 else 0
            
            target_sequences.append(target)

    # Convert into tensors
    all_sequences = np.clip(all_sequences, 0, const.n_bins - 1)
    target_sequences = np.clip(target_sequences, 0, const.n_bins - 1)
    sequences_tensor = pt.tensor(all_sequences)
    target_tensor = pt.tensor(target_sequences)

    return sequences_tensor, target_tensor, scaler, discretizer

def create_transformer_data_full(i) -> Tuple[pt.tensor, pt.tensor, StandardScaler, KBinsDiscretizer]:
    preprocessor_file = new_data_loc("trans_preprocessors.pkl")
    if not os.path.isfile(preprocessor_file):
        initialize_preprocessors()

    scaler = discretizer = None
    with open(preprocessor_file, 'rb') as file:
        scaler, discretizer = pkl.load(file)
    
    X, y = create_transformer_data(get_data(i), sclaer=scaler, discretizer=discretizer)
    return X, y, scaler, discretizer

# def run(i):
#     print(f"Processing dataset {i}", flush=True)
#     start = time.time()
#     try:
#         oracle, xdata, ydata = create_data(get_data(i))
#         data_path = data_loc("class20")
#         xdata = pd.DataFrame(np.array(xdata), columns=colnames)
#         xdata.to_csv(f"{data_path}/xdata_{i}.csv")
#         np.save(f"{data_path}/ydata_{i}.npy", np.array(ydata))
#     except Exception:
#         traceback.print_exc()
#         return traceback.format_exc()
#     end = time.time()
#     print(f"Finished dataset {i}: Took {end - start:.2f} seconds")
#     return ""
