import tqdm
import atd2025
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os

from datetime import *
from typing import *
import itertools

from .data import data_loc, ResultCache, new_data_loc
from .datasets.create_link_data import calculate_link_features, project_forward, get_bearing, haversine_distance_m

cache = ResultCache("lgbm")
batch_size = 50
num_batches = None

feature_cols = ['delta_time', 'distance_m', 'implied_speed_knots', 'delta_speed', 'delta_course', 'bearing_diff', 'kinematic_error']
target_col = 'label'

def data_batches() -> Generator[pd.DataFrame, None, None]:
    global num_batches
    with pq.ParquetFile(data_loc("link_data_small.parquet")) as fulldata:
        n_rowgroups = fulldata.num_row_groups
        batches = list(itertools.batched(range(n_rowgroups), batch_size))
        num_batches = len(batches)
        for batch in batches:
            # Read this file in
            rows = fulldata.read_row_groups(batch)
            data: pd.DataFrame = rows.to_pandas()
            yield data
        
def train_eval():
    gen = data_batches()
    data = next(gen)
    
    X = data[feature_cols]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data = False)
    params = {"objective": "binary", "metric": "binary_logloss", "num_leaves": 70, "num_iterations": 25, "force_row_wise": True, "min_data_in_leaf": 80}
    bst = lgb.train(params, train_data, keep_training_booster=True)
    print(bst.eval(test_data, "test set 0"))
    i = 0

    for data in gen:
        i += 1
        X = data[feature_cols]
        y = data[target_col]

        train_data = lgb.Dataset(X, label=y)
        bst = lgb.train(params, train_data, init_model=bst, keep_training_booster=True, valid_sets=test_data, valid_names=["test set 0"])
        print(f"Iteration {i} / {num_batches - 1}")
        print(bst.eval_valid())

    bst.save_model(new_data_loc("lgbm_trained_model.lgbm"))


# Helper functions for creating the validation dataset
def subset(lastpts, i):
    columns = ["time", "lat", "lon", "speed", "course"]
    return {key: lastpts[key][:i] for key in columns}

def setidx(lastpts, i, val):
    lastpts["time"][i] = val["time"].to_numpy()
    lastpts["lat"][i] = val["lat"]
    lastpts["lon"][i] = val["lon"]
    lastpts["speed"][i] = val["speed"]
    lastpts["course"][i] = val["course"]


def run(file: str) -> Tuple[pd.DataFrame, str]:
    in_cache, path = cache.test_cache(file)
    if in_cache:
        return pd.read_csv(path), path

    if not os.path.exists(data_loc("lgbm_trained_model.lgbm")):
        print("Please run the train_eval function in lgbm via `python -m ucf_atd_model.lgbm` before attempting to make predictions")
        exit(1)

    ml_model = lgb.Booster(model_file=data_loc("lgbm_trained_model.lgbm"))

    data = pd.read_csv(data_loc(file))
    data["time"] = pd.to_datetime(data["time"])

    df = data.sort_values('time').reset_index(drop=True)
    df['track_id'] = -1

    next_track_id = 0
    
    n = df.shape[0]
    lastPtInTrack = {
        "time": np.repeat(pd.Timestamp(year=1970, month=1, day=1, hour=1, minute=1, second=1).to_numpy(), n), 
        "lat": np.repeat(-1.0, n), 
        "lon": np.repeat(-1.0, n), 
        "speed": np.repeat(-1.0, n), 
        "course": np.repeat(-1.0, n)
    }

    for i in tqdm.tqdm(range(len(df))):
        p_current = df.iloc[i]

        if next_track_id == 0:
            df.loc[i, 'track_id'] = next_track_id
            setidx(lastPtInTrack, i, p_current)
            next_track_id += 1
            continue

        best_match_track_id, best_score = -1, -np.inf

        active_tracks_df = subset(lastPtInTrack, next_track_id)
        
        best_score = 0
        best_match_track_id = -1
        time_diff = (p_current["time"].to_numpy() - active_tracks_df["time"]).astype("timedelta64[s]").astype("int")
        timeCorrect: np.ndarray = (0 < time_diff) & (time_diff < 3600 * 4)
        idxs = np.arange(len(timeCorrect))

        kinematic_errors = haversine_distance_m(p_current["lat"], p_current["lon"], *project_forward(active_tracks_df['lat'], active_tracks_df['lon'], active_tracks_df['speed'], active_tracks_df['course'], time_diff))
        kinematic_scores = np.exp(-kinematic_errors / 1000)
        
        # print(active_tracks_df)
        link_features = calculate_link_features(active_tracks_df, p_current)
        if not link_features.empty:
            features_df = link_features[feature_cols]
            ml_scores = ml_model.predict(features_df)
            final_scores: pd.Series = 0.3 * kinematic_scores[timeCorrect] + 0.7 * ml_scores
            
            if final_scores.max() > 0.5:
                best_score = final_scores.max()
                best_match_track_id = idxs[timeCorrect][final_scores.argmax()]

        # Assignment with a confidence threshold
        if best_score > 0.5:
            df.loc[i, 'track_id'] = best_match_track_id
            setidx(lastPtInTrack, best_match_track_id, p_current)
        else:
            df.loc[i, 'track_id'] = next_track_id
            setidx(lastPtInTrack, next_track_id, p_current)
            next_track_id += 1

    output = pd.DataFrame({"point_id": df["point_id"], "track_id": df["track_id"]})
    output.to_csv(path, index=False)

    return output, path


if __name__ == "__main__":
    train_eval()