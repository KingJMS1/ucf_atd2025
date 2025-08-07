import pandas as pd
import geopandas as gpd
import numpy as np
import pyproj as pp
import pytorch_forecasting as ptf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch as pt
import itertools as it
import gc

from ucf_atd_model.data import data_loc, ResultCache

cache = ResultCache("tft_tampa")

# Important constatnts
wgs84 = pp.CRS.from_epsg(4326)
utm_fl = pp.CRS.from_epsg(32617)
ktmps = 0.514444 # Convert knots to meters per second

# Convert to a trainable form
data = pd.read_csv(data_loc("tampa_discrete_time.csv"))
data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data["lon"], data["lat"]), crs=wgs84).to_crs(utm_fl)
data["course"] = data["course"] * (np.pi / 180)
data["speed"] = data["speed"] * ktmps
data["speed_x"] = np.sin(data["course"]) * data["speed"]
data["speed_y"] = np.cos(data["course"]) * data["speed"]
data["course_x"] = np.sin(data["course"])
data["course_y"] = np.cos(data["course"])
data["x"] = data.geometry.x
data["y"] = data.geometry.y

# Insert an indicator to tell model if we are forward filling for missing time steps
cols = ["track_id", "time", "x", "y", "speed", "speed_x", "speed_y", "course_x", "course_y", "batch"]
Xdata = data[cols].copy()
missdata = ["F"] * Xdata.shape[0]
missdata[0] = "T"
Xdata["missing"] = missdata
Xdata["missing"] = Xdata["missing"].astype("category")

# Make track ids unique by batch
Xdata["track_id"] = Xdata["track_id"] + (Xdata["batch"] * (Xdata["track_id"].max() + 1))
validationFilter = Xdata["batch"] > 2475
Xdata["track_id"] = pd.Series(pd.Categorical(Xdata["track_id"], categories=Xdata["track_id"].unique(), ordered=True).codes)
Vdata = Xdata[validationFilter].reset_index(drop=True)
Vdata["track_id"] = Vdata["track_id"] - Vdata["track_id"].min()
Xdata = Xdata[~validationFilter]

first_batch = Xdata[Xdata["batch"] < 15]
first_batch = first_batch.drop(["batch"], axis=1).reset_index(drop=True)

print(Vdata)
print(first_batch)

curr_batch_id = 15
batch_size = 10
last_batch_id = Xdata["batch"].max() + 1

colnames = ["track_id", "time", "x", "y", "speed", "speed_x", "speed_y", "course_x", "course_y", "missing"]

# Create training data
dataset = ptf.TimeSeriesDataSet(
    first_batch, 
    time_idx = "time", 
    target = ["x", "y"], 
    group_ids = "track_id", 
    min_encoder_length=1,
    min_prediction_length=1,
    max_encoder_length = 144, 
    max_prediction_length = 144,
    min_prediction_idx=0,
    time_varying_unknown_reals = ["speed", "speed_x", "speed_y", "course_x", "course_y", "x", "y"], 
    time_varying_unknown_categoricals = ["missing"], 
    constant_fill_strategy = {"missing": "T"},
    allow_missing_timesteps = True
)
train_dataloader = dataset.to_dataloader(train=True, batch_size = 32)

validation = ptf.TimeSeriesDataSet.from_dataset(dataset, Vdata, stop_randomization=True)
val_dataloader = validation.to_dataloader(train=False)

# Set up training
pl.seed_everything(845)
lr_logger = LearningRateMonitor()
tboard = TensorBoardLogger("train_logs")
epochs_per_batch = 2
trainer_args = {"max_epochs": epochs_per_batch, "enable_model_summary": True, "callbacks": [lr_logger], "logger": tboard, "gradient_clip_val": 0.1}
trainer = pl.Trainer(**trainer_args)
tft = ptf.TemporalFusionTransformer.from_dataset(dataset, learning_rate = 0.001, lstm_layers=2, dropout = 0.1, loss = ptf.QuantileLoss(), log_interval = 2)

# Train model
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.save_checkpoint("checkpoints/tft_pt1.out")

tot_epochs = 10 * ((last_batch_id // batch_size) + 1)
# Make 10 passes over the dataset
for i in range(tot_epochs):
    print(f"Epoch {i} of {tot_epochs}")
    if curr_batch_id >= last_batch_id:
        curr_batch_id = 0
    batch = Xdata[(curr_batch_id <= Xdata["batch"]) & (Xdata["batch"] <= curr_batch_id + batch_size)].drop(["batch"], axis=1).reset_index(drop=True)
    batch["track_id"] = batch["track_id"] - batch["track_id"].min()
    train_batch = ptf.TimeSeriesDataSet.from_dataset(dataset, batch, stop_randomization=True)
    train_dataloader = train_batch.to_dataloader(train=True, batch_size=32)

    tft = ptf.TemporalFusionTransformer.load_from_checkpoint(f"checkpoints/tft_pt{i+1}.out")
    trainer_args["max_epochs"] += epochs_per_batch
    trainer = pl.Trainer(**trainer_args)
    
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=f"checkpoints/tft_pt{i+1}.out")
    trainer.save_checkpoint(f"checkpoints/tft_pt{i+2}.out")
    curr_batch_id += batch_size
    gc.collect()