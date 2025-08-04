import pandas as pd
import geopandas as gpd
import numpy as np
import pyproj as pp
import pytorch_forecasting as ptf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch as pt

from ucf_atd_model.data import data_loc, ResultCache

cache = ResultCache("vae_sfl")

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

# Verify this number is the same as after the conversion, indicates correct categorification of track_ids by batch
print(Xdata.groupby("batch")["track_id"].value_counts().max())
Xdata["track_id"] = Xdata["track_id"] + (Xdata["batch"] * (Xdata["track_id"].max() + 1))
print(Xdata["track_id"].value_counts().sort_values().max())
Xdata["track_id"] = Xdata["track_id"].astype("category").cat.codes
Xdata = Xdata.drop(["batch"], axis=1)

# validationFilter = Xdata["track_id"] > 182659
Tdata = Xdata #[~validationFilter]

colnames = ["track_id", "time", "x", "y", "speed", "speed_x", "speed_y", "course_x", "course_y", "missing"]

# Create training data
dataset = ptf.TimeSeriesDataSet(
    Tdata, 
    time_idx = "time", 
    target = ["x", "y"], 
    group_ids = "track_id", 
    min_encoder_length=1,
    min_prediction_length=1,
    max_encoder_length = 144, 
    max_prediction_length = 144, 
    time_varying_unknown_reals = ["speed", "speed_x", "speed_y", "course_x", "course_y", "x", "y"], 
    time_varying_unknown_categoricals = ["missing"], 
    constant_fill_strategy = {"missing": "T"},
    allow_missing_timesteps = True
)
train_dataloader = dataset.to_dataloader(train=True, batch_size = 32)

# validation = ptf.TimeSeriesDataSet.from_dataset(dataset, Xdata, stop_randomization=True, predict=True)
# val_dataloader = validation.to_dataloader(train=False)

# Set up training
pl.seed_everything(845)
lr_logger = LearningRateMonitor()
tboard = TensorBoardLogger("train_logs")
trainer = pl.Trainer(max_epochs=5000, enable_model_summary=True, callbacks=[lr_logger], logger=tboard, gradient_clip_val=0.1)
tft = ptf.TemporalFusionTransformer.from_dataset(dataset, learning_rate = 0.01, hidden_size = 10, attention_head_size = 3, dropout = 0.1, loss = ptf.QuantileLoss(), reduce_on_plateau_patience = 200, log_interval = 10)

# Train model
trainer.fit(tft, train_dataloaders=train_dataloader) # val_dataloaders=val_dataloader

# Try multiple ways to save, first is supposedly the preferred way
trainer.save_checkpoint("tft_ptl.out")
pt.save([tft._hparams, tft.state_dict()], 'tft_pt.out')
# If above line is necessary, may load like this
# kwargs, state_dict = torch.load('tft.out')
# model = TemporalFusionTransformer(**kwargs)
# model.load_state_dict(state_dict)
