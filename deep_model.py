import pandas as pd
import geopandas as gpd
import numpy as np
import pyproj as pp
import pytorch_forecasting as ptf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch as pt
import torch.nn as nn
import itertools as it
import gc
import pyarrow as pa
from time import time
import pyarrow.parquet as pq

from ucf_atd_model.data import data_loc, ResultCache

cache = ResultCache("deep")

inp_dim = 21
h_dim = 500
out_dim = 1

device = pt.device("cuda:0")

model = nn.Sequential(
    nn.Linear(inp_dim, h_dim),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(h_dim, h_dim),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(h_dim, h_dim // 2),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(h_dim // 2, h_dim // 10),
    nn.ReLU(),
    nn.Linear(h_dim // 10, 1),
    nn.Sigmoid()
).double().to(device)


table: pd.DataFrame = None
validation: pd.DataFrame = None
with pq.ParquetFile(data_loc("link_data_small.parquet")) as fulldata:
    n_rowgroups = fulldata.num_row_groups
    validation = fulldata.read_row_group(0).to_pandas()
    table = fulldata.read_row_groups(list(range(1, fulldata.num_row_groups))).to_pandas(self_destruct = True)

labels = table.keys()
train_X = pt.from_numpy(table.drop(["label"], axis=1).values)
xmean = pt.mean(train_X, 0)
xstd = pt.std(train_X, 0)
train_X = (train_X - xmean) / xstd
train_y = pt.from_numpy(table["label"].to_numpy()).double()

test_X = pt.from_numpy(validation.drop(["label"], axis=1).values)
test_X = (test_X - xmean) / xstd
test_X = test_X.to(device)
test_y = pt.from_numpy(validation["label"].to_numpy()).double().to(device)

train_dataset = pt.utils.data.TensorDataset(train_X, train_y)

train_dataloader = pt.utils.data.DataLoader(train_dataset, batch_size=8192, num_workers=10, shuffle=True, pin_memory=True)

num_epochs = 100

optimizer = pt.optim.Adam(model.parameters(), lr=0.001)
lossfn = nn.BCELoss()

for epoch in range(num_epochs):
    print(f"Epoch {epoch} / {num_epochs}")
    startTime = time()
    # Train model
    model.train()
    epoch_train_loss = 0
    for input, label in train_dataloader:
        input = input.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        
        output = None
        loss = None
        with pt.set_grad_enabled(True):
            output = model(input)
            loss = lossfn(output.flatten(), label)
            loss.backward()
            optimizer.step()
        
        epoch_train_loss += loss.item()
    
    print(f"    Train Loss: {epoch_train_loss:.3f}")
    
    model.eval()
    # Evaluate model
    with pt.no_grad():
        output = model(input)
        loss = lossfn(output.flatten(), label)
        print(f"    Test Loss: {loss.item():.3f}")
    
    endTime = time()
    print(f"    Time Elapsed: {(endTime - startTime):.3f} seconds")

    pt.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")