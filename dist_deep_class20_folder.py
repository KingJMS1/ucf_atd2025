import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import tqdm
from pathlib import Path

from time import time
import itertools as it
import gc
import os
from sys import stderr, stdout

from ucf_atd_model.data import data_loc, new_data_loc

import torch as pt
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from ucf_atd_model.c20_consts import *


badnames = [x for x in full_names if x.endswith("_16")]
ynames = [x for x in ynames if not x.endswith("_16")]

data_files = [os.path.join(data_loc("c20_data"), x) for x in os.listdir(data_loc("c20_data"))]
validation_file = data_files[0]
validation = [0, 1, 2, 3]


# Setup the model
inp_dim = len(colnames) - len(badnames) + 1
h_dim = 3000
out_dim = n_norm_classes + 1 - 1

device = pt.device("cuda:0")

model = nn.Sequential(
    nn.Linear(inp_dim, h_dim),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(h_dim, h_dim),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(h_dim, h_dim // 2),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(h_dim // 2, h_dim // 2),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(h_dim // 2, out_dim),
).to(device)


# Read the validation data in
X_test = None
y_test = None
xmean = None
xstd = None

with pq.ParquetFile(validation_file) as fulldata:
    testData = fulldata.read_row_groups(validation).to_pandas()
    X_test = pt.from_numpy(testData.drop(ynames + badnames, axis=1).to_numpy()).float()
    y_test = pt.from_numpy(testData[ynames].to_numpy()).float()
    
    xmean = pt.mean(X_test, 0)
    xstd = pt.std(X_test, 0)
    pt.save(xmean, "xmean.pt")
    pt.save(xstd, "xstd.pt")

    X_test = (X_test - xmean) / xstd

cpu_batch_size = 4

# Figure out how many rowgroups there are overall
num_loops = 0
for data_file in data_files:
    with pq.ParquetFile(data_file) as fulldata:
        num_loops += len(list(it.batched(range(fulldata.num_row_groups), cpu_batch_size)))
num_loops -= len(validation) // cpu_batch_size

# Return data in batches that fit in total memory
def data_generator():
    """Return data in batches that fit in memory"""
    global validation
    global xstd
    global xmean

    for data_file in data_files:
        with pq.ParquetFile(data_file) as fulldata:
            n_rowgroups = fulldata.num_row_groups
            all_data = set(range(n_rowgroups))
            if data_file == validation_file:
                all_data = all_data.difference(validation)
            
            all_data = sorted(list(all_data))

            for idxs in it.batched(all_data, cpu_batch_size):
                table: pd.DataFrame = fulldata.read_row_groups(idxs).to_pandas(self_destruct = True).replace([np.inf, -np.inf], np.nan).dropna()

                train_X = pt.from_numpy(table.drop(ynames + badnames, axis=1).to_numpy()).float()
                train_X = (train_X - xmean) / xstd
                train_y = pt.from_numpy(table[ynames].to_numpy()).float()

                yield train_X, train_y

# Return batches that fit in GPU memory
def rebatch(x: pt.Tensor, y: pt.Tensor, n: int):
    """
    Return data in batches that fit in GPU memory
    
    Parameters
    ----------
        x (pt.tensor): x data
            
        y (pt.tensor): y data
        
        n (int): Number of row_groups to read in 1 batch
    """
    currPlace = 0
    batchlen = (x.shape[0] // n) + 1
    while currPlace < x.shape[0]:
        nextPlace = currPlace + batchlen
        yield x[currPlace:nextPlace], y[currPlace:nextPlace]
        currPlace = nextPlace
    
def run(world_size, rank):
    global model
    global X_test
    global y_test
    global xmean
    global xstd
    global device
    global inp_dim
    global h_dim
    global out_dim
    global data_generator
    global rebatch
    global num_loops

    # Setup the distributed trainer
    num_epochs = 100
    gpu_batch_size = 2

    # model.load_state_dict(pt.load("checkpoints/epoch_25.pt"))

    model = DistributedDataParallel(model, [0])

    optimizer = pt.optim.Adam(model.parameters(), lr=0.0008)
    # optimizer.load_state_dict(pt.load("checkpoints/optim_25.pt"))

    scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    # scheduler.load_state_dict(pt.load("checkpoints/sched_25.pt"))
    # scheduler.step()
    start_epoch = 0

    lossfn = nn.CrossEntropyLoss(reduction="sum")

    print(f"Ready on {rank}", flush=True)
    dist.barrier()

    for epoch in range(start_epoch, num_epochs):
        if rank == 0:
            print(f"Epoch {epoch} / {num_epochs}")
        startTime = time()
        
        epoch_train_loss = 0
        n_epoch = 0

        dataloader = data_generator()

        # Evaluate model
        if rank == 0:
            model.eval()
            with pt.no_grad():
                tot_loss = 0
                n = 0
                for xb, yb in rebatch(X_test, y_test, gpu_batch_size):
                    xb = xb.to(device)
                    yb = yb.to(device)
                    output = model(xb)
                    loss = lossfn(output, yb)
                    
                    n += xb.shape[0]
                    tot_loss += loss.item()
                    
                    del output
                    del loss
                print(f"    Test Loss: {tot_loss / n:.3f}")
            
            del xb
            del yb

            gc.collect()
            pt.cuda.empty_cache()
        
        dist.barrier()

        # Train the model
        model.train()
        for X_train_o, y_train_o in tqdm.tqdm(dataloader, total=num_loops, disable=(rank != 0)):
            if rank == 0:
                stdout.flush()
                stderr.flush()
            for X_train, y_train in rebatch(X_train_o, y_train_o, gpu_batch_size):
                X_train = X_train.to(device)
                y_train = y_train.to(device)
                optimizer.zero_grad()
                
                output = None
                loss = None
                with pt.set_grad_enabled(True):
                    output = model(X_train)
                    loss = lossfn(output, y_train)
                    loss.backward()
                    optimizer.step()
                    del output

                if rank == 0:
                    epoch_train_loss += loss.item()
                    n_epoch += X_train.shape[0]
                del loss
                del X_train
                del y_train

                gc.collect()
                pt.cuda.empty_cache()

        if rank == 0:
            print(f"    Train Loss (rank 0): {epoch_train_loss / n_epoch:.3f}")
            endTime = time()
            print(f"    Time Elapsed: {(endTime - startTime):.3f} seconds")

        if (epoch % 20 == 0) and (rank == 1):
            pt.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")
            pt.save(optimizer.state_dict(), f"checkpoints/optim_{epoch}.pt")
            pt.save(scheduler.state_dict(), f"checkpoints/sched_{epoch}.pt")
        scheduler.step()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.environ.get("SLURM_NTASKS"))
    rank = int(os.environ.get("SLURM_PROCID"))
    dist.init_process_group("nccl", init_method=Path(new_data_loc("comms")).resolve().as_uri(), world_size=world_size, rank=rank, device_id=0)
    run(world_size, rank)