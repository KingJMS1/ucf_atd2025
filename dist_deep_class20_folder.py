import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import tqdm
from pathlib import Path

from time import time
import gc
import os

from ucf_atd_model.data import data_loc, new_data_loc

import torch as pt
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule

from ucf_atd_model.c20_consts import *
import ucf_atd_model.c20_consts as const
from ucf_atd_model.datasets.pt_datasets import C20data

    
def run(world_size, rank):
    dist.init_process_group("nccl", init_method=Path(new_data_loc("comms")).resolve().as_uri(), world_size=world_size, rank=rank, device_id=0)
    badnames = [x for x in full_names if x.endswith("_16")]
    ynames = [x for x in const.ynames if not x.endswith("_16")]

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
    y_test_mask = None
    xmean = None
    xstd = None

    with pq.ParquetFile(validation_file) as fulldata:
        testData = fulldata.read_row_groups(validation).to_pandas()
        X_test = pt.from_numpy(testData.drop(ynames + badnames, axis=1).to_numpy()).float()
        y_test = pt.from_numpy(testData[ynames].to_numpy()).float()
        y_test_mask = pt.zeros_like(y_test, dtype=pt.float32)
        y_test_mask[:, -1] = 0
        y_test_mask[:, 0] = 0

        num_ft_sets = n_norm_classes - 1
        for i in range(1, num_ft_sets):
            ft_names = getNormFeatures(i)
            feats = testData[ft_names]
            y_test_mask[:, i] = pt.from_numpy(((feats == -1).all(axis=1) * -1e8).to_numpy()).float()

        xmean = pt.mean(X_test, 0)
        xstd = pt.std(X_test, 0)
        pt.save(xmean, "xmean.pt")
        pt.save(xstd, "xstd.pt")

        X_test = (X_test - xmean) / xstd

    y_test_mask_gpu = y_test_mask.to(device)

    # Setup the distributed trainer
    num_epochs = 100
    gpu_batch_size = 1

    # model.load_state_dict(pt.load("checkpoints/epoch_25.pt"))

    model = DistributedDataParallel(model, [0])

    optimizer = pt.optim.Adam(model.parameters(), lr=0.0008)
    # optimizer.load_state_dict(pt.load("checkpoints/optim_25.pt"))

    scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    # scheduler.load_state_dict(pt.load("checkpoints/sched_25.pt"))
    # scheduler.step()
    start_epoch = 0

    lossfn = nn.CrossEntropyLoss(reduction="sum")

    dataset = C20data(validation, validation_file, data_files, badnames, ynames, xstd, xmean, gpu_batch_size)
    dataloader = pt.utils.data.DataLoader(dataset, num_workers=2, prefetch_factor=2)

    print(f"Ready on {rank}", flush=True)
    dist.barrier()

    tracing_schedule = schedule(wait=100, warmup=20, active=10)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        with_flops=False,
        schedule=tracing_schedule,
        on_trace_ready=tensorboard_trace_handler(f"tboard", rank, use_gzip=True)
    ) as prof:
            
        for epoch in range(start_epoch, num_epochs):
            if rank == 0:
                print(f"Epoch {epoch} / {num_epochs}")
            startTime = time()
            
            epoch_train_loss = 0

            # Evaluate model
            if rank == 0:
                model.eval()
                with pt.no_grad():
                    tot_loss = 0
                    output = model(X_test.to(device))
                    loss = lossfn(output, y_test.to(device) + y_test_mask_gpu)
                    
                    tot_loss += loss.item()
                    
                    del output
                    del loss
                    print(f"    Test Loss: {tot_loss:.3f}")
                

                gc.collect()
                pt.cuda.empty_cache()
            
            dist.barrier()

            # Train the model
            model.train()
            for X_train, y_train in tqdm.tqdm(dataloader, total=dataset.num_loops, disable=(rank != 0)):
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
                del loss
                del X_train
                del y_train

                gc.collect()
                pt.cuda.empty_cache()
                prof.step()

            if rank == 0:
                print(f"    Train Loss (rank 0): {epoch_train_loss:.3f}")
                endTime = time()
                print(f"    Time Elapsed: {(endTime - startTime):.3f} seconds")

            if (epoch % 10 == 0) and (rank == 1):
                pt.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")
                pt.save(optimizer.state_dict(), f"checkpoints/optim_{epoch}.pt")
                pt.save(scheduler.state_dict(), f"checkpoints/sched_{epoch}.pt")
            scheduler.step()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.environ.get("SLURM_NTASKS"))
    rank = int(os.environ.get("SLURM_PROCID"))
    run(world_size, rank)