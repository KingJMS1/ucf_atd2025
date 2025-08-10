import pandas as pd
import torch as pt
import torch.nn as nn
from time import time
import pyarrow.parquet as pq
import random
import itertools as it
import gc
import tqdm
import numpy as np
from pickle import dump

from ucf_atd_model.data import data_loc, ResultCache

link_features = [
    "distance_m",
    "implied_speed_knots",
    "delta_speed",
    "delta_course",
    "bearing_diff",
    "kinematic_error",
    "delta_time",
    "y1", 
    "y2", 
    "x1",
    "x2",
    "t1",
    "t2",
    "speed1",
    "speed2",
    "course1",
    "course2",
    "dx1",
    "dy1",
    "dx2",
    "dy2",
]

currpt_features = ["dy2", "dx2", "course2", "speed2", "t2", "x2", "y2"]
normal_features = [x for x in link_features if x not in currpt_features]
getNormFeatures = lambda i: [x + f"_{i}" for x in normal_features]

colnames = sum([getNormFeatures(i) for i in range(20)], start=[])
ynames = [f"y_{i}" for i in range(21)]

full_names = colnames + ynames

inp_dim = 280
h_dim = 2000
out_dim = 21

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

num_loops = None
validation = None
def data_generator():
    global validation
    global xstd
    global xmean
    global num_loops

    table: pd.DataFrame = None
    with pq.ParquetFile(data_loc("class20_final.parquet")) as fulldata:
        n_rowgroups = fulldata.num_row_groups
        all_data = set(range(n_rowgroups))
        if validation is None:
            validation = random.sample(sorted(all_data), 1)
        all_data = all_data.difference(validation)
        testData = fulldata.read_row_groups(validation).to_pandas()
        test_X = pt.from_numpy(testData.drop(ynames, axis=1).to_numpy()).float()
        test_y = pt.from_numpy(testData[ynames].to_numpy()).float()

        xmean = pt.mean(test_X, 0)
        xstd = pt.std(test_X, 0)
        pt.save(xmean, "xmean.pt")
        pt.save(xstd, "xstd.pt")
        with open("validaton.pkl", 'wb') as f:
            dump(validation, f)

        test_X = (test_X - xmean) / xstd
        test_X = test_X
        test_y = test_y
        
        all_data = sorted(list(all_data))
        num_loops = len(all_data)

        yield test_X, test_y

        for idx in all_data:
            table = fulldata.read_row_group(idx).to_pandas(self_destruct = True).replace([np.inf, -np.inf], np.nan).dropna()

            train_X = pt.from_numpy(table.drop(ynames, axis=1).to_numpy()).float()
            train_X = (train_X - xmean) / xstd
            train_y = pt.from_numpy(table[ynames].to_numpy()).float()

            yield train_X, train_y

def rebatch(x, y, n):
    currPlace = 0
    batchlen = (x.shape[0] // n) + 1
    while currPlace < x.shape[0]:
        nextPlace = currPlace + batchlen
        yield x[currPlace:nextPlace], y[currPlace:nextPlace]
        currPlace = nextPlace
    

num_epochs = 1000
num_batches = 1

# model.load_state_dict(pt.load("checkpoints/epoch_25.pt"))

optimizer = pt.optim.Adam(model.parameters(), lr=0.0001)
# optimizer.load_state_dict(pt.load("checkpoints/optim_25.pt"))

scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
# scheduler.load_state_dict(pt.load("checkpoints/sched_25.pt"))
# scheduler.step()
start_epoch = 0

lossfn = nn.CrossEntropyLoss(reduction="sum")

for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch} / {num_epochs}")
    startTime = time()
    
    epoch_train_loss = 0
    n_epoch = 0

    dataloader = data_generator()
    X_test, y_test = next(dataloader)

    # Evaluate model
    model.eval()
    with pt.no_grad():
        tot_loss = 0
        n = 0
        for xb, yb in rebatch(X_test, y_test, num_batches):
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)
            loss = lossfn(output, yb)
            
            n += xb.shape[0]
            tot_loss += loss.item()
            
            del output
            del loss
        print(f"    Test Loss: {tot_loss / n:.3f}")
    
    del X_test
    del y_test
    del xb
    del yb

    gc.collect()
    pt.cuda.empty_cache()

    # Train the model
    model.train()
    for X_train_o, y_train_o in tqdm.tqdm(dataloader, total=num_loops):
        for X_train, y_train in rebatch(X_train_o, y_train_o, num_batches):
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

            epoch_train_loss += loss.item()
            n_epoch += X_train.shape[0]
            del loss
            del X_train
            del y_train

            gc.collect()
            pt.cuda.empty_cache()

    
    print(f"    Train Loss: {epoch_train_loss / n_epoch:.3f}")
    endTime = time()
    print(f"    Time Elapsed: {(endTime - startTime):.3f} seconds")

    if epoch % 20 == 0:
        pt.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")
        pt.save(optimizer.state_dict(), f"checkpoints/optim_{epoch}.pt")
        pt.save(scheduler.state_dict(), f"checkpoints/sched_{epoch}.pt")
    scheduler.step()
    