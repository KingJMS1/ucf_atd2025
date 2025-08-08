import pandas as pd
import torch as pt
import torch.nn as nn
from time import time
import pyarrow.parquet as pq
import random
import itertools as it
import gc
import tqdm

from ucf_atd_model.data import data_loc, ResultCache

cache = ResultCache("deep")

inp_dim = 21
h_dim = 500
out_dim = 1

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
    nn.Linear(h_dim // 2, h_dim // 10),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(h_dim // 10, 1),
    nn.Sigmoid()
).to(device)

xmean = pt.load("xmean.pt").float()
xstd = pt.load("xstd.pt").float()

num_loops = None
validation = None
def data_generator():
    global validation
    global xstd
    global xmean
    global num_loops

    batch_size = 8192
    table: pd.DataFrame = None
    with pq.ParquetFile(data_loc("link_data_small.parquet")) as fulldata:
        n_rowgroups = fulldata.num_row_groups
        all_data = set(range(n_rowgroups))
        if validation is None:
            validation = random.sample(sorted(all_data), 3)
        all_data = all_data.difference(validation)
        testData = fulldata.read_row_groups(validation).to_pandas()
        
        test_X = pt.from_numpy(testData.drop(["label"], axis=1).values).float()
        test_X = (test_X - xmean) / xstd
        test_X = test_X.to(device)
        test_y = pt.from_numpy(testData["label"].to_numpy()).float().to(device)
        
        all_data = sorted(list(all_data))
        num_loops = len(list(it.batched(all_data, batch_size)))

        yield test_X, test_y

        for idx in all_data:
            table = fulldata.read_row_group(idx).to_pandas(self_destruct = True)
            
            train_X = pt.from_numpy(table.drop(["label"], axis=1).values).float()
            train_X = (train_X - xmean) / xstd
            train_y = pt.from_numpy(table["label"].to_numpy()).float()

            
            yield train_X, train_y


num_epochs = 100

optimizer = pt.optim.Adam(model.parameters(), lr=0.001)
lossfn = nn.BCELoss()

i = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch} / {num_epochs}")
    startTime = time()
    
    epoch_train_loss = 0

    dataloader = data_generator()
    X_test, y_test = next(dataloader)

    # Evaluate model
    model.eval()
    with pt.no_grad():
        output = model(X_test)
        loss = lossfn(output.flatten(), y_test)
        print(f"    Test Loss: {loss.item():.3f}")
        del output
        del loss
    
    del X_test
    del y_test

    # Train the model
    model.train()
    for X_train, y_train in tqdm.tqdm(dataloader, total=num_loops):
        if i % 100 == 0:
            print(f"Train set is {X_train.nelement() * X_train.element_size() / 1000 / 1000 / 1000} GB")
        i += 1
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        optimizer.zero_grad()
        
        output = None
        loss = None
        with pt.set_grad_enabled(True):
            output = model(X_train)
            loss = lossfn(output.flatten(), y_train)
            loss.backward()
            optimizer.step()
            del output
        
        epoch_train_loss += loss.item()
        del loss
        del X_train
        del y_train

        gc.collect()
        pt.cuda.empty_cache()

    
    print(f"    Train Loss: {epoch_train_loss:.3f}")
    endTime = time()
    print(f"    Time Elapsed: {(endTime - startTime):.3f} seconds")

    pt.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")
    