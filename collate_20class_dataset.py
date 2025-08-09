import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from ucf_atd_model.data import data_loc, new_data_loc
import tqdm
import os
import gc
from time import sleep

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

schema = pa.schema([pa.field(x, pa.float32()) for x in full_names])

pwriter = pq.ParquetWriter(new_data_loc("class20.parquet"), schema=schema)

while True:
    sleep(500)
    
    folder = data_loc("class20")
    filesToCompress = None
    try:
        filesToCompress = os.listdir(folder)
    except:
        continue
    xfiles = [x for x in filesToCompress if "xdata" in x]
    yfiles = [x for x in filesToCompress if "ydata" in x]

    # Ensure both files are present
    xnums = [int(x.split("_")[1].removesuffix(".csv")) for x in xfiles]
    ynums = [int(x.split("_")[1].removesuffix(".npy")) for x in yfiles]

    nums_to_process = [x for x in ynums if x in xnums]

    # Wait for any file writes to finish
    sleep(60)

    print(f"Found {len(nums_to_process)} files to process")
    for i in tqdm.tqdm(nums_to_process):
        xdata_path = os.path.join(folder, f"xdata_{i}.csv")
        ydata_path = os.path.join(folder, f"ydata_{i}.npy")
        
        # Read in data
        xdata = pd.read_csv(xdata_path)
        if "Unnamed: 0" in xdata.keys():
            xdata = xdata.drop(["Unnamed: 0"], axis=1)
        ydata = np.load(ydata_path)

        for j, new_yname in enumerate(ynames):
            xdata[new_yname] = ydata[:, j]        

        xdata = pd.DataFrame(xdata)

        # Convert to parquet recordbatch
        batch = pa.RecordBatch.from_pandas(xdata, schema = schema, preserve_index = False)
        pwriter.write_batch(batch)

        # Remove files from disk
        try:
            os.remove(xdata_path)
            os.remove(ydata_path)
        except:
            continue

    gc.collect()