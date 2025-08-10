import pyarrow as pa
import pyarrow.parquet as pq
from ucf_atd_model.data import data_loc, new_data_loc
import tqdm
import os
import gc

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

pwriter = pq.ParquetWriter(new_data_loc(f"class20_final.parquet"), schema=schema)

files = [data_loc("c20_data") + "/" + x for x in os.listdir(data_loc("c20_data"))]


for j, filename in enumerate(files):
    gc.collect()
    with pq.ParquetFile(filename) as data:
        rowgroups = data.num_row_groups
        print(f"Processing file {j} / {len(files)}")
        for i in tqdm.tqdm(range(rowgroups)):
            table = data.read_row_group(i)
            for batch in table.to_batches():
                pwriter.write_batch(batch)

pwriter.close()