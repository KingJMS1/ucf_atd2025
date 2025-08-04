import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import os
import gc
import tqdm
from ucf_atd_model.data import data_loc, new_data_loc

folder = data_loc("lgbm_data")

schema = pa.schema([
    pa.field("distance_m", pa.float64()),
    pa.field("implied_speed_knots", pa.float64()),
    pa.field("delta_speed", pa.float64()),
    pa.field("delta_course", pa.float64()),
    pa.field("bearing_diff", pa.float64()),
    pa.field("kinematic_error", pa.float64()),
    pa.field("delta_time", pa.float64()),
    pa.field("label", pa.int64())
])

pwriter = pq.ParquetWriter(new_data_loc("link_data_small.parquet"), schema=schema)
files = os.listdir(folder)
for filename in tqdm.tqdm(files):
    path = os.path.join(folder, filename)
    
    # Read in data
    data = pd.read_csv(path)
    data = data.drop(["Unnamed: 0"], axis=1)

    # Convert to parquet recordbatch
    batch = pa.RecordBatch.from_pandas(data, schema = schema, preserve_index = False)
    pwriter.write_batch(batch)
    
pwriter.close()