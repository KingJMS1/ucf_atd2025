import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import os
import gc
import tqdm
from ucf_atd_model.data import data_loc, new_data_loc
import pyproj as pp
import numpy as np

wgs84 = pp.CRS.from_epsg(4326)
utm = pp.CRS.from_epsg(32616)
to_utm = pp.Transformer.from_crs(wgs84, utm)

folder = data_loc("lgbm_data")

schema = pa.schema([
    pa.field("distance_m", pa.float64()),
    pa.field("implied_speed_knots", pa.float64()),
    pa.field("delta_speed", pa.float64()),
    pa.field("delta_course", pa.float64()),
    pa.field("bearing_diff", pa.float64()),
    pa.field("kinematic_error", pa.float64()),
    pa.field("delta_time", pa.float64()),
    pa.field("y1", pa.float64()), 
    pa.field("y2", pa.float64()), 
    pa.field("x1", pa.float64()),
    pa.field("x2", pa.float64()),
    pa.field("t1", pa.float64()),
    pa.field("t2", pa.float64()),
    pa.field("speed1", pa.float64()),
    pa.field("speed2", pa.float64()),
    pa.field("course1", pa.float64()),
    pa.field("course2", pa.float64()),
    pa.field("dx1", pa.float64()),
    pa.field("dy1", pa.float64()),
    pa.field("dx2", pa.float64()),
    pa.field("dy2", pa.float64()),
    pa.field("label", pa.int64())
])

pwriter = pq.ParquetWriter(new_data_loc("link_data_small.parquet"), schema=schema)
files = os.listdir(folder)
for filename in tqdm.tqdm(files):
    path = os.path.join(folder, filename)
    
    # Read in data
    data = pd.read_csv(path)
    data = data.drop(["Unnamed: 0"], axis=1)
    data["t1"] = (pd.to_datetime(data["t1"]) - pd.to_datetime("1970-01-01 00:00:00")) / pd.offsets.Second(1)
    data["t2"] = (pd.to_datetime(data["t2"]) - pd.to_datetime("1970-01-01 00:00:00")) / pd.offsets.Second(1)
    
    x1, y1 = to_utm.transform(data["y1"].to_numpy(), data["x1"].to_numpy())
    x2, y2 = to_utm.transform(data["y2"].to_numpy(), data["x2"].to_numpy())

    data["x1"] = x1 / 1000
    data["y1"] = y1 / 1000
    data["x2"] = x2 / 1000
    data["y2"] = y2 / 1000

    data["dx1"] = 0.000514444 * data["speed1"] * np.sin(data["course1"] * (np.pi) / 180)
    data["dy1"] = 0.000514444 * data["speed1"] * np.cos(data["course1"] * (np.pi) / 180)
    data["dx2"] = 0.000514444 * data["speed2"] * np.sin(data["course2"] * (np.pi) / 180)
    data["dy2"] = 0.000514444 * data["speed2"] * np.cos(data["course2"] * (np.pi) / 180)

    data = pd.DataFrame(data[["distance_m", "implied_speed_knots", "delta_speed", "delta_course", "bearing_diff", "kinematic_error", "delta_time", "y1", "y2", "x1", "x2", "t1", "t2", "speed1", "speed2", "course1", "course2", "dx1", "dy1", "dx2", "dy2", "label"]])

    # Convert to parquet recordbatch
    batch = pa.RecordBatch.from_pandas(data, schema = schema, preserve_index = False)
    pwriter.write_batch(batch)
    
pwriter.close()