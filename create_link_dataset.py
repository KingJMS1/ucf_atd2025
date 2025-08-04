import pyarrow.parquet as pq
from multiprocessing import Pool
from ucf_atd_model.data import data_loc
from ucf_atd_model.datasets.create_link_data import run

rowgroups = None

# Read number of row groups
with pq.ParquetFile(data_loc("historical.parquet")) as fulldata:
    rowgroups = fulldata.num_row_groups


with Pool(processes=5) as pool:
    results = pool.map(run, range(rowgroups))
    for i, result in enumerate(results):
        if result != "":
            print(i)
            print(result)
            print()