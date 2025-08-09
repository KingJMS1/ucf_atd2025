import pyarrow.parquet as pq
from multiprocessing import Pool
from ucf_atd_model.data import data_loc
from ucf_atd_model.datasets.create_20class_data import run
from sys import argv
import itertools as it

if __name__ == "__main__":
    rowgroups = None
    if len(argv) < 2:
        print("Missing argument 2, must be 0, 1, or 2.")
        exit()

    batch_num = int(argv[1])

    # Read number of row groups
    with pq.ParquetFile(data_loc("historical.parquet")) as fulldata:
        rowgroups = fulldata.num_row_groups

    batches = list(it.batched(range(rowgroups), (rowgroups // 3) + 1))
    mybatch = batches[batch_num]

    with Pool(processes=1) as pool:
        results = pool.map(run, mybatch)
        for i, result in enumerate(results):
            if result != "":
                print(i)
                print(result)
                print()

