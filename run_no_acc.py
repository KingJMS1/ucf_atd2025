from ucf_atd_model import river_and_base_model, baseline, data
import atd2025

file = "dataset3.csv"
result, path = river_and_base_model.run(file)
# resultB, pathB = baseline.run(file)

print(path)