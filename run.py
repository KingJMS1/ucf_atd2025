from ucf_atd_model import river_and_base_model, baseline, data
import atd2025

file = "dataset1_truth.csv"
result, path = river_and_base_model.run(file)
resultB, pathB = baseline.run(file)

score = atd2025.accuracy.evaluate_predictions(path, data.data_loc(file))
scoreBase = atd2025.accuracy.evaluate_predictions(pathB, data.data_loc(file))

print(score)
print(scoreBase)