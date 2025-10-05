import numpy as np
import scipy as sp
import pandas as pd
from sklearn import cluster
import atd2025

# Read dataset in
data = pd.read_csv("https://www.maserv.work/ATD/model2/ucf_atd_model/datasets/dataset1_truth.csv")
data["time"] = pd.to_datetime(data["time"])

# ktm mult should be < 1, speed_cutoff should differentiate between 'fast' and 'slow' ships
def bidirectional_distance(x1, y1, t1, speed1, course1, time_window):
    # Setup proper broadcasting rules
    x2 = x1[:, None]
    y2 = y1[:, None]
    t2 = t1[:, None]
    speed2 = speed1[:, None]
    course2 = course1[:, None]

    time_window = time_window.astype("float32")

    x1 = x1[None, :]
    y1 = y1[None, :]
    t1 = t1[None, :]
    speed1 = speed1[None, :]
    course1 = course1[None, :]

    knots_to_mps = 0.514444
    
    # Forward prediction
    dt = (t2 - t1).astype("timedelta64[s]").astype("float32")
    pred_x = x1 + speed1 * np.sin(course1) * dt * knots_to_mps
    pred_y = y1 + speed1 * np.cos(course1) * dt * knots_to_mps
    forward_dist = np.square(x2 - pred_x) + np.square((y2 - pred_y))

    del pred_x
    del pred_y

    # Backward prediction
    back_x = x2 - speed2 * np.sin(course2) * dt * knots_to_mps
    back_y = y2 - speed2 * np.cos(course2) * dt * knots_to_mps
    backward_dist = np.square(x1 - back_x) + np.square(y1 - back_y)

    del back_x
    del back_y
    
    dist = 0.5 * (forward_dist + backward_dist)

    # Can try to apd this factor if we find many vessels close to one another in space and time
    # | (dt < (time_window / 2)) & (speed1 == 0)
    # Ensure that we only look within the next time_window if speed is not 0. 
    toInf = (dt <= 0) | ((dt > time_window) & (speed1 != 0)) | (dt < 300)
    
    dist[toInf] = np.inf
    
    return dist

rad_earth = 6371000

# Approximately convert lon/lat to x/y in meters
def lonlat_to_xy(lon, lat):
    return lon * (np.pi / 180) * rad_earth * np.cos(lat * (np.pi / 180)), lat * rad_earth * (np.pi / 180)

# Approximately convert back to lon/lat
def xy_to_lonlat(x, y):
    lat = y * (180 / (np.pi * rad_earth))
    return x * (180 / (np.pi * rad_earth * np.cos(lat * (np.pi / 180)))), lat

# Read in coordinates
lon = data["lon"].to_numpy().astype("float32")
lat = data["lat"].to_numpy().astype("float32")

# Convert to meters
x, y = lonlat_to_xy(lon, lat)

x = x.astype("float32")
y = y.astype("float32")

# Read in rest of data
t = data["time"].to_numpy()
speed = data["speed"].to_numpy().astype("float32")
course = data["course"].to_numpy().astype("float32") * np.pi / 180

print("Start")

ds_size = data.shape[0]
ground = data[:ds_size]
data[:ds_size].to_csv("ground.csv")

# Initialize hyperparameters
time_window = 2100 # Should be somewhere around 1800
windowLen = np.array([time_window], dtype="timedelta64[s]")

# Create distance matrix, get top 10 values and their indices.
dist_matrix = bidirectional_distance(x[:ds_size], y[:ds_size], t[:ds_size], speed[:ds_size], course[:ds_size], windowLen[:ds_size])
dist_matrix[dist_matrix == np.inf] = 1e30 # Set infinities to some large number so sklearn does not complain
print("Distance matrix calculated")

print("DBSCAN accuracy:")
dbscan = cluster.DBSCAN(eps=40000, metric="precomputed")
out = dbscan.fit_predict(dist_matrix.T)
out[out == -1] = np.arange(np.max(out) + 1, np.max(out) + 1 + np.sum(out == -1))
toGrade = pd.DataFrame({"point_id": data[:ds_size]["point_id"], "track_id": out})
toGrade.to_csv("grademe.csv")
print(atd2025.accuracy.evaluate_predictions("grademe.csv", "ground.csv"))

print("\nAverage linkage accuracy:")
average = cluster.AgglomerativeClustering(distance_threshold=80000000, metric="precomputed", linkage="average", n_clusters=None)
out = average.fit_predict(dist_matrix.T)
out[out == -1] = np.arange(np.max(out) + 1, np.max(out) + 1 + np.sum(out == -1))
toGrade = pd.DataFrame({"point_id": data[:ds_size]["point_id"], "track_id": out})
toGrade.to_csv("grademe.csv")
print(atd2025.accuracy.evaluate_predictions("grademe.csv", "ground.csv"))

print("\nComplete linkage accuracy:")
complete = cluster.AgglomerativeClustering(distance_threshold=100000000, metric="precomputed", linkage="complete", n_clusters=None)
out = complete.fit_predict(dist_matrix.T)
out[out == -1] = np.arange(np.max(out) + 1, np.max(out) + 1 + np.sum(out == -1))
toGrade = pd.DataFrame({"point_id": data[:ds_size]["point_id"], "track_id": out})
toGrade.to_csv("grademe.csv")
print(atd2025.accuracy.evaluate_predictions("grademe.csv", "ground.csv"))

print("\nSingle linkage accuracy:")
single = cluster.AgglomerativeClustering(distance_threshold=100000, metric="precomputed", linkage="single", n_clusters=None)
out = single.fit_predict(dist_matrix.T)
out[out == -1] = np.arange(np.max(out) + 1, np.max(out) + 1 + np.sum(out == -1))
toGrade = pd.DataFrame({"point_id": data[:ds_size]["point_id"], "track_id": out})
toGrade.to_csv("grademe.csv")
print(atd2025.accuracy.evaluate_predictions("grademe.csv", "ground.csv"))