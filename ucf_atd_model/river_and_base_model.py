import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj as pp
import gc
import sklearn.cluster
import itertools

from .simple_clusterer import do_cluster
from .data import *
from .rivers import get_river_data
from .baseline import run as run_baseline

cache = ResultCache("river_and_base")

# Important constatnts
wgs84 = pp.CRS.from_epsg(4326)
rad_earth = 6371000

# Distance metric for dbscan, penalizes time differences
def dbscan_dist(x1, y1, t1, x2, y2, t2) -> np.ndarray:
    x1 = x1[None, :]
    y1 = y1[None, :]
    t1 = t1[None, :]
    x2 = x2[:, None]
    y2 = y2[:, None]
    t2 = t2[:, None]

    dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2) + np.square(0.005 * (t2 - t1).astype("timedelta64[s]").astype("float")))
    return dist

# Distance metric for river model, basically CBTR but in 1d
def bidirectional_distance_river(x1, t1, v1, velocity_penalty) -> np.ndarray:
    # Setup proper broadcasting rules
    x2 = x1[:, None]
    t2 = t1[:, None]
    v2 = v1[:, None]

    x1 = x1[None, :]
    t1 = t1[None, :]
    v1 = v1[None, :]

    # Forward prediction
    dt = (t2 - t1).astype("timedelta64[s]").astype("float64")
    pred_x = x1 + v1 * dt
    forward_dist = np.square(x2 - pred_x)

    # Backward prediction
    back_x = x2 - v2 * dt
    backward_dist = np.square(x1 - back_x)

    vdiff = np.square(v2 - v1)

    # Penalize changes in velocity up/downriver
    dist = 0.5 * (forward_dist + backward_dist) + velocity_penalty * vdiff
    toInf = (dt <= 0)
    dist[toInf] = np.inf
    return dist


# Runs the fully hybridized model, CBTR on most of the data, DBSCAN + River model (1d CBTR with velocity penalty) on rivers
def run(file: str, **kwargs):
    # Handle caching with kwargs
    kwargs_str = "_k".join([str(x) for x in itertools.chain(*kwargs.items())])
    in_cache, path = cache.test_cache(file + kwargs_str)
    if in_cache:
        return pd.read_csv(path), path
    
    # Read in dataset
    truth = pd.read_csv(data_loc(file))
    truth["time"] = pd.to_datetime(truth["time"])
    truth = gpd.GeoDataFrame(truth, geometry=gpd.points_from_xy(truth["lon"], truth["lat"]), crs=wgs84)
    truth["course"] = truth["course"] * (np.pi / 180)

    ktmps = 0.514444 # Convert knots to meters per second
    
    # Allow these hyperparameters as kwargs
    kwl = ["velocity_penalty", "speed_cutoff"]
    hyp = [1000000, 0.7]
    for i, key in enumerate(kwl):
        if key in kwargs:
            hyp[i] = kwargs[key]
    velocity_penalty, speed_cutoff = hyp

    # Load up the river information
    rivers = get_river_data()

    # Run the model on subsets of the data located on rivers/waterways
    in_river = np.repeat(np.False_, truth.shape[0])
    print(in_river.shape)
    river_results = []
    river_filters = []
    print("Running river models")
    for rname, rgeom, rcrs in rivers:
        print("  " + rname)
        truth_proj = truth.to_crs(rcrs)                                     # Project to the coordinate system the river is in
        rfilter = rgeom.distance(truth_proj.geometry).to_numpy() < 1000     # Filter down to points within 1km of the river
        print(np.sum(rfilter))
        rfilter = rfilter & (~in_river)                                     # Ensure we don't conflict with more important rivers, list is sorted in order of importance
        truth_proj = truth_proj[rfilter]
        # Save the subset and make sure we don't use these points again
        river_filters.append(rfilter)
        in_river = in_river | rfilter

        # Setup to apply dbscan to slow points
        xy = truth_proj.get_coordinates()
        x = xy["x"].to_numpy()
        y = xy["y"].to_numpy()
        speed = truth_proj["speed"].to_numpy()
        t = truth_proj["time"].to_numpy()
        course = truth_proj["course"].to_numpy()

        # Get slow points
        slows = speed < speed_cutoff
        lx = x[slows]
        ly = y[slows]
        lt = t[slows]

        # Ensure we don't run dbscan on nothing
        clusters_dbscan = np.array([])
        if np.sum(slows) != 0:
            # Cluster slow points
            dists = dbscan_dist(lx, ly, lt, lx, ly, lt)
            dbscan = sklearn.cluster.DBSCAN(eps=175, metric="precomputed", min_samples=2)
            clusters_dbscan = dbscan.fit_predict(dists)

        # Get distance along the length of the river of all points
        dist_along_river = rgeom.project(truth_proj.geometry).to_numpy()

        # Figure out which direction the ships are moving along the river
        dy = speed * ktmps * np.cos(course)
        dx = speed * ktmps * np.sin(course)
        resX = x + dx
        resY = y + dy
        resPts = gpd.points_from_xy(resX, resY)
        resDist = rgeom.project(resPts)
        origDist = dist_along_river
        upriver = ((resDist - origDist) > 0)

        # Change the speeds into velocities along the river
        v_pred = ktmps * speed * (upriver * 2 - 1)
    
        # Calculate distance matrix for fast vessels on the river, ignoring all slow traffic
        dist_matrix = bidirectional_distance_river(dist_along_river[~slows], t[~slows], v_pred[~slows], velocity_penalty)
        closest = np.argsort(dist_matrix, axis=0)[:10]
        closest_dists = np.sort(dist_matrix, axis=0)[:10]
        gc.collect() # Ensure that we don't run out of memory from running these computations repeatedly
        
        # Get clusters for fast vessels
        rout_fast = do_cluster(closest, closest_dists)
        
        # Interleave slow results with fast results
        nullID = 0
        if len(clusters_dbscan) != 0:
            nullID = np.max(clusters_dbscan + 2)
        newOut = np.ones_like(speed, "int64") * nullID
        newOut[slows] = clusters_dbscan                                             # Fill slow points with DBSCAN results
        newOut[newOut == nullID] = (rout_fast["track_id"].to_numpy() + nullID)      # Replace fast points with results from fast model
        newOut[newOut == -1] = np.arange(-1 * np.sum(newOut == -1), 0)              # Correct unclustered points from DBSCAN
        newOut = newOut - np.min(newOut)
        river_results.append(newOut)

    print("River models finished.")

    # Run the baseline and get those clusters
    normal_clusters = run_baseline(file)[0]

    # Interleave predictions from the various models
    results = np.repeat(-1, truth.shape[0])
    results[~in_river] = normal_clusters["track_id"][~in_river].to_numpy()
    for river_result, river_filter in zip(river_results, river_filters):
        results[river_filter] = river_result + np.max(results) + 2

    output = pd.DataFrame({"point_id": truth["point_id"], "track_id": results})
    output.to_csv(path, index=False)

    return output, path