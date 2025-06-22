import pandas as pd
import numpy as np
import itertools

# Basic collision resolving clusterer
def do_cluster(closest, closest_dists) -> pd.DataFrame:
    closest = np.array(closest)

    # Ensure a consistent number is used to indicate inf distance
    closest[closest_dists == np.inf] = closest.shape[1] - 1
    
    # Cluster.
    clusters = []
    clusterNum = 0
    clusterNums = [-1] * closest.shape[1]
    for i in range(closest.shape[1]):
        elems = closest[:, i]
        dists = closest_dists[:, i]
        if elems[0] == closest.shape[1] - 1:
            if clusterNums[i] == -1:
                clusterNums[i] = clusterNum
                clusters.append([i])
                clusterNum += 1
        else:
            if clusterNums[i] == -1:
                clusterNums[i] = clusterNum
                clustered = False
                for j, elem in enumerate(elems):
                    if clusterNums[elem] == -1:
                        dist = dists[j]
                        collisionDists = closest_dists[:, i:][closest[:, i:] == elem]
                        if dist <= np.min(collisionDists):
                            clustered = True
                            clusterNums[elem] = clusterNum
                            clusters.append([i, elem])
                            break
                if not clustered:
                    clusters.append([i])
                clusterNum += 1
            else:
                for j, elem in enumerate(elems):
                    if clusterNums[elem] == -1:
                        dist = dists[j]
                        collisionDists = closest_dists[:, i:][closest[:, i:] == elem]
                        if dist <= np.min(collisionDists):
                            clusterNums[elem] = clusterNums[i]
                            clusters[clusterNums[i]].append(elem)
                            break
        if i % 1000 == 0:
            print(i)

    point_ids = np.array(list(itertools.chain(*clusters)))
    track_ids = np.array(list(itertools.chain(*[[i for z in x] for i, x in enumerate(clusters)])))
    output = pd.DataFrame(np.array([point_ids, track_ids]).T, columns=["point_id", "track_id"]).sort_values("point_id")
    return output