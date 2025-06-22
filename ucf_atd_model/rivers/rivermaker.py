import pyproj as pp
import geopandas as gpd
import numpy as np
import shapely as shp
import itertools
import os
import urllib.request

def get_river_data() -> list[tuple[str, shp.LineString, pp.CRS]]:
    # Setup projections
    wgs84 = pp.CRS.from_epsg(4326)
    miss_utm = pp.CRS.from_epsg(32615)
    icw_sfl_utm = pp.CRS.from_epsg(32617)

    # Read in river data
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    filePath = os.path.join(location, "USA_Rivers_and_Streams.geojson")
    if not os.path.isfile(filePath):
        # Download the file if necessary
        print("Downloading river information as it was not found.")
        with open(filePath, 'wb') as f:
            with urllib.request.urlopen('https://www.maserv.work/ATD/model2/ucf_atd_model/rivers/USA_Rivers_and_Streams.geojson') as network:
                f.write(network.read())

    rivers = gpd.read_file(filePath)

    # Setup any rivers we will use
    miss_river = rivers[rivers["Name"].str.contains("Mississippi River")]
    miss_river = miss_river[miss_river["State"].str.contains("LA")]
    miss_ilocs = [5, -1, -2, 1]
    miss_river = miss_river.iloc[miss_ilocs].to_crs(miss_utm)

    miss_geom_parts = shp.ops.linemerge(miss_river.union_all())
    coords = [[y for y in x.coords] for x in miss_geom_parts.geoms]
    coords = [x for x in itertools.chain(*coords)]
    miss_geom = shp.LineString(coords)

    icw_all = rivers[rivers["Name"].str.contains("Intracoastal Waterway")]
    icw_fl = icw_all[icw_all["State"].str.contains("FL")]
    icw_fl_sfl_ilocs = [-3]
    icw_sfl = icw_fl.iloc[icw_fl_sfl_ilocs].to_crs(icw_sfl_utm)
    icw_sfl_geom = icw_sfl.geometry.iloc[0]

    icw_n_lonmin = -100
    icw_n_lonmax = -84
    icw_n_latmin = 24
    icw_n_latmax = 32
    icw_n = icw_all[~icw_all.clip_by_rect(icw_n_lonmin, icw_n_latmin, icw_n_lonmax, icw_n_latmax).is_empty].copy()
    icw_n["id"] = np.arange(icw_n.shape[0])

    # Export the river names, geometries, and utm zones used by each
    output = [
        ("Mississippi", miss_geom, miss_utm), 
        # ("ICW_SFL", icw_sfl_geom, icw_sfl_utm)    # Makes accuracy worse for now, ignore
    ]
    
    return output