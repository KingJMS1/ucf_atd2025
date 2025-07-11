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
    mobile_utm = pp.CRS.from_epsg(32616)

    # Read in river data
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    filePaths = [
        "USA_Rivers_and_Streams.geojson", 
        "Houston_Ship_Channel.kml", 
        "Sabine-Neches_Ship_Channel.geojson", 
        "Mobile_Ship_Channel.geojson", 
        "Jacksonville_Port_and_Channel.geojson", 
        "Miami_Port.geojson", 
        "Miami_River.geojson", 
        "Miami_Yacht_Club.geojson", 
        "FT_Lauderdale_River_and_Port.geojson"
    ]
    urls = [ 'https://www.maserv.work/ATD/model2/ucf_atd_model/rivers/' + x for x in filePaths]

    for i, val in enumerate(filePaths):
        filePath = os.path.join(location, val)
        if not os.path.isfile(filePath):
            # Download the file if necessary
            print(f"Downloading river information for {val} as it was not found.")
            with open(filePath, 'wb') as f:
                with urllib.request.urlopen(urls[i]) as network:
                    f.write(network.read())

    filePaths = [os.path.join(location, path) for path in filePaths]

    rivers = gpd.read_file(filePaths[0])
    h_ship_channel = gpd.read_file(filePaths[1])

    # Setup any rivers we will use
    # Mississippi
    miss_river = rivers[rivers["Name"].str.contains("Mississippi River")]
    miss_river = miss_river[miss_river["State"].str.contains("LA")]
    miss_ilocs = [5, -1, -2, 1]
    miss_river = miss_river.iloc[miss_ilocs].to_crs(miss_utm)

    miss_geom_parts = shp.ops.linemerge(miss_river.union_all())
    coords = [[y for y in x.coords] for x in miss_geom_parts.geoms]
    coords = [x for x in itertools.chain(*coords)]
    miss_geom = shp.LineString(coords)

    # Neches
    neches_utm = pp.CRS.from_epsg(32615)
    neches_latmin, neches_latmax = [30, 30.5]
    neches_lonmin, neches_lonmax = [-94.5, -94]

    neches = rivers[rivers["Name"].str.contains("Neches River")]
    # neches = neches.iloc[5:20]
    neches = neches[~neches.clip_by_rect(neches_lonmin, neches_latmin, neches_lonmax, neches_latmax).is_empty].copy()
    neches_geom = neches.to_crs(neches_utm).union_all().segmentize(1000)

    # ICW
    icw_all = rivers[rivers["Name"].str.contains("Intracoastal Waterway")]
    icw_fl = icw_all[icw_all["State"].str.contains("FL")]
    icw_fl_sfl_ilocs = [-3]
    icw_sfl = icw_fl.iloc[icw_fl_sfl_ilocs].to_crs(icw_sfl_utm)
    icw_sfl_geom = icw_sfl.geometry.iloc[0]

    # ICW West
    icw_w_utm = neches_utm
    icw_w_lonmin = -97
    icw_w_lonmax = -90
    icw_w_latmin = 24
    icw_w_latmax = 32
    icw_w = icw_all[~icw_all.clip_by_rect(icw_w_lonmin, icw_w_latmin, icw_w_lonmax, icw_w_latmax).is_empty].copy()
    icw_w["id"] = np.arange(icw_w.shape[0])
    icw_w_geom = shp.ops.linemerge(icw_w.union_all())
    newLS = []
    for ls in icw_w_geom.geoms:
        newLS += [x for x in ls.coords if x[0] < icw_w_lonmax]
    newLS = shp.ops.transform(pp.Transformer.from_crs(wgs84, icw_w_utm, always_xy=True).transform, shp.LineString(newLS))
    icw_w_geom = newLS.segmentize(1000)

    # Houston Ship Channel
    h_ship_utm = icw_w_utm
    h_ship_geom = h_ship_channel.to_crs(h_ship_utm).geometry.iloc[0]
    h_ship_geom = h_ship_geom.segmentize(1000)

    # Sabine-Neches Ship Channel
    sn_utm = icw_w_utm
    sn_ship_geom = gpd.read_file(filePaths[2]).to_crs(sn_utm).geometry.iloc[0]

    # Mobile Ship Channel
    mobile_ship_geom = gpd.read_file(filePaths[3]).to_crs(mobile_utm).geometry.iloc[0]
    mobile_ship_geom = mobile_ship_geom.segmentize(1000)

    # Jacksonville Port/Channel
    jville_port_geom = gpd.read_file(filePaths[4]).to_crs(icw_sfl_utm).geometry.iloc[0]
    jville_utm = icw_sfl_utm

    # Miami Port
    miami_port_geom = gpd.read_file(filePaths[5]).to_crs(icw_sfl_utm).geometry.iloc[0]
    miami_utm = icw_sfl_utm

    # Miami River
    miami_river_geom = gpd.read_file(filePaths[6]).to_crs(miami_utm).geometry.iloc[0]
    
    # Miami Yacht Club
    miami_yacht_geom = gpd.read_file(filePaths[7]).to_crs(miami_utm).geometry.iloc[0]

    # Fort Lauderdale
    prefix = "Fort Lauderdale Geometry "
    lauderdaleOut = []
    for i, x in enumerate(gpd.read_file(filePaths[8]).to_crs(miami_utm).geometry):
        lauderdaleOut.append((f"{prefix}{i}", x, miami_utm))

    # Export the river names, geometries, and utm zones used by each
    output = [
        ("Mississippi", miss_geom, miss_utm), 
        # ("ICW_SFL", icw_sfl_geom, icw_sfl_utm)    # Makes accuracy worse for now, ignore
        # ("Mobile Ship Channel", mobile_ship_geom, mobile_utm), # Makes accuracy worse for now, ignore
        ("ICW_W", icw_w_geom, icw_w_utm),
        ("Neches", neches_geom, neches_utm),
        ("Houston Ship Channel", h_ship_geom, h_ship_utm),
        ("Sabine-Neches Ship Channel", sn_ship_geom, sn_utm),
        ("Jacksonville Port/Channel", jville_port_geom, jville_utm),
        # ("Miami Port", miami_port_geom, miami_utm),   # Probably need to tune dbscan better here
        ("Miami River", miami_river_geom, miami_utm),
        # ("Miami Yacht Club", miami_yacht_geom, miami_utm)     # Probably need to tune dbscan better here
    ]

    # output += lauderdaleOut # Makes performance worse for now, ignore
    
    return output