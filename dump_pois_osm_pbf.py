#pip install pyrosm geopandas numpy scipy shapely
import numpy as np
import pandas as pd
from pyrosm import OSM
from scipy.spatial import cKDTree
import sqlite3

# Load your local .osm or .pbf file
osm = OSM("southern-zone-260422.osm.pbf")

# Extract custom POIs (e.g., amenities, shops)
tags={
        "place": ["neighbourhood", "suburb"],
        "amenity": ["hospital", "police", "bus_station", "clinic", "fire_station"],
        "railway": ["station"],
        "shop": ["pharmacy"],
        "tourism": ["museum", "attraction"],
        "historic": ["heritage", "fort", "monument", "hotel"]
    }

pois = osm.get_pois(tags)

# Drop rows without valid coordinates
pois = pois.dropna(subset=["lat", "lon"]).copy()

# Build a KD-Tree for fast nearest-neighbor lookup
coords = np.radians(pois[["lat", "lon"]].values)
tree = cKDTree(coords)

# Query nearest neighbor for each point (k=2 to skip self-match at index 0)
distances, indices = tree.query(coords, k=2)
pois["nearest_neighbor_idx"] = indices[:, 1]
# Distance in radians multiplied by Earth's radius (approx 6371000 meters)
pois["nearest_distance_m"] = distances[:, 1] * 6371000
conn = sqlite3.connect("pois_output.db")
# Convert geometry to WKT/string if saving spatial data flat
pois_to_db = pois.drop(columns="geometry", errors="ignore")
pois_to_db.to_sql("pois", conn, if_exists="replace", index=False)
conn.close()
np.savez_compressed(
    "pois_output.npz",
    lat=pois["lat"].values,
    lon=pois["lon"].values,
    name=pois["name"].fillna("Unknown").values,
    nearest_dist=pois["nearest_distance_m"].values,
    nearest_idx=pois["nearest_neighbor_idx"].values,
)
