import json

import osmnx as ox
# pip install osmnx geopandas sqlalchemy
import sqlite3
import pandas as pd
import numpy as np
import os
import asyncio


ox.settings.timeout = 300
ox.settings.requests_timeout = 300
ox.settings.use_cache = True
ox.settings.log_console = True # This helps you see what's happening in real-time
ox.settings.overpass_endpoint = "https://kumi.systems"


cities_lst = [
    "Hyderabad, Telangana, India",
    "Bangalore, Karnataka, India",
    "Chennai, Tamil Nadu, India"
]
tags={
        "place": ["neighbourhood", "suburb"],
        "amenity": ["hospital", "police", "bus_station", "clinic", "fire_station"],
        "railway": ["station"],
        "shop": ["pharmacy"],
        "tourism": ["museum", "attraction"],
        "historic": ["heritage", "fort", "monument", "hotel"]
    }
def city_bbox(plc):
    data = []
    boundary = ox.geocode_to_gdf(plc)
    bounds = boundary.geometry.bounds.iloc[0]
    data.append({
        "name": plc,
        "north": bounds.maxy,
        "south": bounds.miny,
        "east": bounds.maxx,
        "west": bounds.minx
    })
    return data

def map_category(row):
    if row.get("amenity") == "hospital":
        return "hospital"
    if row.get("amenity") == "police":
        return "police"
    if row.get("railway") == "station":
        return "transport"
    if row.get("historic"):
        return "landmark"
    if row.get("tourism"):
        return "tourism"
    return "other"

def save_pois(data_frame,city_name):
    np.savez(
        city_name+".npz",
        names=data_frame["name"].values,
        lats=data_frame["lat"].values,
        lons=data_frame["lon"].values,
        categories=data_frame["category"].values
    )
    print("Saved ",city_name+".npz")
    data_frame.to_csv(city_name+".csv")

    conn = sqlite3.connect(city_name+".db")
    data_frame.to_sql("poi", conn, if_exists="replace", index=False)
    conn.close()
    print("Saved ",city_name+".db")

async def city_poi_db(place):
    # place = "Hyderabad, Telangana, India"
    city_name = place.split(',')[0]
    file_city = city_name+'.npz'
    city_bound = city_bbox(place)
    json.dump(city_bound,json_file)

    if os.path.exists(file_city):
       return None

    loop = asyncio.get_event_loop()
    gdf = await loop.run_in_executor(None, lambda: ox.features_from_place(place, tags))
    gdf = gdf[gdf['name'].notna()]

    gdf['lat'] = gdf.geometry.centroid.y
    gdf['lon'] = gdf.geometry.centroid.x

    gdf["category"] = gdf.apply(map_category, axis=1)

    df = gdf[["name", "lat", "lon", "category"]].copy()

    df = df.drop_duplicates(subset=["name", "lat", "lon"])

    print(f"Total POIs: {len(df)}")
    save_pois(df,city_name)

    print(df)

city_json_file = "city_bbox.json"
json_file = open(city_json_file, "w")
for city_detl in cities_lst:
    asyncio.run(city_poi_db(city_detl))
json_file.close()
# city_poi_db("Hyderabad")