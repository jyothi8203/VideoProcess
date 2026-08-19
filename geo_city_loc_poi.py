import osmnx as ox
# from geopy.geocoders import Nominatim
import asyncio
import winsdk.windows.devices.geolocation as wdg
# import gpsd,
import json
import serial.tools.list_ports
import pynmea2
import numpy as np
from sklearn.neighbors import BallTree
import math

def load_city_npz(place):
    city_nm = place.split(',')[0]
    data = np.load(city_nm+".npz", allow_pickle=True)

    names = data["names"]
    lats = data["lats"]
    lons = data["lons"]
    categories = data["categories"]

    # Convert to radians for haversine
    coords = np.column_stack((lats, lons))
    coords_rad = np.radians(coords)

    tree = BallTree(coords_rad, metric="haversine")

    threshold = 100
    weights = {
        "hospital": 1.0,
        "transport": 0.9,
        "police": 0.95,
        "food": 0.5,
        "tourism": 0.6,
        "other": 0.3
    }
    return coords_rad, tree, threshold, weights, names, categories

def nearest_poi(lat, lon, tree_q, nms, ctgrs):
    query = np.radians([[lat, lon]])
    dist, idx = tree_q.query(query, k=1)

    i = idx[0][0]
    d = dist[0][0] * 6371000  # convert to meters

    return nms[i], ctgrs[i], d


def score_poi(category, dist):
    w = weights.get(category, 0.3)
    return w / (dist + 1e-6)

def get_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # meters Haversine formula.

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_city_from_gps(lat, lon):
    geolocator = Nominatim(user_agent="app")
    location = geolocator.reverse(f"{lat}, {lon}")

    address = location.raw["address"]

    city = address.get("city") or address.get("town") or address.get("village")
    state = address.get("state")
    country = address.get("country")

    return city, state, country

def process_latlon(city,plat,plng,lat,lng,tree,nms,ctgrs):
    # load_city_npz(city)
    threshold = 10
    movement = get_distance(plat, plng, lat, lng)
    location_text = ""
    new_loctn = False
    if movement >= threshold:
        new_loctn = True
        name, category, dist = nearest_poi(lat, lng,tree,nms,ctgrs)
        print(f"Near {name} ({category}) | Distance: {dist:.1f} m")
        location_text = name
        plat, plng = lat, lng
        print(location_text)
    return location_text,new_loctn #if distance threshold not satisfies plat, plng is not updated

async def get_coordinates():
    locator = wdg.Geolocator()
    try:
        pos = await locator.get_geoposition_async()
        return pos.coordinate.latitude, pos.coordinate.longitude
    except Exception as e:
        print(f"Error accessing sensor: {e}")
        return None

def find_city(lat, lon):
    with open("city_bbox.json") as f:
        cities = json.load(f)

    for c in cities:
        if (cities[c]["south"] <= lat <= cities[c]["north"] and
            cities[c]["west"] <= lon <= cities[c]["east"]):
            return cities[c]["name"]
    return "Unknown"

def get_ports():
    ports = list(serial.tools.list_ports.comports())
    devc = None
    for p in ports:
        # print(p.description,p.device)
        if "USB Serial Device" in p.description or "u-blox" in p.description:
            devc = p.device
    return devc

def get_gps_coordinates(ntwk, baudrate):
    ser = serial.Serial(ntwk, baudrate, timeout=1)
    print(f"Connected to {ntwk}")

    lat, lng = 0, 0
    gpgga_count = 0
    while True:
        line = ser.readline().decode('ascii', errors='replace')
        if gpgga_count ==30:
            return 0, 0
        if line.startswith('$GPGGA'):
            # print(line.strip())
            gpgga_count += 1
            try:
                msg = pynmea2.parse(line)
                if msg.gps_qual > 0:
                    print("Latitude:", msg.latitude)
                    print("Longitude:", msg.longitude)
                    lat, lng = msg.latitude, msg.longitude
                    return lat, lng
            except Exception as e:
                print(e)


def get_latlon(net_work,baudrate = 0):
    lt, ln = 0, 0
    if net_work is not None and "USB Serial Device" in net_work or "COM3" in net_work:
        baudrate = 9600
        lt, ln = get_gps_coordinates(net_work, baudrate)
        if lt == 0 and ln == 0:
            net_work = "online"
    if net_work is None or net_work == "online":
        baudrate = 0
        result = asyncio.run(get_coordinates())
        if result is not None:
            lt, ln = result
            net_work = "online"



    return lt,ln,baudrate


def init_city():
    try:
        ntwk = get_ports()
        lat,lon,baudrate = get_latlon(ntwk)
        if lat != 0 and lon != 0:
            print(f"Latitude: {lat}, Longitude: {lon}")
            city = find_city(lat, lon)

        # lat,lng = gpsd_curr_loc_latlng()
            place = city
            return city,(ntwk,baudrate,lat,lon)
        else:
            return "INDOOR",(ntwk,baudrate,0,0)
    except PermissionError:
        print("Working Offline, respective location city database not loaded.")
        # print("Permission Denied: Please enable 'Allow apps to access your location' in Windows Settings.")
        return None
