import os
import math
import sqlite3
import xml.etree.ElementTree as ET
import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth
    surface using the Haversine formula.

    Parameters:
    lat1, lon1: Coordinates of the first point in decimal degrees.
    lat2, lon2: Coordinates of the second point in decimal degrees.

    Returns:
    Distance in kilometers.
    """
    # Convert decimal degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula components
    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    earth_radius_km = 6371.0
    return earth_radius_km * c


def parse_osm_pois(osm_filepath, poi_tags=None):
    """
    Iteratively parses an XML-based .osm file to extract Points of Interest (POIs).
    Uses ElementTree.iterparse for memory-efficient parsing of large XML structures.

    Parameters:
    osm_filepath: Path to the target input .osm file.
    poi_tags: Set of OSM keys used to classify nodes as POIs.

    Returns:
    A list of dictionaries containing verified POI elements.
    """
    if poi_tags is None:
        # Default common POI identifying tags in OpenStreetMap
        poi_tags = {'amenity', 'shop', 'tourism', 'leisure', 'historic', 'craft', 'office'}

    pois = []

    # Use 'iterparse' to process the XML stream incrementally and save memory
    context = ET.iterparse(osm_filepath, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)

    current_node = None
    tags_accumulator = {}

    for event, elem in context:
        if event == 'start':
            if elem.tag == 'node':
                current_node = {
                    'id': elem.get('id'),
                    'lat': float(elem.get('lat')),
                    'lon': float(elem.get('lon'))
                }
                tags_accumulator = {}
        elif event == 'end':
            if elem.tag == 'tag' and current_node is not None:
                key = elem.get('k')
                val = elem.get('v')
                tags_accumulator[key] = val

            elif elem.tag == 'node' and current_node is not None:
                # Check if the node contains any tag marking it as a POI
                matching_keys = poi_tags.intersection(tags_accumulator.keys())
                if matching_keys:
                    # Determine primary classification type and generic name
                    primary_type = list(matching_keys)[0]
                    poi_category = tags_accumulator[primary_type]
                    poi_name = tags_accumulator.get('name', 'Unnamed POI')

                    current_node['name'] = poi_name
                    current_node['category'] = poi_category
                    current_node['type'] = primary_type

                    pois.append(current_node)

                # Clear memory allocations of the parsed sub-element tree
                current_node = None
                tags_accumulator = {}
                root.clear()

    return pois


def find_nearest_pois(pois, ref_lat, ref_lon, max_distance_km=None, limit=None):
    """
    Filters, computes geographical distances, and sorts POIs relative to a
    provided central reference coordinate.

    Parameters:
    pois: List of parsed POI dictionaries.
    ref_lat, ref_lon: Spatial coordinates of the lookup origin point.
    max_distance_km: Maximum radius threshold filter in kilometers.
    limit: Max number of sorted closest elements to retain.

    Returns:
    Sorted list of POI elements with an added distance key.
    """
    processed_pois = []

    for poi in pois:
        dist = haversine_distance(ref_lat, ref_lon, poi['lat'], poi['lon'])
        if max_distance_km is None or dist <= max_distance_km:
            poi_with_dist = poi.copy()
            poi_with_dist['distance_km'] = dist
            processed_pois.append(poi_with_dist)

    # Sort primarily by calculated geographic distance ascending
    processed_pois.sort(key=lambda x: x['distance_km'])

    if limit is not None:
        processed_pois = processed_pois[:limit]

    return processed_pois


def save_to_npz(pois, output_filepath):
    """
    Saves parsed spatial POI data structures into a compressed NumPy array archive file.

    Parameters:
    pois: List of POI objects containing attributes.
    output_filepath: Target system location for the generated .npz archive.
    """
    ids = np.array([p['id'] for p in pois], dtype=object)
    names = np.array([p['name'] for p in pois], dtype=object)
    categories = np.array([p['category'] for p in pois], dtype=object)
    types = np.array([p['type'] for p in pois], dtype=object)
    lats = np.array([p['lat'] for p in pois], dtype=np.float64)
    lons = np.array([p['lon'] for p in pois], dtype=np.float64)
    distances = np.array([p.get('distance_km', -1.0) for p in pois], dtype=np.float64)

    np.savez_compressed(
        output_filepath,
        id=ids,
        name=names,
        category=categories,
        type=types,
        lat=lats,
        lon=lons,
        distance_km=distances
    )
    print(f"[Success] Exported {len(pois)} POIs to compressed NumPy archive: '{output_filepath}'")


def save_to_sqlite(pois, output_filepath):
    """
    Saves parsed spatial POI data structures into a persistent, queryable SQLite database.

    Parameters:
    pois: List of POI objects containing attributes.
    output_filepath: Target system location for the generated SQLite database file.
    """
    conn = sqlite3.connect(output_filepath)
    cursor = conn.cursor()

    # Establish standard spatial data schema for parsed entities
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pois (
            id TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            type TEXT,
            latitude REAL,
            longitude REAL,
            distance_km REAL
        )
    ''')

    # Map dictionary keys into row insertion tuples
    insert_data = [
        (
            p['id'],
            p['name'],
            p['category'],
            p['type'],
            p['lat'],
            p['lon'],
            p.get('distance_km', None)
        )
        for p in pois
    ]

    cursor.executemany('''
        INSERT OR REPLACE INTO pois (id, name, category, type, latitude, longitude, distance_km)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', insert_data)

    conn.commit()
    conn.close()
    print(f"[Success] Saved {len(pois)} POIs to relational SQLite Database: '{output_filepath}'")


if __name__ == '__main__':
    # ------------------------------------------------------------------------
    # Execution Demo Context
    # ------------------------------------------------------------------------
    # Creating a temporary mock configuration to demonstrate processing workflow.
    # Replace dummy names below with your local .osm environment files.

    mock_osm_filename = "sample_map.osm"
    output_npz_filename = "nearest_pois.npz"
    output_db_filename = "nearest_pois.db"

    # Generate mock map context if file missing for structural test verification
    if not os.path.exists(mock_osm_filename):
        print(f"'{mock_osm_filename}' not found. Generating a basic mock XML file for simulation...")
        mock_xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <osm version="0.6" generator="MockGenerator">
            <node id="100001" lat="17.3850" lon="78.4860">
                <tag k="amenity" v="cafe"/>
                <tag k="name" v="Central Coffee Bar"/>
            </node>
            <node id="100002" lat="17.3890" lon="78.4895">
                <tag k="shop" v="supermarket"/>
                <tag k="name" v="Downtown Grocery Mart"/>
            </node>
            <node id="100003" lat="17.4010" lon="78.4500">
                <tag k="tourism" v="museum"/>
                <tag k="name" v="City History Museum"/>
            </node>
            <node id="100004" lat="17.3700" lon="78.5000">
                <tag k="leisure" v="park"/>
                <tag k="name" v="Greenwood Nature Reserve"/>
            </node>
        </osm>
        """
        with open(mock_osm_filename, "w", encoding="utf-8") as f:
            f.write(mock_xml_content)

    # 1. Configuration Constants (Using Hyderabad center coordinate as lookup baseline)
    REFERENCE_LATITUDE = 17.3850
    REFERENCE_LONGITUDE = 78.4867
    SEARCH_RADIUS_KM = 5.0

    print(f"Starting parsing phase for raw OSM dataset: '{mock_osm_filename}'...")
    # 2. Extract elements leveraging memory-safe stream parsing loop
    all_extracted_pois = parse_osm_pois(mock_osm_filename)
    print(f"Extracted a total of {len(all_extracted_pois)} structural matching POIs from file mapping data.")

    # 3. Apply geographic distance lookup engine pipelines
    nearest_pois = find_nearest_pois(
        pois=all_extracted_pois,
        ref_lat=REFERENCE_LATITUDE,
        ref_lon=REFERENCE_LONGITUDE,
        max_distance_km=SEARCH_RADIUS_KM
    )

    print(f"Filtered down to {len(nearest_pois)} target POIs inside a radius boundary of {SEARCH_RADIUS_KM} km.")

    # 4. Perform serialization writes matching alternative preferences requested
    if nearest_pois:
        save_to_npz(nearest_pois, output_npz_filename)
        save_to_sqlite(nearest_pois, output_db_filename)
    else:
        print("Warning: No proximity options resolved within specifications to export.")
