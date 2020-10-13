import googlemaps
import webbrowser
import gmplot
import numpy as np
from geopy.geocoders import Nominatim
from geopy.point import Point
import os
import itertools

API_KEY = 'AIzaSyD_4bjN2dduegN3qaO6EcljP1NTiG_akj0'
suffixes = ['st', 'ave', 'bvd']

def search_location(streets, others, output_path):
    gmaps = googlemaps.Client(key=API_KEY)

    if len(streets):
        print("[+] Trying to match exact geo-location data...")  # Geocoding an address
        if _search_geolocation(gmaps, output_path, streets):
            return
    else:
        print("[X] No geo-data was found!")

def _search_geolocation(gmaps, output_path, streets):
    combinations_to_try = __create_suffixes_combinations(streets)
    results = []
    for comb in combinations_to_try:
        geocode_results = gmaps.geocode(components={'route': comb})
        geocode_results2 = gmaps.geocode(comb)

        for res in geocode_results:
            if res not in results:
                results.append(res)
    best = [res for res in results if not res.get('partial_match')]
    all_matches = True
    stop = False
    if len(best):
        print("[*] Best Matches:")
        lats, lngs = __disp_matches(best)

        while True:
            inp = input("Type:\tNUMBER to open map\n\tP to choose from Partial matches"
                        "\n\tG to try to guess\n\tS to stop: ").upper()
            if inp == "S":
                stop = True
                break
            elif inp == "P":
                break
            elif inp == "G":
                all_matches = False
                break
            elif int(inp) > 0 and int(inp) - 1 <= len(best):
                # open the map
                i = int(inp) - 1
                __plot_point(lats[i], lngs[i], output_path, best[i]["formatted_address"])

    if not stop and all_matches and len(results):
        print("[*] Partial Matches")
        lats, lngs = __disp_matches(results)
        while True:
            inp = input("Type:\tNUMBER to open map\n\tG to try to guess\n\tS to stop: ").upper()
            if inp == "S":
                stop = True
                break
            elif inp == "G":
                break
            elif int(inp) > 0 and int(inp) - 1 <= len(results):
                # open the map
                i = int(inp) - 1
                __plot_point(lats[i], lngs[i], output_path, results[i]["formatted_address"])
    return stop


def __disp_matches(results):
    # geolocator = Nominatim(user_agent=API_KEY)
    lats = [res['geometry']['location']['lat'] for res in results]
    lngs = [res['geometry']['location']['lng'] for res in results]
    for i in range(len(results)):
        print("\t%d - %s at (%.5fN, %.5fE)" % (i + 1, results[i]["formatted_address"], lats[i], lngs[i]))
    return lats, lngs


def __create_suffixes_combinations(streets):
    combs = []
    for s in streets:
        if s.key_street_word:
            combs.append([s.word.text])
        else:
            combs.append([' '.join([*comb]) for comb in itertools.product([s.word.text], suffixes)])
    return [' & '.join([*comb]) for comb in itertools.product(*combs)]


def __plot_point(lat, lng, output_path, result_name):
    map = gmplot.GoogleMapPlotter(lat, lng, 16, title=result_name, apikey=API_KEY)
    map.marker(lat, lng)
    map_path = os.path.join(output_path, 'best_map - %s.html' % result_name)
    map.draw(map_path)
    webbrowser.open('file://' + os.path.realpath(map_path))


def __measure_dist(lat1, lon1, lat2, lon2):
    """Haversine formula"""
    r = 6378.137  # Radius of earth in KM
    dLat = lat2 * np.pi / 180 - lat1 * np.pi / 180
    dLon = lon2 * np.pi / 180 - lon1 * np.pi / 180
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = r * c
    return d * 1000  # meters
