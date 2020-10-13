import googlemaps
import webbrowser
from gmplot import *
import numpy as np
from geopy.geocoders import Nominatim
from geopy.point import Point
import os
import itertools

API_KEY = 'AIzaSyD_4bjN2dduegN3qaO6EcljP1NTiG_akj0'
suffixes = ['st', 'ave', 'bvd']

def search_geolocation(streets, others):
    gmaps = googlemaps.Client(key='AIzaSyD_4bjN2dduegN3qaO6EcljP1NTiG_akj0')
    # Geocoding an address
    addr='duanereade'
    comps={'route':'duane'}
    geocode_results = gmaps.geocode(address=addr, components=comps)
    gmaps.places_autocomplete('duane', components={'country': ['US', 'GB', 'CA', 'AU']})
    for result in geocode_results:
        lat = result['geometry']['location']['lat']
        lng = result['geometry']['location']['lng']

        #printing the location
        geolocator = Nominatim(user_agent="AIzaSyD_4bjN2dduegN3qaO6EcljP1NTiG_akj0")
        location = geolocator.reverse(Point(lat, lng))
        print(location.address)

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
