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

        #open the map
        apikey = 'AIzaSyD_4bjN2dduegN3qaO6EcljP1NTiG_akj0'
        gmap = gmplot.GoogleMapPlotter(lat, lng, 14, apikey=apikey)
        gmap.draw('map.html')
        webbrowser.open('file://' + os.path.realpath(r'map.html'))

def __measure_dist(lat1, lon1, lat2, lon2):
    """Haversine formula"""
    r = 6378.137  # Radius of earth in KM
    dLat = lat2 * np.pi / 180 - lat1 * np.pi / 180
    dLon = lon2 * np.pi / 180 - lon1 * np.pi / 180
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = r * c
    return d * 1000  # meters
