import googlemaps
import gmaps
import webbrowser
from gmplot import *
from geopy.geocoders import Nominatim
from geopy.point import Point
import os

def search_geolocation(c_twords):
    lat, lng = None, None
    gmaps = googlemaps.Client(key='AIzaSyD_4bjN2dduegN3qaO6EcljP1NTiG_akj0')
    # Geocoding an address
    geocode_result = gmaps.geocode('c_twords') #TODO:check if needs ' '(due to the way passed)
    lat = geocode_result[0]['geometry']['location']['lat']
    lng = geocode_result[0]['geometry']['location']['lng']

    #printing the location
    geolocator = Nominatim(user_agent="AIzaSyD_4bjN2dduegN3qaO6EcljP1NTiG_akj0")
    location = geolocator.reverse(Point(lat, lng))
    print(location.address)

    #search the place on a map
    #webbrowser.open('https://www.google.com/maps/')

    #open the map
    apikey = 'AIzaSyD_4bjN2dduegN3qaO6EcljP1NTiG_akj0'
    gmap = gmplot.GoogleMapPlotter(lat, lng, 14, apikey=apikey)
    gmap.draw('map.html')
    webbrowser.open('file://' + os.path.realpath(r'map.html'))

    #return lat, lng

    pass
