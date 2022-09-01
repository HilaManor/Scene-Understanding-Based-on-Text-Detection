"""This module handles the 3rd algorithm stage:
finding the physical location of the scene with GoogleMapsAPI."""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~
import googlemaps
import webbrowser
import gmplot
import numpy as np
import os
import itertools

# ~~~~~~~~~~~~~~~~~~~~~~~~ Constants ~~~~~~~~~~~~~~~~~~~~~~~
API_KEY = API KEY
suffixes = ['st', 'ave', 'bvd']
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~


def search_location(streets, others, output_path):
    """Searches for the physical location of the scene, by using the street data and other data
    available. Shows the user a list from which he can choose which option to view.

    :param streets: list of BoxInstances of street signs
    :param others: list of BoxInstances of other, non-street signs. Recommended: shop signs
    :param output_path: path to dump the chosen maps in
    :return: None
    """

    g_maps = googlemaps.Client(key=API_KEY)

    if len(streets):
        print("[+] Trying to match exact geo-location data...")  # Geocoding an address
        if _search_geolocation(g_maps, output_path, streets):
            return
    else:
        print("[X] No geo-data was found!")

    print("[+] Guessing Area...")
    guesses = []
    for o in others:
        complete = o.word.text + ', ' + ' & '.join([s.word.text for s in streets])
        find_results = g_maps.find_place(complete, 'textquery', fields=['geometry/location/lat',
                                                                        'geometry/location/lng',
                                                                        'formatted_address',
                                                                        'name'])["candidates"]
        for res in find_results:
            guesses.append(res)  # will add duplicates, because that means intersection!

    close_guesses = []
    for pair in itertools.combinations(guesses, 2):
        if __measure_dist(pair[0]["geometry"]["location"]["lat"],
                          pair[0]["geometry"]["location"]["lng"],
                          pair[1]["geometry"]["location"]["lat"],
                          pair[1]["geometry"]["location"]["lng"]) < 50:  # less than 50 meters
            close_guesses.append(pair)

    if len(close_guesses):
        print("[+] Found some intersections!")
        lats = [np.mean([pair[0]["geometry"]["location"]["lat"],
                         pair[1]["geometry"]["location"]["lat"]]) for pair in close_guesses]
        lngs = [np.mean([pair[0]["geometry"]["location"]["lng"],
                         pair[1]["geometry"]["location"]["lng"]]) for pair in close_guesses]
        for i in range(len(close_guesses)):
            print("\t%d - %s and %s at (%.5fN, %.5fE)" % (i + 1,
                                                          close_guesses[i][0]["formatted_address"],
                                                          close_guesses[i][0]["formatted_address"],
                                                          lats[i], lngs[i]))

        # let user choose his option
        while True:
            inp = input("Type map NUMBER to open, or S to stop: ").upper()
            if inp == "S":
                return
            elif int(inp) > 0 and int(inp) - 1 <= len(close_guesses):
                # open the map
                i = int(inp) - 1
                __plot_point(lats[i], lngs[i], output_path,
                             close_guesses[i][0]["formatted_address"] + ' and ' +
                             close_guesses[i][0]["formatted_address"])


def _search_geolocation(g_maps, output_path, streets):
    """Search using the Geo-Location functionallity of GoogleAPI.

    This assumes its given data are infact street signs. intersections are marked by '&'

    :param g_maps: Google API's Client instance (api already inside)
    :param output_path: path to dump maps in
    :param streets: BoxInstances list of street signs
    :return: Boolean to stop running this scene or not
    """

    combinations_to_try = __create_suffixes_combinations(streets)
    results = []
    for comb in combinations_to_try:
        geocode_results = g_maps.geocode(components={'route': comb})

        for res in geocode_results:
            if res not in results:
                results.append(res)

    # the best matches are those that google managed to match completely
    best = [res for res in results if not res.get('partial_match')]

    all_matches = True
    stop = False
    if len(best):
        print("[*] Best Matches:")
        lats, lngs = __display_matches(best)

        while True:  # allow user to choose his option
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

    # show matches that Google matched partially (only part of given address)
    if not stop and all_matches and len(results):
        print("[*] Partial Matches")
        lats, lngs = __display_matches(results)
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


def __display_matches(results):
    """Print the list of results

    :param results: list of GoogleAPI returned results
    :return: tuple of latitudes list and longitudes list
    """

    lats = [res['geometry']['location']['lat'] for res in results]
    lngs = [res['geometry']['location']['lng'] for res in results]
    for i in range(len(results)):
        print("\t%d - %s at (%.5fN, %.5fE)" %
              (i + 1, results[i]["formatted_address"], lats[i], lngs[i]))
    return lats, lngs


def __create_suffixes_combinations(streets):
    """Create a combinations of the the streets+suffixes possible, when no suffix is there

    Example:
        for ["Duane", "1st Ave"],
        the result would be ["Duane St & 1st Ave", "Duane Ave & 1st Ave", "Duane Bvd & 1st Ave"]

    :param streets: BoxInstances list of street signs
    :return: all combinations of intersections, already connected by '&' as per Google's notation.
    """

    combs = []
    for s in streets:
        if s.key_street_word:
            combs.append([s.word.text])
        else:
            combs.append([' '.join([*comb]) for
                          comb in itertools.product([s.word.text], suffixes)])
    return [' & '.join([*comb]) for comb in itertools.product(*combs)]


def __plot_point(lat, lng, output_path, result_name):
    """creates, marks, and opens a google map around the given location, under the given name

    :param lat: latitude of location
    :param lng: longitude of location
    :param output_path: path to dump the map in
    :param result_name: name of map file
    :return: None
    """

    g_map = gmplot.GoogleMapPlotter(lat, lng, 16, title=result_name, apikey=API_KEY)
    g_map.marker(lat, lng)
    map_path = os.path.join(output_path, 'best_map - %s.html' % result_name)
    g_map.draw(map_path)
    webbrowser.open('file://' + os.path.realpath(map_path))


def __measure_dist(lat1, lon1, lat2, lon2):
    """Haversine formula"""
    r = 6378.137  # Radius of earth in KM
    d_lat = lat2 * np.pi / 180 - lat1 * np.pi / 180
    d_lon = lon2 * np.pi / 180 - lon1 * np.pi / 180
    a = np.sin(d_lat / 2) * np.sin(d_lat / 2) + np.cos(lat1 * np.pi / 180) * \
        np.cos(lat2 * np.pi / 180) * np.sin(d_lon / 2) * np.sin(d_lon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = r * c
    return d * 1000  # meters
