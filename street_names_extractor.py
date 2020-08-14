import csv
import os

with open('.\\charnet\\config\\StreetNamesVocab.txt', 'a', encoding="utf8") as output_f:
    for file in os.listdir('.\\Data\\names'):
        if file.endswith('.txt'):
            output_f.write('# ' + os.path.basename(file) + '\n')
            with open(os.path.join('.\\Data\\names', file), encoding="utf8") as tsvfile:
                names_reader = csv.DictReader(tsvfile, delimiter='\t',
                                              fieldnames=['geonameid', 'name', 'asciiname',
                                                          'alternatenames', 'latitude',
                                                          'longitude', 'featureclass',
                                                          'featurecode', 'countrycode', 'cc2',
                                                          'admin1code', 'admin2code', 'admin3code',
                                                          'admin4code', 'population', 'elevation',
                                                          'dem',  'timezone', 'modificationdate'])
                output_f.writelines([row['asciiname'] + '\n' for row in names_reader if row['featureclass'] in ['R', 'A']])
        output_f.write('\n')
