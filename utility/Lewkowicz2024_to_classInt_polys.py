#!/usr/bin/env python
# Filename: Lewkowicz2024_to_classInt_polys.py 
"""
introduction:  convert the data from of Antoni G. Lewkowicz
(https://nordicana.cen.ulaval.ca/en/publication.php?doi=45888XD-C644C19F4F414D58)
to shapefile only containing class_int

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 November, 2025
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import re

def clean_column_names(header_line):
    '''Removes descriptions after ":" or "(" and strips quotes'''
    cleaned = []
    for col in re.split(r'\t', header_line.strip()):
        # Remove after ":" or "(" and strip quotes
        col_core = re.split(r'[:(]', col)[0].strip().strip('"').strip("'")
        if col_core:  # skip empty
            cleaned.append(col_core)
    return cleaned

def convert_txt_pd_dataframe(input_txt):

    # 2. Clean up header (because your header row is a bit messy)
    # If the first row is not data but a description, skip it:
    with open(input_txt) as f:
        lines = f.readlines()
    header_line = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Identifier"):
            header_line = i
            break

    columns = clean_column_names(lines[header_line])
    # print(columns)
    # sys.exit(0)

    df = pd.read_csv(input_txt, sep=r"\t", skiprows=header_line, engine='python')

    # Only keep as many columns as needed
    df = df.iloc[:, :len(columns)]
    df.columns = columns

    # 3. Create geometry column
    df['geometry'] = df.apply(lambda row: Point(float(row['Longitude']), float(row['Latitude'])), axis=1)

    # 4. Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    return gdf


def main(options, args):
    input_txts = [item for item in args]
    save_path = options.save_path

    gpd_list = [convert_txt_pd_dataframe(txt) for txt in input_txts]

    # Concatenate them
    merged = pd.concat(gpd_list, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=gpd_list[0].crs)

    # add a column, "class_int"
    merged['class_int'] = 1

    # Save merged GeoDataFrame to a new shapefile
    merged.to_file(save_path)



if __name__ == '__main__':
    usage = "usage: %prog [options] input.txt "
    parser = OptionParser(usage=usage, version="1.0 2025-11-19")
    parser.description = 'Introduction: convert data from Lewkowicz 2024 to simpley polygons'

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path", default='lewkowicz_2024.shp',
                      help="the file path for saving the results")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
