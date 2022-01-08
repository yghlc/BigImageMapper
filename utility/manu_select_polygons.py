#!/usr/bin/env python
# Filename: manu_select_polygons.py 
"""
introduction: manually select polygons

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 07 January, 2022
"""

import os,sys
from optparse import OptionParser

import pandas as pd

deeplabforRS =  os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)
import vector_gpd
import basic_src.io_function as io_function
import basic_src.basic as basic

def select_polygons_by_ids_in_excel(in_shp, table_path, save_path,id_name='id'):
    if os.path.isfile(save_path):
        basic.outputlogMessage('warning, %s exists, skip'%save_path)
        return True

    pol_ids = vector_gpd.read_attribute_values_list(in_shp,id_name)
    ids_table = pd.read_excel(table_path)
    manu_sel_ids = ids_table[id_name].tolist()        # miou of different model

    ## check if there are duplicated ones
    print('manu_sel_ids length', len(manu_sel_ids))
    unique_ids = list(set(manu_sel_ids))
    print('unique manu_sel_ids length', len(unique_ids))
    manu_sel_ids_copy = manu_sel_ids.copy()
    for item in unique_ids:
        manu_sel_ids_copy.remove(item)
    print('duplicated ids:',str(manu_sel_ids_copy))

    # find match index
    select_idx = [ pol_ids.index(sel_id) for sel_id in manu_sel_ids ]

    return vector_gpd.save_shapefile_subset_as(select_idx,in_shp,save_path)


def main(options, args):
    input_shp = args[0]
    # manual section, could a files in a folder or ids in a table
    manual_sel = args[1]
    io_function.is_file_exist(input_shp)

    save_path = options.save_path
    if save_path is None:
        save_path = io_function.get_name_by_adding_tail(input_shp,'manuSelect')

    if manual_sel.endswith('.xlsx'):
        select_polygons_by_ids_in_excel(input_shp,manual_sel,save_path)
    elif os.path.isdir(manual_sel):
        # not support yet.
        pass
    else:
        print('unknown input of manual selection')



if __name__ == '__main__':
    usage = "usage: %prog [options] polygon_shp folder_path/excel.xlsx "
    parser = OptionParser(usage=usage, version="1.0 2021-3-6")
    parser.description = 'Introduction: select mapping polygons manually (manually saved ids or files)  '

    parser.add_option("", "--save_path",
                      action="store", dest="save_path",
                      help="the path for saving file")

    (options, args) = parser.parse_args()
    # print(options.create_mosaic)

    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)