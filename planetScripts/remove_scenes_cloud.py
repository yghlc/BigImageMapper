#!/usr/bin/env python
# Filename: remove_scenes_cloud.py 
"""
introduction: remove planet scenes based on cloud cover (greater than a threshold)

Should run in the same folder of the xlsx file

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 05 November, 2020
"""

import sys,os
from optparse import OptionParser

HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import pandas as pd

import basic_src.io_function as io_function
import basic_src.basic as basic

def main(options, args):
    scene_list_xlsx = args[0]
    scene_pd = pd.read_excel(scene_list_xlsx)

    cloud_cover = options.cloud_cover
    scene_count, col = scene_pd.shape

    rm_sences = []
    incomplete_scenes = []
    # from this table, get cloud cover and scene path
    for idx, row in scene_pd.iterrows():
        # cc = row.
        if row['cloud_cover'] > cloud_cover and row['cloud_cover'] <= 100:
            rm_sences.append(row['folder'])
        elif row['cloud_cover'] > 100:
            incomplete_scenes.append(row['folder'])

    basic.outputlogMessage('There are %d out of %d scenes with high cloud cover (%lf) in the table (%s)'%
                           (len(rm_sences), scene_count,cloud_cover,scene_list_xlsx))

    if len(incomplete_scenes) > 0:
        basic.outputlogMessage('Warning, there are %d out of %d incomplete scenes in the table (%s), please consider handling this' %
                               (len(incomplete_scenes), scene_count, scene_list_xlsx))
    if len(rm_sences) < 1:
        return False

    scene_dir = os.path.dirname(rm_sences[0])
    if os.path.isdir(scene_dir) is False:
        basic.outputlogMessage('%s does not exist, please check the current working folder or update %s using get_scene_list_xlsx.sh'%(scene_dir,scene_list_xlsx))
        return False

    bak_dir = os.path.join(scene_dir, 'scenes_high_cloud_cover')
    # move the folder and geojson to backup folder
    io_function.mkdir(bak_dir)
    for item in rm_sences:
        # move folder
        io_function.movefiletodir(item,bak_dir)

        geojson = item + '.geojson'
        # move geojson
        if os.path.isfile(geojson):
            io_function.movefiletodir(geojson, bak_dir)

    pass


if __name__ == "__main__":

    usage = "usage: %prog [options] image_list_xlsx"
    parser = OptionParser(usage=usage, version="1.0 2020-11-05")
    parser.description = 'Introduction: remove planet scenes based on cloud cover (greater than a threshold) '

    parser.add_option("-c", "--cloud_cover",
                      action="store", dest="cloud_cover", type=float, default=30,
                      help="the could cover threshold, images with cloud cover greater than this threshold will be removed")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)



