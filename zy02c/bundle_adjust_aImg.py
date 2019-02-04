#!/usr/bin/env python
# Filename: bundle_adjust_aImg 
"""
introduction: bundle adjust using ASP by providing ground control points

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 03 February, 2019
"""
import sys,os
from optparse import OptionParser

HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.io_function as io_function
import parameters

import basic_src.map_projection as map_projection
import basic_src.RSImage as RSImage

def read_envi_pts(file):
    '''
    read the tie points files of ENVI
    :param file: file path
    :return: base file name, warp file name, a list of tie point (Base Image (x,y), Warp Image (x,y))
    '''

    with open(file,'r') as f_obj:
        lines = f_obj.readlines()
        line_count = len(lines)
        if line_count < 4:
            raise ValueError('error, the envi pts should have at least four lines')

        # get file name
        base_file = os.path.basename(lines[1].split(':')[1].strip())
        warp_file = os.path.basename(lines[2].split(':')[1].strip())

        # get tie points
        tie_points = []
        for idx in range(3,line_count):
            if ';' in lines[idx]:
                continue

            #Base Image (x,y), Warp Image (x,y), based on pixels
            a_tie_point =  [float(item) for item in lines[idx].split()]
            tie_points.append(a_tie_point)

        return base_file, warp_file, tie_points


def convert_gcp_format(ref_image,warp_image,dem_file,pts_files,output=None):
    '''
    convert the ground control points to ASP format
    :param ref_image: the reference image on which selected ground control points
    :param warp_image: the input image need to co-registration or orthorectification
    :param dem_file: the dem file
    :param pts_files: ground control points by using ImageMatchsiftGPU (Envi format)
    :return: the path of new ground control points if successful, otherwise, None
    '''
    # check file
    assert io_function.is_file_exist(ref_image)
    assert io_function.is_file_exist(warp_image)
    assert io_function.is_file_exist(pts_files)
    assert io_function.is_file_exist(dem_file)

    # read pts file
    # tie_points (x1,y1,x2,y2): (x1,y1) and (x2,y2) are column and row on the base and warp image
    base_file, warp_file, tie_points = read_envi_pts(pts_files)

    # check the ref and wrap image
    if base_file not in ref_image:
        raise ValueError('error, the reference image: %s in the pts file is not the same as the input:%s'
                          % (base_file,ref_image))
    if warp_file not in warp_image:
        raise ValueError('error, the warp image: %s in the pts file is not the same as the input:%s'
                            % (warp_file, warp_image))

    # get latitude, longitude from base image

    lon_lat_list = [map_projection.convert_pixel_xy_to_lat_lon(x1,y1,ref_image) for [x1,y1,_,_] in tie_points]

    print(lon_lat_list)
    # get elevation
    # the DEM (SRTM) already convert from geoid (EGM96) based to ellipsoid (WGS 84) based
    ele_list = [RSImage.get_image_location_value(dem_file,lon,lat,'lon_lat_wgs84',1)
                for (lon,lat) in lon_lat_list ]

    # save to file
    x2_list = [x2 for [_,_,x2,_] in tie_points]
    y2_list = [y2 for [_, _, _, y2] in tie_points]
    with open(output,'w') as fw:
        for idx, ((lon,lat),ele,x2,y2) in enumerate(zip(lon_lat_list,ele_list,x2_list, y2_list)):

            fw.writelines('%d %.6lf %.6lf %.6lf %.2lf %.2lf %.2lf %s %.2lf %.2lf %.2lf %.2lf \n'%
                          (idx, lat, lon, ele, 1.0,1.0, 1.0, warp_image, x2, y2, 1.0, 1.0))


def main(options, args):
    ref_image = args[0]
    pts_files = args[1]
    warp_image = args[2]

    dem_file = parameters.get_string_parameters('','dem_datum_wgs84')
    output = os.path.splitext(warp_image)[0]+'.gcp'

    convert_gcp_format(ref_image, warp_image, dem_file, pts_files, output)

    pass

if __name__ == "__main__":
    usage = "usage: %prog [options] ref_image pts_files warp_image"
    parser = OptionParser(usage=usage, version="1.0 2019-2-3")
    parser.description = 'Introduction: bundle adjust for an image'

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    parser.add_option("-p", "--para",
                      action="store", dest="para_file",
                      help="the parameters file")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    ## set parameters files
    if options.para_file is None:
        print('error, no parameters file')
        parser.print_help()
        sys.exit(2)
    else:
        parameters.set_saved_parafile_path(options.para_file)

    main(options, args)
