#!/usr/bin/env python
# Filename: apply_masked_image 
"""
introduction: Apply mask file *_udm.tif to images. Set cloud cover as non data.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 29 August, 2018
"""


import sys,os
from optparse import OptionParser

import subprocess

import rasterio
import numpy as np

def get_mask_file_name(input_tif):
    folder = os.path.dirname(input_tif)
    file_name = os.path.basename(input_tif)
    if file_name.find('SR') > 0:
        # if the input is surface reflectance
        p_str = file_name.split('_')
        p_str = p_str[:-1]
        return os.path.join(folder,"_".join(p_str)+"_DN_udm.tif")
    else:
        name_noext = os.path.splitext(file_name)[0]
        return os.path.join(folder,name_noext+"_DN_udm.tif")

def get_output_name(input_tif):
    folder = os.path.dirname(input_tif)
    file_name = os.path.basename(input_tif)
    name_noext = os.path.splitext(file_name)[0]
    return os.path.join(folder, name_noext + "_maskC.tif")


def get_planet_cloud_mask_file(mask_file):

    if os.path.isfile(mask_file) is False:
        raise IOError("%s not exist"%mask_file)

    # Load the mask file
    with rasterio.open(mask_file) as src:
        if src.dtypes[0] != rasterio.uint8:
            raise ValueError("Only support unit8, but input is %s"%str(src.dtypes[0]))
        mask_band = src.read(1)

    # get cloud mask
    mask_shape = mask_band.shape
    mask_unpackbits = np.unpackbits(mask_band.reshape(mask_shape[0],mask_shape[1],1),axis=2)

    # the second bit is cloud cover (from back)
    could_mask = mask_unpackbits[:,:,-2:-1]

    could_mask = could_mask.reshape(mask_shape[0],mask_shape[1])

    # Invert 0 and 1 in a binary array
    could_mask = 1 - could_mask

    # print(could_mask.shape)

    text = os.path.splitext(mask_file)
    output_cloud_mask = text[0] + '_' + "cloud" + text[1]

    ## Create the file
    kwargs = src.meta
    with rasterio.open(output_cloud_mask, 'w', **kwargs) as dst:
        dst.write_band(1, could_mask.astype(rasterio.uint8))

    print("save clould mask to %s"%output_cloud_mask)

    return output_cloud_mask


def apply_could_mask(input_tif,mask_file,output):

    ## get could mask
    if os.path.isfile(mask_file) is False:
        raise IOError("%s not exist" % mask_file)

    # Load the mask file
    with rasterio.open(mask_file) as src:
        if src.dtypes[0] != rasterio.uint8:
            raise ValueError("Only support unit8, but input is %s" % str(src.dtypes[0]))
        mask_band = src.read(1)

        # get cloud mask
    mask_shape = mask_band.shape
    mask_unpackbits = np.unpackbits(mask_band.reshape(mask_shape[0], mask_shape[1], 1), axis=2)

    # the second bit is cloud cover (from back)
    could_mask = mask_unpackbits[:, :, -2:-1]

    could_mask = could_mask.reshape(mask_shape[0], mask_shape[1])
    # Invert 0 and 1 in a binary array
    could_mask = 1 - could_mask

    ## apply the mask

    # Load the mask file
    with rasterio.open(input_tif) as input_src:
        # read all bands
        input_bands = input_src.read()

    ## save the file
    kwargs = input_src.meta
    with rasterio.open(output, 'w', **kwargs) as dst:
        # dst.write_band(1, could_mask)
        for idx,band in enumerate(input_bands):
            # apply mask
            band = band*could_mask
            # save
            dst.write_band(idx+1, band)


    pass


def main(options, args):
    input_tif = args[0]

    if options.output is not None:
        output = options.output
    else:
        output = get_output_name(input_tif)

    mask_file = get_mask_file_name(input_tif)

    apply_could_mask(input_tif,mask_file,output)

    # cloud_mask_file = get_planet_cloud_mask_file(mask_file) #mask_file

    # if os.path.isfile(cloud_mask_file):
        # args_list = ['gdal_calc.py','-A',cloud_mask_file,'-B',input_tif,'--debug',
        #              '--overwrite','--outfile='+output,'--calc="A*B"']
        # ps = subprocess.Popen(args_list)
        # returncode = ps.wait()
        # if os.path.isfile(output):
        #     print('masked and saved to %s'%output)

        #

    pass



if __name__ == "__main__":
    usage = "usage: %prog [options] input_image "
    parser = OptionParser(usage=usage, version="1.0 2018-7-15")
    parser.description = 'Introduction: apply mask file (*_udm.tif) to the planet images'

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")
    parser.add_option("-m", "--mask",
                      action="store", dest="mask",
                      help="the path of mask file")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)