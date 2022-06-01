#!/usr/bin/env python
# Filename: split_image 
"""
introduction: split a large image to many separate patches

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 15 July, 2017
"""
import sys,os,subprocess
from optparse import OptionParser

from multiprocessing import Pool

def sliding_window(image_width,image_height, patch_w,patch_h,adj_overlay_x=0,adj_overlay_y=0):
    """
    get the subset windows of each patch
    Args:
        image_width: width of input image
        image_height: height of input image
        patch_w: the width of the expected patch
        patch_h: the height of the expected patch
        adj_overlay_x: the extended distance (in pixel of x direction) to adjacent patch, make each patch has overlay with adjacent patch
        adj_overlay_y: the extended distance (in pixel of y direction) to adjacent patch, make each patch has overlay with ad
    Returns: The list of boundary of each patch

    """

    count_x = int(image_width/patch_w)
    count_y = int(image_height/patch_h)

    leftW = int(image_width)%int(patch_w)
    leftH = int(image_height)%int(patch_h)
    if leftW < patch_w/3 and count_x > 0:
        # count_x = count_x - 1
        leftW = patch_w + leftW
    else:
        count_x = count_x + 1
    if leftH < patch_h/3 and count_y > 0:
        # count_y = count_y - 1
        leftH = patch_h + leftH
    else:
        count_y = count_y + 1

    # output split information
    # f_obj = open('split_image_info.txt','w')
    # f_obj.writelines('### This file is created by split_image.py. mosaic_patches.py need it. Do not edit it\n')
    # f_obj.writelines('image_width:%d\n' % image_width)
    # f_obj.writelines('image_height:%d\n' % image_height)
    # f_obj.writelines('expected patch_w:%d\n' % patch_w)
    # f_obj.writelines('expected patch_h:%d\n'%patch_h)
    # f_obj.writelines('adj_overlay_x:%d\n' % adj_overlay_x)
    # f_obj.writelines('adj_overlay_y:%d\n' % adj_overlay_y)

    patch_boundary = []
    for i in range(0,count_x):
        # f_obj.write('column %d:'%i)
        for j in range(0,count_y):
            w = patch_w
            h = patch_h
            if i==count_x -1:
                w = leftW
            if j == count_y - 1:
                h = leftH

            # f_obj.write('%d ' % (i*count_y + j))

            # extend the patch
            xoff = max(i*patch_w - adj_overlay_x,0)  # i*patch_w
            yoff = max(j*patch_h - adj_overlay_y, 0) # j*patch_h
            xsize = min(i*patch_w + w + adj_overlay_x,image_width) - xoff   #w
            ysize = min(j*patch_h + h + adj_overlay_y, image_height) - yoff #h

            new_patch = (xoff,yoff ,xsize, ysize)
            patch_boundary.append(new_patch)

        # f_obj.write('\n')

    # f_obj.close()
    # remove duplicated patches
    patch_boundary_unique = set(patch_boundary)
    if len(patch_boundary_unique) != len(patch_boundary):
        patch_boundary = patch_boundary_unique

    return patch_boundary

def get_one_patch(input, index, patch,output_dir,out_format,extension,pre_name):
    # print information
    print(patch)
    output_path = os.path.join(output_dir, pre_name + '_p_%d' % index + extension)

    args_list = ['gdal_translate', '-of', out_format, '-srcwin', str(patch[0]), str(patch[1]), str(patch[2]),
                 str(patch[3]), input, output_path]
    ps = subprocess.Popen(args_list)
    returncode = ps.wait()
    if os.path.isfile(output_path) is False:
        print('Failed in gdal_translate, return codes: ' + str(returncode))


def split_image(input,output_dir,patch_w=1024,patch_h=1024,adj_overlay_x=0,adj_overlay_y=0,out_format='PNG', pre_name = None, process_num=1):
    """
    split a large image to many separate patches
    Args:
        input: the input big images
        output_dir: the folder path for saving output files
        patch_w: the width of wanted patch
        patch_h: the height of wanted patch

    Returns: True is successful, False otherwise

    """
    if os.path.isfile(input) is False:
        raise IOError("Error: %s file not exist"%input)
    if os.path.isdir(output_dir) is False:
        raise IOError("Error: %s Folder not exist" % output_dir)

    Size_str = os.popen('gdalinfo '+input + ' |grep Size').readlines()
    temp = Size_str[0].split()
    img_witdh = int(temp[2][0:len(temp[2])-1])
    img_height = int(temp[3])

    print('input Width %d  Height %d'%(img_witdh,img_height))
    # print(('patch Width %d  Height %d'%(patch_w,patch_h)))
    patch_boundary = sliding_window(img_witdh,img_height,patch_w,patch_h,adj_overlay_x,adj_overlay_y)

    # remove duplicated patches
    patch_boundary_unique = set(patch_boundary)
    if len(patch_boundary_unique) != len(patch_boundary):
        print('remove %d duplicated patches of images %s'%(len(patch_boundary)-len(patch_boundary_unique),
              os.path.basename(input)))
        patch_boundary = patch_boundary_unique

    index = 0
    if pre_name is None:
        pre_name = os.path.splitext(os.path.basename(input))[0]
    # f_obj = open('split_image_info.txt', 'a+')
    # f_obj.writelines("pre FileName:"+pre_name+'_p_\n')
    # f_obj.close()
    if out_format.upper() == 'PNG':
        extension = '.png'
    elif out_format.upper() == 'GTIFF':  # GTiff
        extension = '.tif'
    elif out_format.upper() == 'JPEG':  # jpg
        extension = '.jpg'
    else:
        raise ValueError("unknow output format:%s" % out_format)

    if process_num==1:
        for patch in patch_boundary:
            # print information
            print(patch)
            output_path = os.path.join(output_dir, pre_name + '_p_%d'%index + extension)

            args_list = ['gdal_translate','-of',out_format,'-srcwin',str(patch[0]),str(patch[1]),str(patch[2]),str(patch[3]), input, output_path]
            ps = subprocess.Popen(args_list)
            returncode = ps.wait()
            if os.path.isfile(output_path) is False:
                raise IOError('Failed in gdal_translate, return codes: ' + str(returncode))
            index = index + 1
    elif process_num > 1:
        parameters_list = [(input, idx, patch, output_dir, out_format, extension, pre_name) for idx, patch in enumerate(patch_boundary)]
        theadPool = Pool(process_num)  # multi processes
        results = theadPool.starmap(get_one_patch, parameters_list)  # need python3
        theadPool.close()
    else:
        raise ValueError('incorrect process number: %s'%(str(process_num)))



def main(options, args):
    if options.s_width is None:
        patch_width = 1024
    else:
        patch_width = int(options.s_width)
    if options.s_height is None:
        patch_height = 1024
    else:
        patch_height = int(options.s_width)

    adj_overlay = 0
    if options.extend is not None:
        adj_overlay = options.extend


    if options.out_dir is None:
        out_dir = "split_save"
    else:
        out_dir = options.out_dir
    if os.path.isdir(out_dir) is False:
        os.makedirs(out_dir)

    out_format = options.out_format

    image_path = args[0]

    split_image(image_path,out_dir,patch_width,patch_height,adj_overlay,adj_overlay,out_format)


    pass

if __name__ == "__main__":
    usage = "usage: %prog [options] image_path"
    parser = OptionParser(usage=usage, version="1.0 2017-7-15")
    parser.description = 'Introduction: split a large image to many separate parts '
    parser.add_option("-W", "--s_width",
                      action="store", dest="s_width",
                      help="the width of wanted patches")
    parser.add_option("-H", "--s_height",
                      action="store", dest="s_height",
                      help="the height of wanted patches")
    parser.add_option("-e", "--extend_dis",type=int,
                      action="store", dest="extend",
                      help="extend distance (in pixel) of the patch to adjacent patch, make patches overlay each other")
    parser.add_option("-o", "--out_dir",
                      action="store", dest="out_dir",
                      help="the folder path for saving output files")
    parser.add_option("-f", "--out_format",
                      action="store", dest="out_format",default='PNG',
                      help="the format of output images")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    # if options.para_file is None:
    #     basic.outputlogMessage('error, parameter file is required')
    #     sys.exit(2)

    main(options, args)
