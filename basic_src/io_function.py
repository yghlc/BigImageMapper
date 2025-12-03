#!/usr/bin/env python
# Filename: io_function.py
"""
introduction: support I/O operation for normal files

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 May, 2016
"""

import os,shutil
import basic_src.basic as basic
import subprocess

from datetime import datetime

import re

import json
import urllib

def mkdir(path):
    """
    create a folder
    Args:
        path: the folder name

    Returns:True if successful, False otherwise.
    Notes:  if IOError occurs, it will exit the program
    """
    path = path.strip()
    path = path.rstrip("\\")
    isexists = os.path.exists(path)
    if not isexists:
        try:
            os.makedirs(path)
            basic.outputlogMessage(path + ' Create Success')
            return True
        except IOError:
            basic.outputlogMessage('creating %s failed'%path)
            assert False
    else:
        print(path + '  already exist')
        return False


def delete_file_or_dir(path):
    """
    remove a file or folder
    Args:
        path: the name of file or folder

    Returns: True if successful, False otherwise
    Notes: if IOError occurs or path not exist, it will exit the program
    """
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        else:
            basic.outputlogMessage('%s not exist'%path)
            assert False
    except IOError:
        basic.outputlogMessage('remove file or dir failed : ' + str(IOError))
        assert False

    return True

def is_file_exist_subfolder(folder, file_name, bsub_folder=True):
    """
    Determine whether the file is in the specified folder or its subfolders.

    Args:
        folder (str): The root folder to search.
        file_name (str): The file name to search for.
        bsub_folder (bool): If True, search subfolders as well; otherwise, check only the given folder.

    Returns:
        str: The absolute path of the file if found.
        bool: False if the file is not found.
    """
    # Check if the folder exists
    if not os.path.isdir(folder):
        raise IOError(f"Input error: directory '{folder}' is invalid.")

    # Check only the current folder
    if not bsub_folder:
        file_path = os.path.join(folder, file_name)
        return file_path if os.path.isfile(file_path) else False

    # Search the folder and its subfolders
    for root, _, files in os.walk(folder):
        if file_name in files:
            return os.path.join(root, file_name)

    # File not found
    return False


def is_file_exist(file_path):
    """
    determine whether the file_path is a exist file
    Args:
        file_path: the file path

    Returns:True if file exist, False otherwise

    """
    if os.path.isfile(file_path):
        return True
    else:
        basic.outputlogMessage("File : %s not exist"%os.path.abspath(file_path))
        raise IOError("File : %s not exist"%os.path.abspath(file_path))
        # return False

def is_folder_exist(folder_path):
    """
    determine whether the folder_path is a exist folder
    :param folder_path: folder path
    :return:True if folder exist, False otherwise
    """
    if len(folder_path) < 1:
        basic.outputlogMessage('error: The input folder path is empty')
        return False
    if os.path.isdir(folder_path):
        return True
    else:
        basic.outputlogMessage("Folder : %s not exist"%os.path.abspath(folder_path))
        raise IOError("Folder : %s not exist"%os.path.abspath(folder_path))
        # return False


def os_list_folder_dir(top_dir):
    if not os.path.isdir(top_dir):
        basic.outputlogMessage('the input string is not a dir, input string: %s'%top_dir)
        return False

    sub_folders = []
    for file in sorted(os.listdir(top_dir)):
        file_path = os.path.abspath(os.path.join(top_dir, file))
        if os.path.isfile(file_path):
            continue
        elif os.path.isdir(file_path):
            sub_folders.append(file_path)
    if len(sub_folders) < 1:
        basic.outputlogMessage('There is no sub folder in %s'%top_dir)
        return False
    return sub_folders

def os_list_folder_files(top_dir):
    if not os.path.isdir(top_dir):
        basic.outputlogMessage('the input string is not a dir, input string: %s'%top_dir)
        return False

    list_files = []
    for file in sorted(os.listdir(top_dir)):
        file_path = os.path.abspath(os.path.join(top_dir, file))
        if os.path.isfile(file_path):
            list_files.append(file_path)


    if len(list_files) < 1:
        basic.outputlogMessage('There is no file in %s'%top_dir)
        return False
    return list_files


def get_index_from_filename(filename):
    # get the index for a polygon in the original file: such as 18268 in "all_composited-image_18268.tif"
    idx = int(re.findall(r"_([0-9]+)\.", filename)[0])
    return idx

def get_file_list_by_ext(ext, folder, bsub_folder):
    """
    Args:
        ext: Extension name(s) of files to find, can be a string for a single extension or a list for multiple extensions,
             e.g., '.tif' or ['.tif', '.TIF']
        folder: The directory to explore.
        bsub_folder: True for searching subdirectories, False for searching the current directory only.

    Returns:
        A list with the absolute paths of matching files, e.g., ['/user/data/1.tif', '/user/data/2.tif']

    Notes:
        If input is invalid, it will raise an appropriate error.
    """
    # Ensure extensions are in a list
    if isinstance(ext, str):
        extensions = [ext.lower()]
    elif isinstance(ext, list):
        extensions = [e.lower() for e in ext]
    else:
        raise ValueError("Input extension type must be a string or a list of strings.")

    # Check if the folder exists and is a directory
    if not os.path.isdir(folder):
        raise IOError(f"Input error: directory '{folder}' is invalid.")

    # Ensure bsub_folder is a boolean
    if not isinstance(bsub_folder, bool):
        raise ValueError("Input error: bsub_folder must be a boolean value.")

    # Use os.walk if searching subdirectories, else only list the current directory
    files = []
    if bsub_folder:
        for root, _, filenames in os.walk(folder):
            for file in filenames:
                if os.path.splitext(file)[1].lower() in extensions:
                    files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in extensions:
                files.append(file_path)

    return files

def get_file_list_by_pattern_ls(folder,pattern):
    """
    get the file list by file pattern
    :param folder: /home/hlc
    :param pattern: eg. '*imgAug*.ini'
    :return: the file list
    """
    # get the path of all the porosity profile
    file_pattern = os.path.join(folder, pattern)
    # basic.outputlogMessage('find pattern for: '+ file_pattern)

    ## on ITSC service, the following code is not working, output empty, which is strange (9 Dec 2019)
    proc = subprocess.Popen('ls ' + file_pattern, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    profiles, err = proc.communicate()
    file_list = profiles.split()
    # to string, not byte
    file_list = [item.decode() for item in file_list]
    return file_list

def get_file_list_by_pattern(folder,pattern):
    """
    get the file list by file pattern
    :param folder: /home/hlc
    :param pattern: eg. '*imgAug*.ini'
    :return: the file list
    """
    # get the path of all the porosity profile
    file_pattern = os.path.join(folder, pattern)
    # basic.outputlogMessage('find pattern for: '+ file_pattern)

    ## on ITSC service, the following code is not working, output empty, which is strange (9 Dec 2019)
    # proc = subprocess.Popen('ls ' + file_pattern, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # profiles, err = proc.communicate()
    # file_list = profiles.split()
    # # to string, not byte
    # file_list = [item.decode() for item in file_list]

    import glob
    file_list = glob.glob(file_pattern)

    return file_list

def get_free_disk_space_GB(dir):
    total, used, free = shutil.disk_usage(dir)  # output in bytes
    return free/(1000*1000*1000)    # convert to GB

def get_absolute_path(path):
    return os.path.abspath(path)

def get_file_modified_time(path):
    return datetime.fromtimestamp(os.path.getmtime(path))

def get_url_file_size(url_path):
    # curl -I url_path
    req = urllib.request.Request(url_path,method='HEAD')
    f = urllib.request.urlopen(req)
    if f.status == 200:
        size = f.headers['Content-Length']      # in Bytes
        return int(size)

    basic.outputlogMessage('error, get size of %s failed'%url_path)
    return False

def get_file_size_bytes(path):
    return os.path.getsize(path)        # return the file size in bytes

def get_file_path_new_home_folder(in_path):
    if in_path is None:
        return None
    # try to change the home folder path if the file does not exist
    if os.path.isfile(in_path) or os.path.isdir(in_path):
        return in_path
    else:
        tmp_str = in_path.split('/')
        new_tmp = '~/' + '/'.join(tmp_str[3:])
        new_path = os.path.expanduser(new_tmp)
        basic.outputlogMessage('Warning, change to a new path under the new home folder: %s'%new_path)
        return new_path

def get_name_no_ext(file_path):
    """
    get file name without extension
    Args:
        file_path: exist file name

    Returns: a new name if successfull
    Notes: if input error, it will exit program

    """
    # get file name without extension
    filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    return filename_no_ext

def get_name_by_adding_tail(basename,tail):
    """
    create a new file name by add a tail to a exist file name
    Args:
        basename: exist file name
        tail: the tail name

    Returns: a new name if successfull
    Notes: if input error, it will exit program

    """
    text = os.path.splitext(basename)
    if len(text)<2:
        basic.outputlogMessage('ERROR: incorrect input file name: %s'%basename)
        assert False
    return text[0]+'_'+tail+text[1]


def copy_file_to_dst(file_path, dst_name, overwrite=False, b_verbose=True):
    """
    copy file to a destination file
    Args:
        file_path: the copied file
        dst_name: destination file name

    Returns: True if successful or already exist, False otherwise.
    Notes:  if IOError occurs, it will exit the program
    """
    if os.path.isfile(dst_name) and overwrite is False:
        basic.outputlogMessage("%s already exist, skip copy file"%dst_name)
        return True

    if file_path==dst_name:
        basic.outputlogMessage('warning: shutil.SameFileError')
        return True

    try:
        shutil.copy(file_path,dst_name)
    # except shutil.SameFileError:
    #     basic.outputlogMessage('warning: shutil.SameFileError')
    #     pass
    except IOError:
        raise IOError('copy file failed: '+ file_path)



    if not os.path.isfile(dst_name):
        basic.outputlogMessage('copy file failed, from %s to %s.'%(file_path,dst_name))
        return False
    else:
        if b_verbose:
            basic.outputlogMessage('copy file success: '+ file_path)
        return True


def move_file_to_dst(file_path, dst_name,overwrite=False, b_verbose=True):
    """
    move file to a destination file
    Args:
        file_path: the moved file
        dst_name: destination file name

    Returns: True if successful, False otherwise.
    Notes:  if IOError occurs, it will exit the program

    """
    if os.path.isfile(dst_name) and overwrite is False:
        basic.outputlogMessage("%s already exist, skip move file"%dst_name)
        return True
    if os.path.isfile(dst_name) and overwrite is True:
        delete_file_or_dir(dst_name)

    try:
        shutil.move(file_path,dst_name)
    except IOError:
        raise IOError('move file failed: '+ file_path)

    if os.path.isfile(dst_name):
        if b_verbose:
            basic.outputlogMessage('move file success: ' + file_path)
        return True
    elif os.path.isdir(dst_name):
        if b_verbose:
            basic.outputlogMessage('move folder success: ' + file_path)
        return True
    else:
        basic.outputlogMessage('move file or folder failed, from %s to %s.'%(file_path,dst_name))
        return False

def movefiletodir(file_path, dir_name,overwrite=False, b_verbose=True):
    """
    move file to a destination folder
    Args:
        file_path: the moved file
        dir_name: destination folder name

    Returns: True if successful or already exist, False otherwise.
    Notes:  if IOError occurs, it will exit the program

    """
    dst_name =  os.path.join(dir_name,os.path.split(file_path)[1])
    return move_file_to_dst(file_path,dst_name, overwrite=overwrite, b_verbose=b_verbose)

def copyfiletodir(file_path, dir_name,overwrite=False, b_verbose=True):
    """
    copy file to a destination folder
    Args:
        file_path: the copied file
        dir_name: destination folder name

    Returns: True if successful or already exist, False otherwise.
    Notes:  if IOError occurs, it will exit the program

    """
    dst_name =  os.path.join(dir_name,os.path.split(file_path)[1])
    return copy_file_to_dst(file_path,dst_name,overwrite=overwrite, b_verbose=b_verbose)

def unzip_file(file_path, work_dir):
    '''
    unpack a *.zip package,
    :param file_path:
    :param work_dir:
    :return:  the absolute path of a folder which contains the decompressed files
    '''
    if os.path.isdir(work_dir) is False:
        raise IOError('dir %s not exist'%os.path.abspath(work_dir))

    if file_path.endswith('.zip') is False:
        raise ValueError('input %s do not end with .zip')

    file_basename = os.path.splitext(os.path.basename(file_path))[0]

    # decompression file and remove it
    dst_folder = os.path.join(work_dir, file_basename)
    if os.path.isdir(dst_folder):
        # on Mac, .DS_Store count one file, ignore all hidden files
        tmp_list = [ item for item in os.listdir(dst_folder) if item.startswith('.') is False ]
        if len(tmp_list) > 0:
            basic.outputlogMessage('%s exists and is not empty, skip unpacking' % dst_folder)
            return dst_folder
    else:
        mkdir(dst_folder)
    # CommandString = 'tar -xvf  ' + file_tar + ' -C ' + dst_folder
    args_list = ['unzip', file_path, '-d', dst_folder]
    # (status, result) = basic.exec_command_string(CommandString)
    returncode = basic.exec_command_args_list(args_list)
    # print(returncode)
    if returncode != 0:
        return False

    return dst_folder

def unpack_tar_gz_file(file_path,work_dir):
    '''
    unpack a *.tar.gz package, the same to decompress_gz_file (has a bug)
    :param file_path:
    :param work_dir:
    :return:  the absolute path of a folder which contains the decompressed files
    '''
    if os.path.isdir(work_dir) is False:
        raise IOError('dir %s not exist'%os.path.abspath(work_dir))

    if file_path.endswith('.tar.gz') is False:
        raise ValueError('input %s do not end with .tar.gz')

    file_basename = os.path.basename(file_path)[:-7]

    # decompression file and remove it
    dst_folder = os.path.join(work_dir,file_basename)
    if os.path.isdir(dst_folder):
        files = [ item for item in os.listdir(dst_folder) if not item.startswith('.') ] # on Mac, .DS_Store count one file; ignore hide files
        if len(files) > 0:
            basic.outputlogMessage('%s exists and is not empty, skip unpacking'%dst_folder)
            return dst_folder
    else:
        mkdir(dst_folder)
    # CommandString = 'tar -xvf  ' + file_tar + ' -C ' + dst_folder
    args_list = ['tar', '-zxvf', file_path,'-C',dst_folder]
    # (status, result) = basic.exec_command_string(CommandString)
    returncode = basic.exec_command_args_list(args_list)
    # print(returncode)
    if returncode != 0:
        return False

    return dst_folder


def decompress_gz_file(file_path,work_dir,bkeepmidfile):
    """
    decompress a compressed file with gz extension (has a bug if end with *.*.tar.gz)
    Args:
        file_path:the path of gz file
        bkeepmidfile: indicate whether keep the middle file(eg *.tar file)

    Returns:the absolute path of a folder which contains the decompressed files

    """
    if os.path.isdir(work_dir) is False:
        basic.outputlogMessage('dir %s not exist'%os.path.abspath(work_dir))
        return False
    file_basename = os.path.basename(file_path).split('.')[0]
    # file_tar = os.path.join(os.path.abspath(work_dir), file_basename + ".tar")
    file_tar = os.path.join(os.path.dirname(file_path), file_basename + ".tar")


    # decompression file and keep it
    # CommandString = 'gzip -dk ' + landsatfile
    # change commond line like below, bucause gzip version on cry01 do not have the -k option  by hlc 2015.12.26
    # CommandString = 'gzip -dc ' + file_path + ' > ' + file_tar
    args_list = ['gzip','-dk',file_path]
    # (status, result) = basic.exec_command_string(CommandString)
    # if status != 0:
    #     basic.outputlogMessage(result)
    #     return False
    if os.path.isfile(file_tar):
        basic.outputlogMessage('%s already exist')
    else:
        basic.exec_command_args_list(args_list)

    # decompression file and remove it
    dst_folder = os.path.join(os.path.abspath(work_dir),file_basename)
    mkdir(dst_folder)
    # CommandString = 'tar -xvf  ' + file_tar + ' -C ' + dst_folder
    args_list = ['tar', '-xvf', file_tar,'-C',dst_folder]
    # (status, result) = basic.exec_command_string(CommandString)
    basic.exec_command_args_list(args_list)
    # if status != 0:
    #     basic.outputlogMessage(result)
    #     return False
    if bkeepmidfile is False:
        os.remove(file_tar)
    return dst_folder


def keep_only_used_files_in_list(output_list_file,old_image_list_txt,used_images_txt,syslog):
    if is_file_exist(old_image_list_txt) is False:
        return False
    if is_file_exist(used_images_txt) is False:
        return False

    output_list_obj = open(output_list_file,"w")

    image_list_txt_obj = open(old_image_list_txt,'r')
    image_list = image_list_txt_obj.readlines()
    if len(image_list)< 1:
        syslog.outputlogMessage('%s open failed or do not contains file paths'%os.path.abspath(old_image_list_txt))
        return False
    used_images_txt_obj = open(used_images_txt,'r')
    used_images = used_images_txt_obj.readlines()
    if len(used_images)<1:
        syslog.outputlogMessage('%s open failed or do not contains file paths'%os.path.abspath(used_images_txt))
        return False

    for image_file in image_list:
        file_id = os.path.basename(image_file).split('.')[0]
        for used_file in used_images:
            used_file = os.path.splitext(os.path.basename(used_file))[0]
            used_file = used_file.split('_')[0]
            if file_id == used_file:
                output_list_obj.writelines(image_file)
                break

    image_list_txt_obj.close()
    used_images_txt_obj.close()
    output_list_obj.close()

def delete_shape_file(input):
    arg1 = os.path.splitext(input)[0]
    exts = ['.shx', '.shp','.prj','.dbf','.cpg']
    for ext in exts:
        file_path = arg1 + ext
        if os.path.isfile(file_path):
            delete_file_or_dir(file_path)

    return True

def copy_shape_file(input, output):

    assert is_file_exist(input)

    arg1 = os.path.splitext(input)[0]
    arg2 = os.path.splitext(output)[0]
    # arg_list = ['cp_shapefile', arg1, arg2]
    # return basic.exec_command_args_list_one_file(arg_list, output)

    copy_file_to_dst(arg1+'.shx', arg2 + '.shx', overwrite=True)
    copy_file_to_dst(arg1+'.shp', arg2 + '.shp', overwrite=True)
    copy_file_to_dst(arg1+'.prj', arg2 + '.prj', overwrite=True)
    copy_file_to_dst(arg1+'.dbf', arg2 + '.dbf', overwrite=True)

    basic.outputlogMessage('finish copying %s to %s'%(input,output))

    return True

def save_list_to_txt(file_name, save_list):
    with open(file_name, 'w') as f_obj:
        for item in save_list:
            f_obj.writelines(item + '\n')

def read_list_from_txt(file_name):
    with open(file_name,'r') as f_obj:
        lines = f_obj.readlines()
        lines = [item.strip() for item in lines]
        return lines

def save_text_to_file(file_name, text):
    with open(file_name, 'w') as f_obj:
        f_obj.write(text + '\n')

def append_text_to_file(file_name, text):
    with open(file_name, 'a') as f_obj:
        f_obj.write(text + '\n')

def save_dict_to_txt_json(file_name, save_dict):

    # check key is string, int, float, bool or None,
    strKey_dict = {}
    for key in save_dict.keys():
        # print(type(key))
        if type(key) not in [str, int, float, bool, None]:
            strKey_dict[str(key)] = save_dict[key]
        else:
            strKey_dict[key] = save_dict[key]

    # ,indent=2 makes the output more readable
    json_data = json.dumps(strKey_dict,indent=2)
    with open(file_name, "w") as f_obj:
        f_obj.write(json_data)

def read_dict_from_txt_json(file_path):
    if os.path.getsize(file_path) == 0:
        return None
    with open(file_path) as f_obj:
        data = json.load(f_obj)
        return data

def get_path_from_txt_list_index(txt_name,input=''):
    '''
    get the a line (path or file pattern) for a txt files, the index is in the file name
    Args:
        txt_name:
        input:

    Returns:

    '''

    # current folder path
    cwd_path = os.getcwd()
    if os.path.isfile(txt_name) is False:
        # if the txt does not exist, then cd ../.., find the file againn
        txt_name = os.path.join(os.path.dirname(os.path.dirname(cwd_path)), txt_name)
    with open(txt_name, 'r') as f_obj:
        lines = f_obj.readlines()
        lines = [item.strip() for item in lines]

    # find the index
    folder = os.path.basename(cwd_path)
    import re
    I_idx_str = re.findall(r'I\d+', folder)
    if len(I_idx_str) == 1:
        index = int(I_idx_str[0][1:])
    else:
        # try to find the image idx from file name
        file_name = os.path.basename(input)
        I_idx_str = re.findall(r'I\d+', file_name)
        if len(I_idx_str) == 1:
            index = int(I_idx_str[0][1:])
        else:
            raise ValueError('Cannot find the I* which represents image index')

    val_path = lines[index]

    return val_path

def check_file_or_dir_is_old(file_folder, time_hour_thr):
    # if not exists, then return False
    if os.path.isfile(file_folder) is False and os.path.isdir(file_folder) is False:
        return False
    now = datetime.now()
    m_time = datetime.fromtimestamp(os.path.getmtime(file_folder))
    print('%s modified time: %s'%(file_folder,str(m_time)))
    diff_time = now - m_time
    diff_time_hour = diff_time.total_seconds()/3600
    if diff_time_hour > time_hour_thr:
        return True
    else:
        return False

def write_metadata(key, value, filename=None):
    if isinstance(key,list):
        keys = key
    else:
        keys = [key]
    if isinstance(value, list):
        values = value
    else:
        values = [value]

    if len(keys) != len(values):
        raise ValueError('the number of keys (%d) and values (%d) is different'%(len(keys), len(values)))

    if filename is None:
        filename = 'metadata.txt'
    # with open(filename,'a') as f_obj:
    #     f_obj.writelines(str(key)+': '+str(value) + '\n')
    if os.path.isfile(filename):
        meta_dict = read_dict_from_txt_json(filename)
    else:
        meta_dict = {}
    for k, v in zip(keys, values):
        meta_dict[k] = v
    save_dict_to_txt_json(filename,meta_dict)

def create_soft_link(src, dst):
    # Resolve src if it is a symlink
    if os.path.islink(src):
        # Follow the symlink to its final target
        abs_src = os.path.realpath(src)
        # print(f'Source {src} is a symlink, resolving to {abs_src}')
    else:
        abs_src = os.path.abspath(src)
    if os.path.exists(dst):
        print(f'Warning: {dst} exists, skipping')
        return
    if os.path.islink(dst):
        raise IOError(f'The soft link {dst} exists, but the dst not existing')
    os.symlink(abs_src, dst)

if __name__=='__main__':
    pass
