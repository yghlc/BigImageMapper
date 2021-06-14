#!/usr/bin/env python
# Filename: timeTools 
"""
introduction: functions and classes to handle datetime

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 29 December, 2020
"""

import os,sys
import basic_src.basic as basic

from datetime import datetime
from dateutil.parser import parse
import re

def get_yeardate_yyyymmdd(in_string, pattern='[0-9]{8}',format='%Y%m%d'):
    '''
    get datetime object from a filename (or string)
    Args:
        in_string: input string containing yyyymmdd

    Returns: datetime

    '''


    found_strs = re.findall(pattern,in_string)
    if len(found_strs) < 1:
        basic.outputlogMessage('Warning, cannot found yyyymmdd string in %s'%(in_string))
        return None
    # for str in found_strs:
    #     print(str)
    #     date_obj = parse(str,yearfirst=True)
    #     print(date_obj)
    if pattern.endswith('_'):  # if pattern is: '[0-9]{8}_'
        found_strs = [item[:-1] for item in found_strs]     # remove "_"

    datetime_list = []
    for str in found_strs:
        try:
            date_obj = datetime.strptime(str, format)
            # print(date_obj)
            datetime_list.append(date_obj)
        except ValueError as e:
            # print(e)
            pass

    # print(datetime_list)
    if len(datetime_list) != 1:
        basic.outputlogMessage('Warning, found %d yyyymmdd string in %s' % (len(datetime_list), in_string))
        return None
    return datetime_list[0]

def get_now_time_str(fromat="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(fromat)

def str2date(date_str,format = '%Y%m%d'):
    date_obj = datetime.strptime(date_str, format)
    return date_obj

def date2str(date, format='%Y%m%d'):
    return date.strftime(format)

def datetime2str(date_time, format='%Y%m%d_%H%M%S'):
    return date_time.strftime(format)

def diff_yeardate(in_date1, in_date2):
    '''
    calculate the difference between two date
    Args:
        in_date1:
        in_date2:

    Returns: absolute of days

    '''
    diff = in_date1 - in_date2
    # print(diff)
    diff_days = diff.days + diff.seconds / (3600*24)
    # print(diff_days)
    return abs(diff_days)

def group_files_yearmonthDay(demTif_list, diff_days=30):
    '''
    groups files based on date information in the filename
    :param demTif_list:
    :return:
    '''
    file_groups = {}
    for tif in demTif_list:
        tif_name = os.path.basename(tif)
        yeardate =  get_yeardate_yyyymmdd(tif_name)  # time year is at the begining
        if yeardate is None:
            # try again by adding '_' in the pattern
            print('waning, try again to get datetime for %s using [0-9]{8}_'%tif_name)
            yeardate = get_yeardate_yyyymmdd(tif_name,pattern='[0-9]{8}_')
            if yeardate is None:
                raise ValueError('get date info from %s failed'%tif)

        b_assgined = False
        for time in file_groups.keys():
            if diff_yeardate(time,yeardate) <= diff_days:
                file_groups[time].append(tif)
                b_assgined = True
                break
        if b_assgined is False:
            file_groups[yeardate] = [tif]

    return file_groups

def test():
    # out = get_yeardate_yyyymmdd('20170301_10300100655B5A00_1030010066B4AA00.tif')
    out = get_yeardate_yyyymmdd('20201230_10300100655B5A00_1030010066B4AA00.tif')
    print(out)
    diffdays = diff_yeardate(out,datetime.now())
    print(diffdays)


if __name__=='__main__':
    test()
    pass