#!/usr/bin/env python
# Filename: vector_features 
"""
introduction: shapefile operation based on pyshp

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 28 October, 2016
"""

from optparse import OptionParser

import shapely
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.ops import cascaded_union

import basic_src.basic as basic
import basic_src.io_function as io_function
import basic_src.map_projection as map_projection
import os,sys
import numpy

import  parameters

#pyshp library
import shapefile

# some changes in 2.x the new changes are incompatible with previous versions (1.x)
# many place need to change, such as "shapefile.Writer()"
if shapefile.__version__ >= '2.0.0':
    raise ValueError('Current do not support pyshp version 2 or above, please use pyshp version 1.2.12')

import random

#rasterstats
from rasterstats import zonal_stats

# minimum_rotated_rectangle need shapely >= 1.6
if shapely.__version__ < '1.6.0':
    raise ImportError('Please upgrade your shapely installation to v1.6.0 or newer!')

class shape_opeation(object):

    def __init__(self):
        # self.__qgis_app = None # qgis environment

        pass


    def __del__(self):
        # if self.__qgis_app is not None:
        #     basic.outputlogMessage("Release QGIS resource")
        #     self.__qgis_app.exitQgis()
        pass

    def __find_field_index(self,all_fields, field_name):
        """
        inquire the index of specific field_name in the all fields
        :param all_fields: all fields
        :param field_name: specific field name
        :return: index of specific field name if successful, False Otherwise
        """

        field_name_index = -1
        field_len = len(all_fields)
        for t_index in range(0,field_len):
            t_field = all_fields[t_index]
            if isinstance(t_field, tuple):
                # t_index += 1  occur once
                continue
            if field_name == t_field[0]:
                field_name_index = t_index - 1  # first field not show in records
                break
        if field_name_index < 0:
            basic.outputlogMessage('error, attribute %s not found in the fields' % field_name)
            return False
        return field_name_index

    def has_field(self,input_shp,field_name):
        """
        inquires whether the shape file contains the specific field given by the field name
        :param input_shp: shape file path
        :param field_name: the name of the specific field
        :return: True if exist, False otherwise
        """
        if io_function.is_file_exist(input_shp) is False:
            return False
        try:
            org_obj = shapefile.Reader(input_shp)
        except IOError:
            basic.outputlogMessage(str(IOError))
            return False
        all_fields = org_obj.fields
        field_len = len(all_fields)

        for t_index in range(0, field_len):
            t_field = all_fields[t_index]
            if isinstance(t_field, tuple):
                # t_index += 1  occur once
                continue
            if field_name == t_field[0]:
                return True  #find the specific field of the given name

        return False



    def get_polygon_shape_info(self, input_shp, out_box, bupdate=False):
        """
        get Oriented minimum bounding box for a polygon shapefile,
        and update the shape information based on oriented minimum bounding box to
        the input shape file
        :param input_shp: input polygon shape file
        :param out_box: output Oriented minimum bounding box shape file
        :param bupdate: indicate whether update the original input shapefile
        :return:True is successful, False Otherwise
        """
        if io_function.is_file_exist(input_shp) is False:
            return False

        # using QGIS causes troubles in many places, such as in the Singularity container, through tmate connection
        # so use minimum_rotated_rectangle in shapely instead
        # hlc Jan 11 2019

        # args_list = ['qgis_function.py',input_shp,out_box]
        # if basic.exec_command_args_list_one_file(args_list, out_box) is False:
        #     basic.outputlogMessage('get polygon shape information of %s failed' % input_shp)
        #     return False
        # else:
        #     basic.outputlogMessage('get polygon shape information of %s completed, output file is %s' % (input_shp, out_box))
        #     return True

        try:
            org_obj = shapefile.Reader(input_shp)
        except:
            basic.outputlogMessage(str(IOError))
            return False

        # Create a new shapefile in memory
        w = shapefile.Writer()
        w.shapeType = org_obj.shapeType

        org_records = org_obj.records()
        if (len(org_records) < 1):
            basic.outputlogMessage('error, no record in shape file ')
            return False

        # Copy over the geometry without any changes
        shapes_list = org_obj.shapes()

        polygon_shapely = []
        for idx, temp in enumerate(shapes_list):
            polygon_shapely.append(shape_from_pyshp_to_shapely(temp, shape_index=idx,shp_path=input_shp))

        # minimum_rotated_rectangle of the polygons (i.e., orientedminimumboundingbox in QGIS)
        polygon_min_r_rectangles = [item.minimum_rotated_rectangle for item in polygon_shapely]

        # rectangle info:
        AREA_list = []
        PERIMETER_list =[]
        WIDTH_list = []
        HEIGHT_list = []
        for rectangle in polygon_min_r_rectangles:
            AREA_list.append(rectangle.area)
            PERIMETER_list.append(rectangle.length)
            points = list(rectangle.boundary.coords)
            point1 = Point(points[0])
            point2 = Point(points[1])
            point3 = Point(points[2])
            width = point1.distance(point2)
            height = point2.distance(point3)
            WIDTH_list.append(width)
            HEIGHT_list.append(height)

        # add the info to shapefile
        # attr_list = [field_name, 'N', 24, 6]
        # w.fields.append(attr_list)
        w.field('rAREA', fieldType="N", size="24",decimal=6)
        w.field('rPERIMETER', fieldType="N", size="24", decimal=6)
        w.field('rWIDTH', fieldType="N", size="24", decimal=6)
        w.field('rHEIGHT', fieldType="N", size="24", decimal=6)

        # save the buffer area (polygon)
        pyshp_polygons = [shape_from_shapely_to_pyshp(shapely_polygon, keep_holes=True) for shapely_polygon in
                          polygon_min_r_rectangles]

        # org_records = org_obj.records()
        for i in range(0, len(pyshp_polygons)):
            w._shapes.append(pyshp_polygons[i])
            rec = [AREA_list[i],PERIMETER_list[i],WIDTH_list[i],HEIGHT_list[i]]
            w.records.append(rec)
        #
        # copy prj file
        org_prj = os.path.splitext(input_shp)[0] + ".prj"
        out_prj = os.path.splitext(out_box)[0] + ".prj"
        io_function.copy_file_to_dst(org_prj, out_prj, overwrite=True)
        #
        # overwrite original file
        w.save(out_box)

        return True

    def get_shapes_count(self,input_shp):
        """
        get the number of shape in the shape file
        :param input_shp: path of shape file
        :return: the number of shape (polygon, points), Fasle Otherwise
        """
        if io_function.is_file_exist(input_shp) is False:
            return False
        try:
            org_obj = shapefile.Reader(input_shp)
        except IOError:
            basic.outputlogMessage(str(IOError))
            return False
        return len(org_obj.shapes())

    def get_shapes(self,input_shp):
        """
        get shape (geometries) in the shape file
        :param input_shp: path of shape file
        :return: shapes (polygon, points), Fasle Otherwise
        """
        if io_function.is_file_exist(input_shp) is False:
            return False
        try:
            org_obj = shapefile.Reader(input_shp)
        except IOError:
            basic.outputlogMessage(str(IOError))
            return False
        return org_obj.shapes()


    def add_fields_shape(self,ori_shp,new_shp,output_shp):
        """
        add fields from another shapefile(merge the fields of two shape files), the two shape files should have the same number of features
        :param ori_shp: the path of original shape file which will be added new field
        :param new_shp: the shape file contains new fields
        :output_shp: saved shape file
        :return:True if successful, False otherwise

        """
        # Read in our existing shapefile
        if io_function.is_file_exist(ori_shp) is False or io_function.is_file_exist(new_shp) is False:
            return False
        try:
            org_obj = shapefile.Reader(ori_shp)
            new_obj = shapefile.Reader(new_shp)
        except IOError:
            basic.outputlogMessage(str(IOError))
            return False

        if len(org_obj.shapes()) != len(new_obj.shapes()):
            raise ValueError("error: the input two shape file do not have the same number of features")

        if org_obj.shapeType != new_obj.shapeType:
            raise ValueError("error: the input two shape file have different shapeType")
            # return False

        # Create a new shapefile in memory
        w = shapefile.Writer()
        w.shapeType = org_obj.shapeType

        # Copy over the existing fields
        w.fields = list(org_obj.fields)
        for t_field in list(new_obj.fields):
            if isinstance(t_field,tuple):
                continue
            w.fields.append(t_field)


        # Add our new field using the pyshp API
        # w.field("KINSELLA", "C", "40")

        # # We'll create a counter in this example
        # # to give us sample data to add to the records
        # # so we know the field is working correctly.
        # i = 1
        #
        # Loop through each record, add a column.  We'll
        # insert our sample data but you could also just
        # insert a blank string or NULL DATA number
        # as a place holder
        org_records = org_obj.records()
        new_records = new_obj.records()
        for i in range(0,len(org_records)):
            rec = org_records[i]
            for value in new_records[i]:
                rec.append(value)

            # Add the modified record to the new shapefile
            w.records.append(rec)

        # Copy over the geometry without any changes
        w._shapes.extend(org_obj.shapes())

        # copy prj file
        org_prj = os.path.splitext(ori_shp)[0]+".prj"
        out_prj = os.path.splitext(output_shp)[0]+".prj"
        io_function.copy_file_to_dst(org_prj,out_prj,overwrite=True)

        # Save as a new shapefile (or write over the old one)
        w.save(output_shp)

        pass

    def add_multi_field_records_to_shapefile(self,shape_file,record_value_list,field_name_list):
        """
        add multiple to field and their records to shape file
        Args:
            shape_file: shape file
            record_value_list: the values list in 2D
            field_name_list: the 1D list of name

        Returns: True is successful, Flase otherwise

        """
        for idx,field_name in enumerate(field_name_list):
            values = [item[idx] for item in record_value_list ]
            self.add_one_field_records_to_shapefile(shape_file,values,field_name)


    def add_one_field_records_to_shapefile(self,shape_file,record_value,field_name,field_type=None):
        """
        add one field and records to shape file (add one column to attributes table)
        :param shape_file: shape file path
        :param record_value: a list contians the records value
        :param field_name: field name (the column title)
        :param field_type:  field type, eg. float, int, string,  can read from type(record_value[0])
        :return:True is successful, Flase otherwise
        """
        if io_function.is_file_exist(shape_file) is False:
            return False
        if isinstance(record_value,list) is False:
            basic.outputlogMessage('record_value must be list')
        if None in record_value:
            # raise ValueError("None in record_value, please check the projection or extent of shapefile and raster file. "
            #                  "The projection should be the same. The shape file should be within the raster extent. ")
            basic.outputlogMessage("warning: None in record_value, it will be replaced as -9999")
            record_value = [-9999 if item is None else item for item in record_value ]

        records_count = len(record_value)
        if(records_count<1):
            basic.outputlogMessage('error, no input records')
            return False

        try:
            org_obj = shapefile.Reader(shape_file)
        except :
            basic.outputlogMessage(str(IOError))
            return False
        if len(org_obj.shapes()) != records_count:
            raise ValueError("error: the input field_name_value do not have the same number of features")
            # return False

        # Create a new shapefile in memory
        w = shapefile.Writer()
        w.shapeType = org_obj.shapeType

        # Copy over the existing fields
        w.fields = list(org_obj.fields)

        #check whether the field name already exist
        exist_index = -1
        for i in range(0,len(w.fields)):
            if w.fields[i][0] == field_name:
                exist_index = i - 1   # -1 means ignore the 'DeletionFlag' (first column)
                basic.outputlogMessage('warning, field name: %s already in table %d (first column is 0) column, '
                                       'this will replace the original value'%(field_name,exist_index))
                break
        if exist_index >= 0:
            pass
        else:
            #create a new fields at the last
            first_record = record_value[0]
            if isinstance(first_record,float):
                attr_list = [field_name, 'N',24,6]
            elif isinstance(first_record, int):
                attr_list = [field_name, 'N', 24, 0]
            elif isinstance(first_record, str):
                attr_list = [field_name, 'C', 255, 0]  # limit to 255  # ubyte format requires 0 <= number <= 255
            else:
                basic.outputlogMessage('error, unsupport data type')
                return False
            # attr_list = [field_name, 'N', 24, 0]
            w.fields.append(attr_list)


        org_records = org_obj.records()
        if exist_index >= 0:
            for i in range(0, len(org_records)):
                rec = org_records[i]
                rec[exist_index] = record_value[i]
                # Add the modified record to the new shapefile
                w.records.append(rec)
        else:
            for i in range(0, len(org_records)):
                rec = org_records[i]
                rec.append(record_value[i])
                # Add the modified record to the new shapefile
                w.records.append(rec)

        # check field whose type is str and convert bytes to str (fill NULL) if applicable
        str_filed_index = []
        for i in range(0, len(w.fields)):
            if w.fields[i][1] == 'C':
                str_filed_index.append(i - 1)  # -1 means ignore the 'DeletionFlag' (first column)
        if len(str_filed_index)>0:
            for i in range(0,len(w.records)):
                for j in str_filed_index:
                    if isinstance(w.records[i][j],bytes):
                        w.records[i][j] = 'NULL'

        # Copy over the geometry without any changes
        w._shapes.extend(org_obj.shapes())

        # copy prj file
        # org_prj = os.path.splitext(ori_shp)[0] + ".prj"
        # out_prj = os.path.splitext(output_shp)[0] + ".prj"
        # io_function.copy_file_to_dst(org_prj, out_prj,overwrite=True)

        # overwrite original file
        w.save(shape_file)

        return True



    def add_fields_to_shapefile(self,shape_file,field_name_value,prefix):
        """
        add one new field and its records to shape file, currently, the records value must be digital number
        :param shape_file: the shape file for adding new fields
        :param field_name_value: a list contains records value of this fields
        :param prefix: first part of field name, eg. "prefix_key"
        :return: True if successful, False otherwise
        """
        if io_function.is_file_exist(shape_file) is False:
            return False
        if isinstance(field_name_value,list) is False:
            basic.outputlogMessage('field_name_value must be list')

        records_count = len(field_name_value)
        if(records_count<1):
            basic.outputlogMessage('error, no input records')
            return False

        # try:
        #     org_obj = shapefile.Reader(shape_file)
        # except :
        #     basic.outputlogMessage(str(IOError))
        #     return False
        # if len(org_obj.shapes()) != records_count:
        #     basic.outputlogMessage("error: the input field_name_value do not have the same number of features")
        #     return False
        #
        # # Create a new shapefile in memory
        # w = shapefile.Writer()
        # w.shapeType = org_obj.shapeType
        #
        # # Copy over the existing fields
        # w.fields = list(org_obj.fields)
        # first_record = field_name_value[0]
        # for t_key in first_record.keys():
        #     # if isinstance(t_field, tuple):
        #     #     continue
        #     attr_list = [prefix +'_'+ t_key, 'N',24,5] #prefix + t_key  #["name", type,max_length, showed_length] only accept digital number now
        #     w.fields.append(attr_list)
        #
        # org_records = org_obj.records()
        #
        # for i in range(0, len(org_records)):
        #     rec = org_records[i]
        #     dict_in=field_name_value[i]
        #
        #     for t_key in dict_in.keys():
        #         rec.append(dict_in.get(t_key))
        #
        #     # Add the modified record to the new shapefile
        #     w.records.append(rec)
        #
        # # Copy over the geometry without any changes
        # w._shapes.extend(org_obj.shapes())
        #
        # # copy prj file
        # # org_prj = os.path.splitext(ori_shp)[0] + ".prj"
        # # out_prj = os.path.splitext(output_shp)[0] + ".prj"
        # # io_function.copy_file_to_dst(org_prj, out_prj,overwrite=True)
        #
        # # overwrite original file
        # w.save(shape_file)


        first_record = field_name_value[0]
        for t_key in first_record.keys():
            # if isinstance(t_field, tuple):
            #     continue
            new_field_name = prefix +'_'+ t_key
            new_record_value = [ dict_in.get(t_key) for dict_in in field_name_value]
            self.add_one_field_records_to_shapefile(shape_file,new_record_value,new_field_name)

        #

        return True


    def add_fields_from_raster(self,ori_shp,raster_file,field_name,band=1,stats_list = None,all_touched=False):
        """
        get field value from raster file by using "rasterstats"

        """
        if io_function.is_file_exist(ori_shp) is False or io_function.is_file_exist(raster_file) is False:
            return False
        # stats_list = ['min', 'max', 'mean', 'count','median','std']
        if stats_list is None:
            stats_list = ['mean', 'std']

        # check projection of vector and raster
        # shp_wkt = map_projection.get_raster_or_vector_srs_info_wkt(ori_shp)
        # raster_wkt = map_projection.get_raster_or_vector_srs_info_wkt(raster_file)
        shp_proj4 = map_projection.get_raster_or_vector_srs_info_proj4(ori_shp)
        raster_proj4 = map_projection.get_raster_or_vector_srs_info_proj4(raster_file)
        # if shp_wkt != raster_wkt:
        if shp_proj4 != raster_proj4:
            raise ValueError('erros: %s and %s do not have the same projection. '
                             'Their WKT info are \n %s \n and \n%s'%(ori_shp,raster_file,shp_proj4,raster_proj4))

        # band = 1
        stats = zonal_stats(ori_shp,raster_file,band = band,stats = stats_list,all_touched=all_touched)
        #test
        # for tp in stats:
        #     print("mean:",tp["mean"],"std:",tp["std"])

        if self.add_fields_to_shapefile(ori_shp, stats, field_name) is False:
            basic.outputlogMessage('add fields to shape file failed')

        return True

    def remove_shape_baseon_field_value(self,shape_file,out_shp,class_field_name,threashold,smaller=True):
        """
        remove features from shapefile based on the field value,
        if smaller is true, then the value smaller than threashold will be removed
        if smaller is False, then the value greater than threashold will be remove
        :param shape_file: input shape file
        :param out_shp: saved shape file
        :param class_field_name: the name of class field, such as area
        :param threashold: threashold value
        :param smaller:  if smaller is true, then the value smaller than threashold will be removed,
        :return: True if successful, False otherwise
        """
        if io_function.is_file_exist(shape_file) is False:
            return False

        try:
            org_obj = shapefile.Reader(shape_file)
        except:
            basic.outputlogMessage(str(IOError))
            return False

        # Create a new shapefile in memory
        w = shapefile.Writer()
        w.shapeType = org_obj.shapeType

        org_records = org_obj.records()
        if (len(org_records) < 1):
            basic.outputlogMessage('error, no record in shape file ')
            return False

        # Copy over the geometry without any changes
        w.fields = list(org_obj.fields)
        field_index = self.__find_field_index(w.fields, class_field_name)
        if field_index is False:
            return False
        shapes_list = org_obj.shapes()
        i = 0
        removed_count = 0
        if smaller is True:
            for i in range(0,len(shapes_list)):
                rec = org_records[i]
                if rec[field_index] < threashold:    # remove the record which is smaller than threashold
                    removed_count = removed_count +1
                    continue
                w._shapes.append(shapes_list[i])
                rec = org_records[i]
                w.records.append(rec)
        else:
            for i in range(0, len(shapes_list)):
                rec = org_records[i]
                if rec[field_index] >  threashold:  # remove the record which is greater than threashold
                    removed_count = removed_count +1
                    continue
                w._shapes.append(shapes_list[i])
                rec = org_records[i]
                w.records.append(rec)

        basic.outputlogMessage('Remove polygons based on %s, count: %d, remain %d ones' % (class_field_name,removed_count, len(w.records)))
        # w._shapes.extend(org_obj.shapes())

        # copy prj file
        org_prj = os.path.splitext(shape_file)[0] + ".prj"
        out_prj = os.path.splitext(out_shp)[0] + ".prj"
        io_function.copy_file_to_dst(org_prj, out_prj,overwrite=True)

        w.save(out_shp)
        return True

    def get_k_fold_of_polygons(self,shape_file,out_shp,k_value,sep_field_name=None,shuffle=False):
        """
        split polygons to k-fold,
        Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
        similar to sklearn.model_selection.KFold, but apply to polygons in a shapefile
        Args:
        Args:
            shape_file: input
            out_shp: output, will output k shapefiles in the same directory with the basename of "out_shp"
            k_value: e.g., 5, 10 or others
            sep_field_name: field name storing different class
            shuffle: shuffle before splitting, usually it is true

        Returns: True if successful, False Otherwise

        """
        if io_function.is_file_exist(shape_file) is False:
            return False

        try:
            org_obj = shapefile.Reader(shape_file)
        except:
            basic.outputlogMessage(str(IOError))
            return False

        org_records = org_obj.records()
        if (len(org_records) < 1):
            basic.outputlogMessage('error, no record in shape file ')
            return False

        shapes_list = org_obj.shapes()
        org_shape_count = len(shapes_list)

        # get index of polygons
        if sep_field_name is not None:
            field_index = self.__find_field_index(org_obj.fields, sep_field_name)
            if field_index is False:
                return False

            all_shape_index = list(range(0, org_shape_count))

            # get class_int from this field
            class_int_list = [rec[field_index] for rec in org_records]
            class_unique_list = list(set(class_int_list)) # unique class id

            # for each class, choose a subset of them
            all_shape_index_per_class = []
            for class_id in class_unique_list:
                tmp_class_shp_idx = []
                for shp_indx in all_shape_index:
                    if org_records[shp_indx][field_index] == class_id:
                        tmp_class_shp_idx.append(shp_indx)

                all_shape_index_per_class.append(tmp_class_shp_idx)

        else:
            # select from the whole polgyons
            all_shape_index = list(range(0,org_shape_count))


        # shuffle samples
        if shuffle:
            from random import shuffle
            if sep_field_name is not None:
                for shape_index_a_class in all_shape_index_per_class:
                    shuffle(shape_index_a_class)
            else:
                shuffle(all_shape_index)

        # k-folder subsample
        for k_idx in range(0,k_value):
            # Create a new shapefile in memory
            w = shapefile.Writer()
            w.shapeType = org_obj.shapeType

            # Copy over the geometry without any changes
            w.fields = list(org_obj.fields)

            if sep_field_name is not None:
                # selection in each class
                select_shape_index_per_class = []
                for class_id, shape_idx_a_class in enumerate(all_shape_index_per_class):

                    # # don't subsample the polygons of class_id=0, hlc 2019-Jan 3
                    # if class_id == 0:
                    #     select_shape_index_per_class.append(shape_idx_a_class)
                    #     continue

                    split_arrays = numpy.array_split(shape_idx_a_class,k_value)
                    tmp_selected_index = []
                    for arr_idx, tmp_array in enumerate(split_arrays):
                        if arr_idx != k_idx:                    # remove the k(th) portion
                            tmp_selected_index.extend(tmp_array)
                    select_shape_index_per_class.append(tmp_selected_index)

                # convert to 1-d list
                select_shape_idx = [item for alist in select_shape_index_per_class for item in alist]

                # print, for test
                basic.outputlogMessage("selected polygons index: " +
                                       " ".join([str(ii) for ii in select_shape_idx]))

                pass
            else:
                select_shape_idx = []
                split_arrays = numpy.array_split(all_shape_index, k_value)
                for arr_idx, tmp_array in enumerate(split_arrays):
                    if arr_idx != k_idx:
                        select_shape_idx.extend(tmp_array)

                # for test
                basic.outputlogMessage("selected polygons index: " +
                                        " ".join([str(ii) for ii in select_shape_idx]))

            for shape_idx in select_shape_idx:
                rec = org_records[shape_idx]
                w._shapes.append(shapes_list[shape_idx])
                w.records.append(rec)

            # save the shape file
            save_path = io_function.get_name_by_adding_tail(out_shp,'%dfold_%d'%(k_value,k_idx+1))

            # copy prj file
            org_prj = os.path.splitext(shape_file)[0] + ".prj"
            out_prj = os.path.splitext(save_path)[0] + ".prj"
            io_function.copy_file_to_dst(org_prj, out_prj, overwrite=True)

            w.save(save_path)

        return True



    def get_portition_of_polygons(self,shape_file,out_shp,percentage,sep_field_name=None):
        """
        randomly select polygons from different classes of saved polygons
        Args:
            shape_file: intput
            out_shp: output
            percentage: percentage
            sep_field_name: field name for different class

        Returns: True if successful, False Otherwise

        """

        if io_function.is_file_exist(shape_file) is False:
            return False

        try:
            org_obj = shapefile.Reader(shape_file)
        except:
            basic.outputlogMessage(str(IOError))
            return False

        # Create a new shapefile in memory
        w = shapefile.Writer()
        w.shapeType = org_obj.shapeType

        org_records = org_obj.records()
        shapes_list = org_obj.shapes()
        org_shape_count = len(shapes_list)

        if (len(org_records) < 1):
            basic.outputlogMessage('error, no record in shape file ')
            return False

        # Copy over the geometry without any changes
        w.fields = list(org_obj.fields)

        if sep_field_name is not None:
            field_index = self.__find_field_index(w.fields, sep_field_name)
            if field_index is False:
                return False

            all_shape_index = list(range(0, org_shape_count))

            # get class_int from this field
            class_int_list = [rec[field_index] for rec in org_records]
            class_unique_list = list(set(class_int_list)) # unique class id

            # for each class, choose a subset of them
            all_shape_index_per_class = []
            for class_id in class_unique_list:
                tmp_class_shp_idx = []
                for shp_indx in all_shape_index:
                    if org_records[shp_indx][field_index] == class_id:
                        tmp_class_shp_idx.append(shp_indx)

                all_shape_index_per_class.append(tmp_class_shp_idx)

            # selection in each class
            select_shape_index_per_class = []
            for class_id,shape_idx_a_class in enumerate(all_shape_index_per_class):
                select_count = int(len(shape_idx_a_class) * percentage)

                # don't subsample the non-gully polygons, hlc 2018-oct 21, as eboling case, class_id=0 is non-gully
                if class_id==0:
                    select_count = len(shape_idx_a_class)

                select_shape_idx_a_class = random.sample(shape_idx_a_class, select_count)
                select_shape_index_per_class.append(select_shape_idx_a_class)

            # convert to 1-d list
            select_shape_idx = [item for alist in select_shape_index_per_class for item in alist]

            #print
            basic.outputlogMessage("selected polygons index: "+
                                    " ".join([str(ii) for ii in select_shape_idx]))

        else:
            # select from the whole polgyons
            all_shape_index = list(range(0,org_shape_count))
            select_count = int(org_shape_count*percentage)
            select_shape_idx = random.sample(all_shape_index, select_count)

            # for test
            basic.outputlogMessage("selected polygons index: " +
                                    " ".join([str(ii) for ii in select_shape_idx]))

        for shape_idx in select_shape_idx:
            rec = org_records[shape_idx]
            w._shapes.append(shapes_list[shape_idx])
            w.records.append(rec)

        # copy prj file
        org_prj = os.path.splitext(shape_file)[0] + ".prj"
        out_prj = os.path.splitext(out_shp)[0] + ".prj"
        io_function.copy_file_to_dst(org_prj, out_prj, overwrite=True)

        w.save(out_shp)
        return True

    def remove_shapes_by_list(self,shape_file,out_shp,remove_list):
        """
        remove polygons based on the list
        :param shape_file: input shapefile containing all the polygons
        :param out_shp: output shapefile
        :param remove_list: if True in the list, then the polygon will be removed
        :return: True if successful, False Otherwise
        """
        if io_function.is_file_exist(shape_file) is False:
            return False

        try:
            org_obj = shapefile.Reader(shape_file)
        except:
            basic.outputlogMessage(str(IOError))
            return False

        # Create a new shapefile in memory
        w = shapefile.Writer()
        w.shapeType = org_obj.shapeType

        org_records = org_obj.records()
        if (len(org_records) < 1):
            basic.outputlogMessage('error, no record in shape file ')
            return False

        # Copy over the geometry without any changes
        w.fields = list(org_obj.fields)
        shapes_list = org_obj.shapes()
        org_shape_count = len(shapes_list)
        if org_shape_count != len(remove_list):
            basic.outputlogMessage('warning, the count of remove list is not equal to the shape count')

        i = 0
        removed_count = 0
        # for i in range(0,len(shapes_list)):
        for i, (shape, bremove) in enumerate(zip(shapes_list,remove_list)):
            if bremove :       # remove the record
                removed_count = removed_count +1
                continue

            w._shapes.append(shapes_list[i])
            rec = org_records[i]
            w.records.append(rec)

        basic.outputlogMessage('Remove shapes, total count: %d'%removed_count)
        # w._shapes.extend(org_obj.shapes())
        if removed_count==org_shape_count:
            basic.outputlogMessage('error: already remove all the shapes in the file')
            return False

        # copy prj file
        org_prj = os.path.splitext(shape_file)[0] + ".prj"
        out_prj = os.path.splitext(out_shp)[0] + ".prj"
        io_function.copy_file_to_dst(org_prj, out_prj,overwrite=True)

        w.save(out_shp)
        return True

    def remove_polygons_intersect_multi_polygons(self,shp_file, shp_ref, output, copy_fields=None):
        """
        remove polygon in shp_file if intersect with two or more polygons in shp_ref
        copy the attributes if it only intersects one polygon in shp_ref
        remov polygon if it don't intersects any polygons in shp_ref
        Args:
            shp_file: path of shape file
            shp_ref: path of reference, e.g., validation shape file
            output: save path
            copy_fields: the fields want to copy

        Returns: True is successful, Fasle otherwise

        """
        polygons = self.get_shapes(shp_file)
        ref_polygons = self.get_shapes(shp_ref)

        if len(polygons) < 1:
            raise ValueError('there is no shapes in %s' % shp_file)
        if len(ref_polygons) < 1:
            raise ValueError('there is no polygon in %s' % shp_ref)

        # to shapyly object
        shapely_polygons = [shape_from_pyshp_to_shapely(item) for item in polygons]
        ref_shapely_polygons = [ shape_from_pyshp_to_shapely(item) for item in ref_polygons]

        # get the field values
        field_records_list = self.get_shape_records_value(shp_ref, attributes=copy_fields)
        if field_records_list is False:
            raise IOError('read field values:%s from %s failed'%(str(copy_fields),shp_ref))

        #     operation_obj.add_multi_field_records_to_shapefile(output_shp, field_value_list, copy_field)
        # else:
        #     basic.outputlogMessage('get field %s failed' % str(copy_field))

        intersect_counts = []
        copied_field_records = [] # 3d list
        for polygon in shapely_polygons:
            count = 0
            records_list = []
            for ref,record in zip(ref_shapely_polygons,field_records_list):
                inte_res = polygon.intersection(ref)
                if inte_res.is_empty is False:
                    count += 1
                    records_list.append(record)

            intersect_counts.append(count)
            copied_field_records.append(records_list)

        ##############################################################
        # copy field and save to files
        org_obj = shapefile.Reader(shp_file)

        # Create a new shapefile in memory
        w = shapefile.Writer()
        w.shapeType = org_obj.shapeType

        org_records = org_obj.records()
        if (len(org_records) < 1):
            basic.outputlogMessage('error, no record in shape file ')
            return False

        # Copy over the geometry without any changes
        w.fields = list(org_obj.fields)
        shapes_list = org_obj.shapes()
        org_shape_count = len(shapes_list)

        removed_count = 0
        removed_index = []
        for i in range(0,len(shapes_list)):
            if intersect_counts[i] != 1:
            # if intersect_counts[i] < 2:
                # delelte the copied field records
                removed_count += 1
                removed_index.append(i)
                basic.outputlogMessage('remove (%d)th shape, which intersect with other %d shapes'%(i,intersect_counts[i]))
                continue

            w._shapes.append(shapes_list[i])
            rec = org_records[i]
            w.records.append(rec)

        basic.outputlogMessage('Remove shapes, total count: %d'%removed_count)
        # w._shapes.extend(org_obj.shapes())
        if removed_count==org_shape_count:
            basic.outputlogMessage('error: already remove all the shapes in the file')
            return False

        # copy prj file
        org_prj = os.path.splitext(shp_file)[0] + ".prj"
        out_prj = os.path.splitext(output)[0] + ".prj"
        io_function.copy_file_to_dst(org_prj, out_prj,overwrite=True)
        w.save(output)

        # add the copied records
         # 3d to 2d list
        copied_field_records_2d = []
        for idx,item in enumerate(copied_field_records):
            if idx in removed_index:
                continue
            copied_field_records_2d.append(item[0])
        return self.add_multi_field_records_to_shapefile(output, copied_field_records_2d, copy_fields)



    def remove_nonclass_polygon(self,shape_file,out_shp,class_field_name):
        """
        remove polygons that are not belong to targeted class, it means the value of class_field_name is 0
        :param shape_file: input shapefile containing all the polygons
        :param out_shp: output shapefile
        :param class_field_name: the name of class field, such as svmclass, treeclass
        :return: True if successful, False Otherwise
        """
        if io_function.is_file_exist(shape_file) is False:
            return False

        try:
            org_obj = shapefile.Reader(shape_file)
        except:
            basic.outputlogMessage(str(IOError))
            return False

        # Create a new shapefile in memory
        w = shapefile.Writer()
        w.shapeType = org_obj.shapeType

        org_records = org_obj.records()
        if (len(org_records) < 1):
            basic.outputlogMessage('error, no record in shape file ')
            return False

        # Copy over the geometry without any changes
        w.fields = list(org_obj.fields)
        field_index = self.__find_field_index(w.fields, class_field_name)
        if field_index is False:
            return False
        shapes_list = org_obj.shapes()
        org_shape_count = len(shapes_list)
        i = 0
        removed_count = 0
        for i in range(0,len(shapes_list)):
            rec = org_records[i]
            if rec[field_index] == 0:       # remove the record which class is 0, 0 means non-gully
                removed_count = removed_count +1
                continue

            w._shapes.append(shapes_list[i])
            rec = org_records[i]
            w.records.append(rec)

        basic.outputlogMessage('Remove non-class polygon, total count: %d'%removed_count)
        # w._shapes.extend(org_obj.shapes())
        if removed_count==org_shape_count:
            basic.outputlogMessage('error: already remove all the polygons')
            return False

        # copy prj file
        org_prj = os.path.splitext(shape_file)[0] + ".prj"
        out_prj = os.path.splitext(out_shp)[0] + ".prj"
        io_function.copy_file_to_dst(org_prj, out_prj,overwrite=True)

        w.save(out_shp)
        return True



    def get_shape_records_value(self,input_shp,attributes=None):
        """
        get several fields value of shape file
        :param input_shp: shape file path
        :param attributes: a list contains the field name want to read
        :return: a 2D list contains the records value of the several fields.
        """
        #  read existing shape file
        if io_function.is_file_exist(input_shp) is False:
            return False
        attributes_value = [] # 2 dimensional
        attributes_index = [] # 1 dimensional

        try:
            org_obj = shapefile.Reader(input_shp)
        except IOError:
            basic.outputlogMessage(str(IOError))
            return False
        all_fields = org_obj.fields

        # test
        # txt_obj = open('attributes.txt','w')
        # outstr=''


        if attributes is None:
            # return all attribute values
            all_records = org_obj.records()
            return all_records
        elif isinstance(attributes,list):
            # return selected attributes
            field_len = len(all_fields)
            sel_attrib_len = len(attributes)
            attributes_index = [-1]*sel_attrib_len
            for sel_index in range(0, sel_attrib_len):
                sel_attrib = attributes[sel_index]
                for t_index in range(0,field_len):
                    t_field = all_fields[t_index]
                    if isinstance(t_field, tuple):
                        # t_index += 1  occur once
                        continue
                    if sel_attrib == t_field[0]:
                        attributes_index[sel_index] = t_index - 1  # first field not show in records
                        break

            # check whether all the attributes are found
            b_attribute_not_found = False
            for sel_index in range(0, sel_attrib_len):
                if attributes_index[sel_index] == -1:
                    b_attribute_not_found = True
                    basic.outputlogMessage('error, attribute %s not found in the fields'%attributes[sel_index])
            if b_attribute_not_found:
                return False
        else:
            basic.outputlogMessage('error, input attributes list is not correct')
            return False

        all_records = org_obj.records()

        for i in range(0, len(all_records)):
            rec = all_records[i]
            # sel_attrib_value = []
            # for j in attributes_index:
            #     if isinstance(rec[j],bytes):
            #         sel_attrib_value.append("NULL")
            #     else:
            #         sel_attrib_value.append(rec[j])
            sel_attrib_value = [rec[j] for j in attributes_index]
            attributes_value.append(sel_attrib_value)

        return attributes_value

    def save_attributes_values_to_text(self,attributes_values,out_file):
        """
        save attributes values (2D list) to text file
        :param attributes_values: 2D list attributes values (records count , nfeature)
        :param out_file: save file path
        :return: True is successful, False otherwise
        """
        if isinstance(attributes_values,list) is False:
            basic.outputlogMessage('input attributes_values must be list')
            return False

        if attributes_values is not False:
            f_obj = open(out_file, 'w')
            for record in attributes_values:
                out_str = ''
                for value in record:
                    if isinstance(value, float):
                        out_str += '%.2f' % value + '\t'
                    elif isinstance(value, bytes):
                        out_str += 'NULL' + '\t'  # str(value,'utf-8') + '\t'
                    else:
                        out_str += str(value) + '\t'
                        # if value=='gully':
                        #     test = 1
                f_obj.writelines(out_str + '\n')
            f_obj.close()

        return True


def shape_from_shapely_to_pyshp(shapely_shape,keep_holes=True):
    """
    convert shape object in the shapely object to pyshp object (shapefile)
    Note however: the function will NOT handle any interior polygon holes if there are any,
    it simply ignores them.
    :param shapely_shape: the shapely object
    :return: object if successful, False otherwise
    """
    # first convert shapely to geojson
    try:
        shapelytogeojson = shapely.geometry.mapping
    except:
        import shapely.geometry
        shapelytogeojson = shapely.geometry.mapping

    pyshptype = 0  # mean NULL, which is default when shapely_shape.is_empty is true
    geoj = shapelytogeojson(shapely_shape)
    # create empty pyshp shape
    if shapefile.__version__ < '2.0.0': # different verion of pyshp,  Jan 15, 2019 hlc
        record = shapefile._Shape()
    else:
        record = shapefile.Shape()
    # set shapetype
    if geoj["type"] == "Null":
        pyshptype = 0
    elif geoj["type"] == "Point":
        pyshptype = 1
    elif geoj["type"] == "LineString":
        pyshptype = 3
    elif geoj["type"] == "Polygon":
        pyshptype = 5
    elif geoj["type"] == "MultiPoint":
        pyshptype = 8
    elif geoj["type"] == "MultiLineString":
        pyshptype = 3
    elif geoj["type"] == "MultiPolygon":
        pyshptype = 5
    record.shapeType = pyshptype
    # set points and parts
    if geoj["type"] == "Point":
        record.points = geoj["coordinates"]
        record.parts = [0]
    elif geoj["type"] in ("MultiPoint","Linestring"):
        record.points = geoj["coordinates"]
        record.parts = [0]
    elif geoj["type"] in ("Polygon"):
        if keep_holes is False:
            record.points = geoj["coordinates"][0]
            record.parts = [0]
        else:
            # need to consider the holes
            all_coordinates = geoj["coordinates"];
            parts_count = len(all_coordinates)
            # print(parts_count)
            record.parts = []
            for i in range(0,parts_count):
                # record.points = geoj["coordinates"][0]
                record.parts.append(len(record.points))
                record.points.extend(all_coordinates[i])


    elif geoj["type"] in ("MultiPolygon","MultiLineString"):
        if keep_holes is False:
            index = 0
            points = []
            parts = []
            for eachmulti in geoj["coordinates"]:
                points.extend(eachmulti[0])
                parts.append(index)
                index += len(eachmulti[0])
            record.points = points
            record.parts = parts
        else:
            # need to consider the holes
            all_polygons_coor = geoj["coordinates"]
            polygon_count = len(all_polygons_coor)
            # print(parts_count)
            record.parts = []
            for i in range(0, polygon_count):
                for j in range(0,len(all_polygons_coor[i])):
                    # record.points = geoj["coordinates"][0]
                    record.parts.append(len(record.points))
                    record.points.extend(all_polygons_coor[i][j])


        # index = 0
        # points = []
        # parts = []
        # skip = 1
        # for eachmulti in geoj["coordinates"]:
        #     if skip ==1 :
        #         skip = skip + 1
        #         continue
        #
        #     points.extend(eachmulti[0])
        #     parts.append(index)
        #     index += len(eachmulti[0])
        # record.points = points
        # record.parts = parts
    return record

def shape_from_pyshp_to_shapely(pyshp_shape,shape_index=None, shp_path=None):
    """
     convert pyshp object to shapely object
    :param pyshp_shape: pyshp (shapefile) object
    :return: shapely object if successful, False otherwise
    """

    if pyshp_shape.shapeType is 5:  # Polygon or multiPolygon
        parts_index = pyshp_shape.parts
        if len(parts_index) < 2:
            # Create a polygon with no holes
            record = Polygon(pyshp_shape.points)
        else:
            # Create a polygon with one or several holes
            seperate_parts = []
            parts_index.append(len(pyshp_shape.points))
            for i in range(0, len(parts_index)-1):
                points = pyshp_shape.points[parts_index[i]:parts_index[i+1]]
                seperate_parts.append(points)

            # if list(parts_index)==[0,121,130,135,140]:
            #     debug = 1

            # assuming the first part is exterior
            # exterior = seperate_parts[0]  # assuming the first part is exterior
            # interiors = [seperate_parts[i] for i in range(1,len(seperate_parts))]
            # assuming the last part is exterior
            # exterior = seperate_parts[len(parts_index)-2]
            # interiors = [seperate_parts[i] for i in range(0,len(seperate_parts)-2)]

            all_polygons = []

            while(len(seperate_parts)>0):
                if shapefile.signed_area(seperate_parts[0]) < 0: # the area of  ring is clockwise, it's not a hole
                    exterior = tuple(seperate_parts[0])
                    seperate_parts.remove(seperate_parts[0])

                    # find all the holes attach to the first exterior
                    interiors = []
                    holes_points = []
                    for points in seperate_parts:
                        if shapefile.signed_area(points) >= 0: # the value >= 0 means the ring is counter-clockwise,  then they form a hole
                            interiors.append(tuple(points))
                            holes_points.append(points)
                    # remove the parts which are holes
                    for points in holes_points:
                        seperate_parts.remove(points)
                        # else:
                        #     break
                    if len(interiors) < 1:
                        interiors = None
                    else:
                        interiors = tuple(interiors)
                    polygon = Polygon(shell=exterior,holes=interiors)
                    all_polygons.append(polygon)
                else:
                    # basic.outputlogMessage('error, holes found in the first ring')
                    basic.outputlogMessage("parts_index:"+str(parts_index)+'\n'+"len of seperate_parts:"+str(len(seperate_parts)))
                    if shape_index is not None:
                        basic.outputlogMessage("the %dth (0 index) polygon causing an error, "
                                               "please check its validity and fix it"%shape_index)
                    if shp_path is not None:
                        basic.outputlogMessage("the shape is in file: %s " % shp_path)
                    raise ValueError('error, holes found in the first ring')
                    # return False

            if len(all_polygons) > 1:
                record = MultiPolygon(polygons=all_polygons)
            else:
                record = all_polygons[0]

    elif pyshp_shape.shapeType is 13:   #POLYLINEZ = 13
        record = LineString(pyshp_shape.points) # have z value
    elif pyshp_shape.shapeType is 3:    #POLYLINE = 3
        record = LineString(pyshp_shape.points)
    else:
        # basic.outputlogMessage('have not complete, other type of shape is not consider!')
        # return False
        raise ValueError('currently, the shapeType is %d not supported'%pyshp_shape.shapeType)

    # # plot shape for checking
    # from matplotlib import pyplot as plt
    # from descartes import PolygonPatch
    # from math import sqrt
    # # from shapely.geometry import Polygon, LinearRing
    # # from shapely.ops import cascaded_union
    # BLUE = '#6699cc'
    # GRAY = '#999999'
    #
    # # plot these two polygons separately
    # fig = plt.figure(1,  dpi=90) #figsize=SIZE,
    # ax = fig.add_subplot(111)
    # poly1patch = PolygonPatch(record, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
    # # poly2patch = PolygonPatch(polygon2, ec=BLUE, alpha=0.5, zorder=2)
    # ax.add_patch(poly1patch)
    # # ax.add_patch(poly2patch)
    # boundary = record.bounds
    # xrange = [boundary[0], boundary[2]]
    # yrange = [boundary[1], boundary[3]]
    # ax.set_xlim(*xrange)
    # # ax.set_xticks(range(*xrange) + [xrange[-1]])
    # ax.set_ylim(*yrange)
    # # ax.set_yticks(range(*yrange) + [yrange[-1]])
    # # ax.set_aspect(1)
    #
    # plt.show()



    return record

def get_area_length_geometric_properties(shapely_polygons):
    """
    get the area, length, and other geometric properties of polygons (shapely object)
    :param shapely_polygons: a list contains the shapely object
    :return: list of area and length (perimeter)
    """
    area = []
    length = [] #(perimeter of polygon)

    if len(shapely_polygons) < 1:
        basic.outputlogMessage('error, No input polygons')
        return area, length    #  area and length is none

    for polygon in shapely_polygons:
        area_value = polygon.area
        length_value = polygon.length
        area.append(abs(area_value))   # area_value could be negative for a irregular polygons
        length.append(length_value)


    return area, length


def merge_touched_polygons(polygon_list,adjacent_matrix):
    """
    merge all the connected (touched, ) polygons into one polygon
    :param polygon_list: a list containing at least two polygons, shapely object not pyshp object
    :param adjacent_matrix: the matrix storing the touching relationship of all the polygons
    :return: a polygon list after merging if successful, False Otherwise
    """
    count = len(polygon_list)
    matrix_size = adjacent_matrix.shape
    merged_polygons = []
    if count !=matrix_size[0] or matrix_size[0] != matrix_size[1]:
        basic.outputlogMessage('the size of list and matrix do not agree')
        return False
    remain_polygons = list(range(0,count))
    for i in remain_polygons:
        if i < 0:
            continue
        index = [i]
        seed_index = [i]

        while(len(seed_index)>0):
            # set no new find
            new_find_index = []
            for seed in seed_index:
                i_adja = numpy.where(adjacent_matrix[seed,:]==1)
                new_find_index.extend(i_adja[0])

            # to be unique, because one polygon has several adjacent polygon
            new_find_index = numpy.unique(new_find_index).tolist()
            seed_index = []
            for new_value in new_find_index:
                if new_value not in index:
                    seed_index.append(new_value)
            index.extend(seed_index)

        print(i,index)
        for loc in index:
            remain_polygons[loc] = -1
        # merge polygons connected to each other
        if len(index) >1:
            # in my test data, the holes disappear in result. When the number of polygon is small, it work well
            # but when the number is large, the problem occurs.
            # try to merged them one by one
            touched_polygons = [polygon_list[loc] for loc in index]
            union_result = cascaded_union(touched_polygons)
            # union_result = touched_polygons[0]
            # for loc in range(1,len(touched_polygons)):
            #     union_result = cascaded_union([union_result,touched_polygons[loc]])
        else:
            union_result =  polygon_list[index[0]]
            # print(union_result)
        merged_polygons.append(union_result)

    return merged_polygons


def find_adjacent_polygons(polygon1, polygon2):
    """
    decide whether this two polygons are adjacent (share any same point)
    :param polygon1: polygon 1
    :param polygon2: polygon 2
    :return: True if they touch, False otherwise
    """
    try:
        # result = polygon1.touches(polygon2)
        # # two polygon share one line, then consider they are adjacent.
        # # if only share one point, then consider they are not adjacent.
        # if result is True:
        #     line = polygon1.intersection(polygon2)
        #     print( line.length , line.area )
        #     if line.length > 0:
        #         return True
        #     else:
        #         return False
        # tmp1 = polygon1.buffer(0.1)
        # tmp2 = polygon2.buffer(0.1)

        # due to use buffer can enlarge polygon, so the adjacent polygon not touch, they have overlay
        result = polygon1.intersection(polygon2)
        # print(result.length, result.area)
        if result.length > 0 or result.area > 0:
            return True
        else:
            return False


    except:
        # print("Unexpected error:", sys.exc_info()[0])
        basic.outputlogMessage('failed in touch polygon1 (valid %s) and polygon2 (valid %s)'%(polygon1.is_valid,polygon2.is_valid))
        # find whether they have shared points


        # assert False
        return False


def build_adjacent_map_of_polygons(polygons_list):
    """
    build an adjacent matrix of the tou
    :param polygons_list: a list contains all the shapely (not pyshp) polygons
    :return: a matrix storing the adjacent (shared points) for all polygons
    """
    polygon_count = len(polygons_list)
    if polygon_count < 2:
        basic.outputlogMessage('error, the count of polygon is less than 2')
        return False

    ad_matrix = numpy.zeros((polygon_count, polygon_count),dtype=numpy.int8)
    # ad_matrix= ad_matrix.astype(int)
    for i in range(0,polygon_count):
        # print(i, polygons_list[i].is_valid)
        if polygons_list[i].is_valid is False:
            polygons_list[i] = polygons_list[i].buffer(0.1)  # trying to solve self-intersection
        for j in range(i+1,polygon_count):
            # print(i,j,polygons_list[i].is_valid,polygons_list[j].is_valid)
            if polygons_list[j].is_valid is False:
                polygons_list[j] = polygons_list[j].buffer(0.1)  # trying to solve self-intersection
            if find_adjacent_polygons(polygons_list[i],polygons_list[j]) is True:
                # print(i,j)
                ad_matrix[i,j] = 1
                ad_matrix[j,i] = 1  # also need the low part of matrix, or later polygon can not find previous neighbours
    # print(ad_matrix)
    return ad_matrix



def merge_touched_polygons_in_shapefile(shape_file,out_shp):
    """
    merge touched polygons by using sharply function based on GEOS library
    :param shape_file: input shape file path
    :param output:  output shape file path
    :return: True if successful, False otherwise
    """
    if io_function.is_file_exist(shape_file) is False:
        return False

    try:
        org_obj = shapefile.Reader(shape_file)
    except:
        basic.outputlogMessage(str(IOError))
        return False

    # Create a new shapefile in memory
    w = shapefile.Writer()
    w.shapeType = org_obj.shapeType

    org_records = org_obj.records()
    if (len(org_records) < 1):
        basic.outputlogMessage('error, no record in shape file ')
        return False

    # Copy over the geometry without any changes
    w.field('id',fieldType = "N", size = "24")
    shapes_list = org_obj.shapes()

    # polygon1 = Polygon(shapes_list[5].points)
    # polygon2 = Polygon(shapes_list[6].points)
    # # polygon2 = Polygon([(0, 0), (3, 10), (3, 0)])
    # polygons = [polygon1, polygon2]
    # u = cascaded_union(polygons)

    polygon_shapely = []
    for temp in shapes_list:
        polygon_shapely.append(shape_from_pyshp_to_shapely(temp))
    #test save polygon with holes
    # merge_result = polygon_shapely

    adjacent_matrix = build_adjacent_map_of_polygons(polygon_shapely)

    if adjacent_matrix is False:
        return False
    merge_result = merge_touched_polygons(polygon_shapely,adjacent_matrix)

    b_keep_holse = parameters.get_b_keep_holes()
    pyshp_polygons = [shape_from_shapely_to_pyshp(shapely_polygon,keep_holes=b_keep_holse) for shapely_polygon in merge_result ]
    # test
    # pyshp_polygons = [shape_from_shapely_to_pyshp(merge_result[0])]

    # org_records = org_obj.records()
    for i in range(0,len(pyshp_polygons)):
        w._shapes.append(pyshp_polygons[i])
        rec =  [i]     # add id
        w.records.append(rec)
    #
    # copy prj file
    org_prj = os.path.splitext(shape_file)[0] + ".prj"
    out_prj = os.path.splitext(out_shp)[0] + ".prj"
    io_function.copy_file_to_dst(org_prj, out_prj,overwrite=True)
    #
    # overwrite original file
    w.save(out_shp)


    return True

def cal_area_length_of_polygon(shape_file):

    if io_function.is_file_exist(shape_file) is False:
        return False

    try:
        org_obj = shapefile.Reader(shape_file)
    except:
        basic.outputlogMessage(str(IOError))
        return False

    # Create a new shapefile in memory
    # w = shapefile.Writer()
    # w.shapeType = org_obj.shapeType

    org_records = org_obj.records()
    if (len(org_records) < 1):
        basic.outputlogMessage('error, no record in shape file ')
        return False

    # Copy over the geometry without any changes
    shapes_list = org_obj.shapes()

    # polygon1 = Polygon(shapes_list[5].points)
    # polygon2 = Polygon(shapes_list[6].points)
    # # polygon2 = Polygon([(0, 0), (3, 10), (3, 0)])
    # polygons = [polygon1, polygon2]
    # u = cascaded_union(polygons)

    polygon_shapely = []
    for temp in shapes_list:
        polygon_shapely.append(shape_from_pyshp_to_shapely(temp))

    # add area, perimeter to shapefile
    # caluclate the area, perimeter
    area, perimeter = get_area_length_geometric_properties(polygon_shapely)

    if len(area)>0 and len(perimeter) > 0:
        shp_obj = shape_opeation()
        shp_obj.add_one_field_records_to_shapefile(shape_file,area,'INarea')
        shp_obj.add_one_field_records_to_shapefile(shape_file, perimeter, 'INperimete')
        shp_obj = None

    # copy prj file
    # org_prj = os.path.splitext(shape_file)[0] + ".prj"
    # out_prj = os.path.splitext(out_shp)[0] + ".prj"
    # io_function.copy_file_to_dst(org_prj, out_prj,overwrite=True)
    #

    return True

def save_shapely_shapes_to_file(shapes_list, ref_shp, output_shp,copy_field=None):
    """
    save shapes in shapely format to a file
    Args:
        shapes_list: shapes list, can be polygon, line, and so on
        ref_shp: reference shapefile containing the projection information
        output_shp: save path
        copy_field: a list contain filed names, copy the fields from "ref_shp", the count in shapes_list should be the same as the record in ref_shp

    Returns:

    """
    # Create a new shapefile in memory
    w = shapefile.Writer()
    # w.shapeType = org_obj.shapeType ##???

    # create a field
    w.field('id', fieldType="N", size="24")
    # the empty will be removed
    remove_index = []

    # to shapes in pyshp format
    # shapely_shape may contain empty
    pyshp_shapes = [shape_from_shapely_to_pyshp(shapely_shape, keep_holes=True) for shapely_shape in
                      shapes_list]

    for i, save_shape in enumerate(pyshp_shapes):
        if save_shape.shapeType is 0:   # null, don't have geometry
            basic.outputlogMessage('skip empty geometry at %d'%i)
            remove_index.append(i)
            continue
        w._shapes.append(save_shape)
        rec = [i]  # add id
        # rec = [org_records[i][0]] # copy id
        w.records.append(rec)

    # copy prj file
    org_prj = os.path.splitext(ref_shp)[0] + ".prj"
    out_prj = os.path.splitext(output_shp)[0] + ".prj"
    io_function.copy_file_to_dst(org_prj, out_prj, overwrite=True)

    # save the file
    w.save(output_shp)

    # copy the field from the ref_shp
    operation_obj = shape_opeation()
    if copy_field is not None:
        field_value_list = operation_obj.get_shape_records_value(ref_shp, attributes=copy_field)
        if field_value_list is not False:
            # add the field values
            if len(remove_index) > 0:
                for index in remove_index:
                    del field_value_list[index]

            operation_obj.add_multi_field_records_to_shapefile(output_shp,field_value_list,copy_field)
        else:
            basic.outputlogMessage('get field %s failed'%str(copy_field))

    return True


def IoU(polygon1,polygon2):
    """
    calculate IoU (intersection over union ) of two polygon
    :param polygon1: polygon 1 of shaply object
    :param polygon2: polygon 2 of shaply object
    :return: IoU value [0, 1], Flase Otherwise
    """

    # # plot shape for checking
    # from matplotlib import pyplot as plt
    # from descartes import PolygonPatch
    # from math import sqrt
    # # from shapely.geometry import Polygon, LinearRing
    # # from shapely.ops import cascaded_union
    # BLUE = '#6699cc'
    # GRAY = '#999999'
    #
    # # plot these two polygons separately
    # fig = plt.figure(1,  dpi=90) #figsize=SIZE,
    # ax = fig.add_subplot(111)
    # poly1patch = PolygonPatch(polygon1, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
    # # poly2patch = PolygonPatch(polygon2, ec=BLUE, alpha=0.5, zorder=2)
    # ax.add_patch(poly1patch)
    # # ax.add_patch(poly2patch)
    # boundary = polygon1.bounds
    # xrange = [boundary[0], boundary[2]]
    # yrange = [boundary[1], boundary[3]]
    # ax.set_xlim(*xrange)
    # # ax.set_xticks(range(*xrange) + [xrange[-1]])
    # ax.set_ylim(*yrange)
    # # ax.set_yticks(range(*yrange) + [yrange[-1]])
    # # ax.set_aspect(1)
    #
    # plt.show()

    intersection = polygon1.intersection(polygon2)
    if intersection.is_empty is True:
        return 0.0

    union = polygon1.union(polygon2)

    return intersection.area/union.area

def max_IoU_score(polygon, polygon_list):
    """
    get the IoU score of one polygon compare to many polygon (validation polygon)
    :param polygon: the detected polygon
    :param polygon_list: a list contains training polygons
    :return: the max IoU score, False otherwise
    """
    max_iou = 0.0
    for training_polygon in polygon_list:
        temp_iou = IoU(polygon,training_polygon)
        if temp_iou > max_iou:
            max_iou = temp_iou
    return max_iou

def calculate_IoU_scores(result_shp,val_shp):
    """
    calculate the IoU score of polygons in shape file
    :param result_shp: result shape file contains detected polygons
    :param val_shp: shape file contains validation polygons
    :return: a list contains the IoU of all the polygons
    """
    if io_function.is_file_exist(result_shp) is False or io_function.is_file_exist(val_shp) is False:
        return False

    try:
        result_obj = shapefile.Reader(result_shp)
    except:
        basic.outputlogMessage(str(IOError))
        raise IOError('failed to read %s'%result_shp)
        # return False
    try:
        val_obj = shapefile.Reader(val_shp)
    except:
        basic.outputlogMessage(str(IOError))
        raise IOError('failed to read %s' % val_shp)
        # return False

    result_polygon_list = result_obj.shapes()
    val_polygon_list = val_obj.shapes()

    if (len(result_polygon_list) < 1):
        basic.outputlogMessage('error, no detected polygon in %s'%result_shp)
        return False
    if (len(val_polygon_list) < 1):
        basic.outputlogMessage('error, no detected polygon in %s'%val_shp)
        return False

    result_polygon_shapely = []
    for temp in result_polygon_list:
        shaply_obj = shape_from_pyshp_to_shapely(temp)
        shaply_obj = shaply_obj.buffer(0.01)  # buffer (0) solve the self-intersection problem, but don't know how it work
        result_polygon_shapely.append(shaply_obj)

    val_polygon_shapely = []
    for temp in val_polygon_list:
        shaply_obj = shape_from_pyshp_to_shapely(temp)
        shaply_obj = shaply_obj.buffer(0.01)  # buffer (0) solve the self-intersection problem
        val_polygon_shapely.append(shaply_obj)
    #
    # for temp in result_polygon_shapely:
    #     # temp = temp.buffer(0.0)
    #     print('result',temp.is_valid)
    #
    # for temp in val_polygon_shapely:
    #     # temp = temp.buffer(0.0)
    #     print('val',temp.is_valid)

    #output shp


    IoU_socres = []
    index = 0
    for result_polygon in result_polygon_shapely:
        index = index + 1
        iou = max_IoU_score(result_polygon,val_polygon_shapely)
        print(index, iou)
        IoU_socres.append(iou)

    return IoU_socres

def read_attribute(shapefile, field_name):
    """
    read one attribute of shapefile
    Args:
        shapefile: shapefile path
        field_name: name of the attribute

    Returns: a list contains the values of the field, False otherwise

    """
    operation_obj = shape_opeation()
    output_list = operation_obj.get_shape_records_value(shapefile, attributes=[field_name])
    if len(output_list) < 1:
        return False
    else:
        value_list = [item[0] for item in output_list]
        return value_list

def get_buffer_polygons(input_shp,output_shp,buffer_size):
    """
    get the buffer area (polygon with hole) of all the polygons. it may not work if the input polygons is complicate,
    such as the polygon has holes
    Args:
        input_shp: input shape file containing polygons
        output_shp: output shape file containing buffer area
        buffer_size: the buffer size in meters (or should be the same metric of the input shape file)

    Returns: True if successful, False othewise

    """

    if io_function.is_file_exist(input_shp) is False:
        return False

    try:
        org_obj = shapefile.Reader(input_shp)
    except:
        basic.outputlogMessage(str(IOError))
        return False

    # Create a new shapefile in memory
    w = shapefile.Writer()
    w.shapeType = org_obj.shapeType

    org_records = org_obj.records()
    if (len(org_records) < 1):
        basic.outputlogMessage('error, no record in shape file ')
        return False

    # Copy over the geometry without any changes
    w.field('id_buf', fieldType="N", size="24")
    shapes_list = org_obj.shapes()

    polygon_shapely = []
    for temp in shapes_list:
        polygon_shapely.append(shape_from_pyshp_to_shapely(temp))

    # buffer the polygons
    expansion_polygons = [ item.buffer(buffer_size) for item in  polygon_shapely]
    buffer_area = []
    for i in range(0,len(expansion_polygons)):
        try:
            expansion_poly = expansion_polygons[i]
            org_poly = polygon_shapely[i]
            buffer_area.append(expansion_poly.difference(org_poly))
        except : #BaseException as e
            raise ValueError('Error at the %d (start from 1) polygon'%(i+1))

    # save the buffer area (polygon)
    pyshp_polygons = [shape_from_shapely_to_pyshp(shapely_polygon, keep_holes=True) for shapely_polygon in
                      buffer_area]

    # org_records = org_obj.records()
    for i in range(0, len(pyshp_polygons)):
        w._shapes.append(pyshp_polygons[i])
        # rec = [i]  # add id
        rec = [org_records[i][0]] # copy id
        w.records.append(rec)
    #
    # copy prj file
    org_prj = os.path.splitext(input_shp)[0] + ".prj"
    out_prj = os.path.splitext(output_shp)[0] + ".prj"
    io_function.copy_file_to_dst(org_prj, out_prj, overwrite=True)
    #
    # overwrite original file
    w.save(output_shp)


    return True

def get_intersection_of_line_polygon(shp_line,shp_polygon):
    '''
    get the intersection between lines and polygons
    Args:
        shp_line:
        shp_polygon:

    Returns: a list of intersection corresponding to the lines, it is none if no intersection

    '''
    # read
    operation_obj = shape_opeation()
    line_shapes = operation_obj.get_shapes(shp_line)
    polygon_shapes = operation_obj.get_shapes(shp_polygon)

    if len(line_shapes) < 1:
        raise ValueError('there is no shapes in %s' % shp_line)
    if len(polygon_shapes) < 1:
        raise ValueError('there is no polygon in %s' % shp_polygon)

    # to shapyly object
    lines = [shape_from_pyshp_to_shapely(item) for item in line_shapes]
    polygons = [shape_from_pyshp_to_shapely(item) for item in polygon_shapes]

    lines_inters_list = []
    polygon_count = len(polygons)
    for line in lines:
        inte_res = line.intersection(polygons[0])  # initial
        if inte_res.is_empty:
            for idx in range(1,polygon_count):  # skip the first one
                inte_res = line.intersection(polygons[idx])
                if inte_res.is_empty is False:
                    break
        lines_inters_list.append(inte_res)

    basic.outputlogMessage('compete computing the intersection of %d shapes '%len(lines))
    return lines_inters_list

def get_intersection_of_polygon_polygon(shp_polygon1,shp_polygon2,output,copy_field=None):
    '''
    get the intersection between polygons in two shape files
    Args:
        shp_polygon1: based shape
        shp_polygon2: find inteser
        output: save intersection to file
        copy_field: copy the fields from "shp_polygon1"

    Returns:

    '''
    # get intersections (only consider the first intersection)
    # a list of intersection corresponding to the polygons in the first shape file, it is none if no intersection
    inters_list = get_intersection_of_line_polygon(shp_polygon1, shp_polygon2)

    # save to file
    return save_shapely_shapes_to_file(inters_list,shp_polygon1,output,copy_field=copy_field)

def get_adjacent_polygon_count(polygons_shp,buffer_size):
    '''
    get the polygon count at the buffer area for each polygon
    Args:
        polygons_shp:
        buffer_size:

    Returns: a list cotaining the count

    '''
    operation_obj = shape_opeation()
    pyshp_polygons = operation_obj.get_shapes(polygons_shp)

    if len(pyshp_polygons) < 1:
        raise ValueError('there is no polygon in %s' % polygons_shp)

    # to shapyly object
    shapely_polygons = [shape_from_pyshp_to_shapely(item) for item in pyshp_polygons]

    adjacent_counts = []
    for idx in range(0,len(shapely_polygons)):
        # get buffer areas
        polygon = shapely_polygons[idx]
        expansion_polygon = polygon.buffer(buffer_size)
        # get count based on intersection with others
        tmp_list = list(range(0,len(shapely_polygons)))
        tmp_list.remove(idx)    # remove himself
        count = 0
        for index in tmp_list:
            inte_res = expansion_polygon.intersection(shapely_polygons[index])
            if inte_res.is_empty is False:
                count += 1
        #
        adjacent_counts.append(count)

    return adjacent_counts



def test(input,output):

    operation_obj = shape_opeation()
    operation_obj.get_polygon_shape_info(input,output)

    # save_shp = "saved.shp"
    # operation_obj.add_fields_shape(input,output,save_shp)

    # operation_obj.add_fields_from_raster(input,output,"color")

    # operation_obj.remove_shape_baseOn_field_value(input,output)
    # operation_obj.remove_nonclass_polygon(input, output,'svmclass')
    # merge_touched_polygons_in_shapefile(input,output)

    # result_shp = '/Users/huanglingcao/Data/eboling/eboling_uav_images/dom/output.shp'
    # val_shp = '/Users/huanglingcao/Data/eboling/training_validation_data/gps_rtk/gps_rtk_polygons_2.shp'
    # result_shp = "/Users/huanglingcao/Data/eboling/training_validation_data/gps_rtk/result_iou_test2.shp"
    # val_shp = "/Users/huanglingcao/Data/eboling/training_validation_data/gps_rtk/val_iou_test2.shp"
    #
    # # result_shp = "/Users/huanglingcao/Data/eboling/training_validation_data/gps_rtk/result_iou.shp"
    # # val_shp = "/Users/huanglingcao/Data/eboling/training_validation_data/gps_rtk/val_iou.shp"
    # iou_score = calculate_IoU_scores(result_shp,val_shp)
    # # print (iou_score)

    # merge_touched_polygons_in_shapefile(input,output)

    operation_obj = None

    pass

def test_get_attribute_value(input,parameter_file):
    if io_function.is_file_exist(parameter_file) is False:
        return False
    attributes_names = parameters.get_attributes_used(parameter_file)
    operation_obj = shape_opeation()
    attributes_values = operation_obj.get_shape_records_value(input, attributes=attributes_names)
    if attributes_values is not False:
        out_file = 'attribute_value.txt'
        operation_obj.save_attributes_values_to_text(attributes_values, out_file)

    operation_obj = None


def test_get_buffer_polygon():

    input_shp = "/Users/huanglingcao/visual_dir/test_polygon_merge/EboDOM_deeplab_12_gully_post.shp"
    output_shp = "/Users/huanglingcao/visual_dir/test_polygon_merge/EboDOM_deeplab_12_gully_post_buffer.shp"
    buffer_size = 5 # meters

    return  get_buffer_polygons(input_shp, output_shp,buffer_size)

def test_add_one_field_records_to_shapefile():

    shp = '/Users/huanglingcao/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/polygon_based_ChangeDet/' \
          'manu_blh_2017To2019/change_manu_blh_2017To2019_T_201807_vs_201907.shp'

    insert_text = []
    for idx in range(349):
        tmp_str = '_'.join([str(i) for i in range(idx+1)])
        insert_text.append(tmp_str)

    shp_obj = shape_opeation()
    shp_obj.add_one_field_records_to_shapefile(shp, insert_text, 'text')

def main(options, args):
    # if len(args) != 2:
    #     basic.outputlogMessage('error, the number of input argument is 2')
    #     return False

    if options.para_file is None:
        basic.outputlogMessage('warning, no parameters file ')
    else:
        parameters.set_saved_parafile_path(options.para_file)

    # input = args[0]
    # output = args[1]
    # test(input,output)

    # test_get_buffer_polygon()

    test_add_one_field_records_to_shapefile()

    pass


if __name__=='__main__':
    usage = "usage: %prog [options] input_path output_file"
    parser = OptionParser(usage=usage, version="1.0 2016-10-26")
    parser.add_option("-p", "--para",
                      action="store", dest="para_file",
                      help="the parameters file")
    # parser.add_option("-s", "--used_file", action="store", dest="used_file",
    #                   help="the selectd used files,only need when you set --action=2")
    # parser.add_option('-o', "--output", action='store', dest="output",
    #                   help="the output file,only need when you set --action=2")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 2:
        parser.print_help()
        sys.exit(2)
    main(options, args)
