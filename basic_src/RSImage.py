#!/usr/bin/env python
# Filename: RSImage.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 05 May, 2016
"""

import sys,os,json,subprocess,numpy
# import basic
from basic_src import basic


# node: import gdal clobbering PATH environment variable on Ubuntu, add on 11 Nov 2020  gdal version 2.3.3
# https://github.com/OSGeo/gdal/issues/1231
try:
    from osgeo import ogr, osr, gdal
except:
    sys.exit('ERROR: cannot find GDAL/OGR modules')


def dependInit():
    basic.outputlogMessage('The version of gdal in Python environment is '\
                            '(maybe not be the same as the one on OS) :' +  gdal.__version__  )
    # this allows GDAL to throw Python Exceptions
    gdal.UseExceptions()

# example GDAL error handler function
def gdal_error_handler(err_class, err_num, err_msg):
    errtype = {
            gdal.CE_None:'None',
            gdal.CE_Debug:'Debug',
            gdal.CE_Warning:'Warning',
            gdal.CE_Failure:'Failure',
            gdal.CE_Fatal:'Fatal'
    }
    err_msg = err_msg.replace('\n',' ')
    err_class = errtype.get(err_class, 'None')
    print ('Error Number: %s' % (err_num))
    print ('Error Type: %s' % (err_class))
    print ('Error Message: %s' % (err_msg))

def test_error_handler():
     # install error handler
    gdal.PushErrorHandler(gdal_error_handler)

    # Raise a dummy error
    gdal.Error(1, 2, 'test error')

    #uninstall error handler
    gdal.PopErrorHandler()


class RSImageclass(object):
    """
    support remote sensing images reading and writting by GDAL
    """
    def __init__(self):
        self.imgpath = ''
        self.ds = None
        self.spatialrs = None
        self.geotransform = None
        dependInit()
    def __del__(self):
        # close dataset
        # print 'RSImageclass__del__'
        self.ds = None

    def open(self,imgpath):
        """
        open image file
        Args:
            imgpath: the path of image file

        Returns:True if succeessful, False otherwise

        """
        try:
            self.imgpath = imgpath
            self.ds = gdal.Open(imgpath)
        except RuntimeError as e:
            basic.outputlogMessage('Unable to open: '+ self.imgpath)
            basic.outputlogMessage(str(e))
            return False

        prj = self.ds.GetProjection()
        self.spatialrs = osr.SpatialReference(wkt=prj)

        basefilename = os.path.basename(self.imgpath).split('.')[0]
        self.geotransform = self.ds.GetGeoTransform()

        return True

    def New(self,imgpath,imgWidth, imgHeight,bandCount,GDALDatatype, _format='GTiff'):
        """
        New a image file by GDAL
        Args:
            imgpath: the path of image file
            imgWidth: image width
            imgHeight: image height
            bandCount: the bandcount
            GDALDatatype: datatype represented by GDAL
            _format: the image file format, default is Geotiff is not set the value

        Returns:True if succeessful, False otherwise

        """

        format = _format
        driver = gdal.GetDriverByName(format)
        metadata = driver.GetMetadata()
        if gdal.DCAP_CREATE in metadata.keys()  and metadata[gdal.DCAP_CREATE] == 'YES':
            basic.outputlogMessage( 'Driver %s supports Create() method.' % format)
        else:
            basic.outputlogMessage( 'Driver %s not supports Create() method.' % format)
            return False
        # if metadata.has_key(gdal.DCAP_CREATECOPY) and metadata[gdal.DCAP_CREATECOPY] == 'YES':
        #     syslog.outputlogMessage('Driver %s supports CreateCopy() method.' % format)
        # dst_ds = driver.CreateCopy(Outputtiff, dataset, 0)

        try:
            self.imgpath = imgpath
            self.ds = driver.Create(imgpath, imgWidth, imgHeight, bandCount,GDALDatatype)
        except RuntimeError as e:
            basic.outputlogMessage('Unable to create: '+ self.imgpath)
            basic.outputlogMessage(str(e))
            return False

        # for bandindex in range(0,image.GetBandCount()):
        #     bandobject = image.Getband(bandindex+1)
        #     banddata = bandobject.ReadRaster(0,0,image.GetWidth(), image.GetHeight(),image.GetWidth(), image.GetHeight(),datatype)
        #     #byte
        #     # if banddata is 1:
        #     #     bandarray = struct.unpack('B'*image.GetWidth()*image.GetHeight(), banddata)
        #     dst_ds.GetRasterBand(bandindex+1).WriteRaster(0,0,image.GetWidth(), image.GetHeight(),banddata,image.GetWidth(), image.GetHeight(),datatype)
        #
        #     dst_ds.SetGCPs(allgcps,projection)

        return True


    def ReadbandData(self,bandindex,xoff,yoff, width,height,gdalDatatype):
        if not self.ds is None:
            try:
                banddata = self.ds.GetRasterBand(bandindex).ReadRaster(xoff,yoff,width,height,width,height,gdalDatatype)
            except RuntimeError as e:
                basic.outputlogMessage('Unable band %d data: '%bandindex)
                basic.outputlogMessage(str(e))
                return False
            return banddata
        else:
            basic.outputlogMessage('Please Open file first')
            return False

    def WritebandData(self,bandindex,xoff,yoff, width,height,banddata,gdalDatatype):
        if not self.ds is None:
            try:
                self.ds.GetRasterBand(bandindex).WriteRaster(xoff,yoff,width, height,banddata,width, height,gdalDatatype)
            except RuntimeError as e:
                basic.outputlogMessage('Unable write band %d data: '%bandindex)
                basic.outputlogMessage(str(e))
                return False
            return True
        else:
            basic.outputlogMessage('Please Create file first')
            return False


    def GetGetDriverShortName(self):
        if not self.ds is None:
            return self.ds.GetDriver().ShortName
        else:
            return False
    def GetGetDriverLongName(self):
        if not self.ds is None:
            return self.ds.GetDriver().LongName
        else:
            return False

    def GetProjection(self):
        if not self.ds is None:
            return self.ds.GetProjection()
        else:
            return False
    def SetProjection(self,prj_wkt):
        if not self.ds is None:
            try:
                self.ds.SetProjection(prj_wkt)
            except RuntimeError as e:
                basic.outputlogMessage(str(e))
                return False
            return True
        else:
            return False

    def GetPROJCS(self):
        if not self.spatialrs is None:
            if self.spatialrs.IsProjected:
                return self.spatialrs.GetAttrValue('projcs')
            else:
                return False
        else:
            return False

    def GetGEOGCS(self):
        if not self.spatialrs is None:
            if self.spatialrs.IsProjected:
                return self.spatialrs.GetAttrValue('geogcs')
            else:
                return False
        else:
            return False

    def GetUTMZone(self):
        if not self.spatialrs is None:
            return self.spatialrs.GetUTMZone()
        else:
            return False

    def GetGeoTransform(self):
        if not self.ds is None:
            return self.ds.GetGeoTransform()
        else:
            return False
    def SetGeoTransform(self,geotransform):
        if not self.ds is None:
            try:
                self.ds.SetGeoTransform(geotransform)
            except RuntimeError as e:
                basic.outputlogMessage(str(e))
                return False
            return True
        else:
            return False

    def GetStartX(self):
        if not self.geotransform is None:
            return self.geotransform[0]
        else:
            return False
    def GetStartY(self):
        if not self.geotransform is None:
            return self.geotransform[3]
        else:
            return False

    def GetXresolution(self):
        if not self.geotransform is None:
            return self.geotransform[1]
        else:
            return False

    def GetYresolution(self):
        if not self.geotransform is None:
            return self.geotransform[5]
        else:
            return False

    def GetGDALDataType(self):
        if not self.ds is None:
            band1 = self.Getband(1)
            return band1.DataType
        else:
            return False

    def GetWidth(self):
        if not self.ds is None:
            return self.ds.RasterXSize
        else:
            return -1

    def GetHeight(self):
        if not self.ds is None:
            return self.ds.RasterYSize
        else:
            return -1

    def GetBandCount(self):
        if not self.ds is None:
            return self.ds.RasterCount
        else:
            return -1

    def GetMetadata(self):
        if not self.ds:
            basic.outputlogMessage('Please Open the file first')
            return False
        return self.ds.GetMetadata()

    def Getband(self,bandindex):
        if not self.ds:
            basic.outputlogMessage('Please Open the file first')
            return False
        bandindex = int(bandindex)
        try:
            srcband = self.ds.GetRasterBand(bandindex)
        except RuntimeError as e:
            # for example, try GetRasterBand(10)
            basic.outputlogMessage('Band ( %i ) not found' % bandindex)
            basic.outputlogMessage(str(e))
            return False
        return srcband

    def Getband_names(self):
        '''
        get the all the band names (description) in this raster
        Returns:

        '''
        if not self.ds:
            basic.outputlogMessage('Please Open the file first')
            return False
        names = [self.ds.GetRasterBand(idx+1).GetDescription() for idx in range(self.ds.RasterCount)]
        return names

    def set_band_name(self,bandindex, band_name):
        '''
        set band name (description)
        Args:
            name:

        Returns:

        '''
        if not self.ds:
            basic.outputlogMessage('Please Open the file first')
            return False
        return self.ds.GetRasterBand(bandindex).SetDescription(band_name)


    def GetBandNoDataValue(self,bandindex):
        if not self.ds is None:
            try:
                self.ds.GetRasterBand(bandindex).GetNoDataValue()
            except RuntimeError as e:
                basic.outputlogMessage('Unable get no data value for  band %d data: '%bandindex)
                basic.outputlogMessage(str(e))
                return False
            return True
        else:
            basic.outputlogMessage('Error,Please Open file first')
            return False

    def SetBandNoDataValue(self,bandindex,nodatavalue):
        if not self.ds is None:
            try:
                self.ds.GetRasterBand(bandindex).SetNoDataValue(nodatavalue)
            except RuntimeError as e:
                basic.outputlogMessage('Unable set no data value for  band %d data: '%bandindex)
                basic.outputlogMessage(str(e))
                return False
            return True
        else:
            basic.outputlogMessage('Please Create file first')
            return False

def get_image_max_min_value(imagepath):
    """
    get image first band max vlaue and min value
    Args:
        imagepath: image path

    Returns:(max value list, min value list) is successful, (False,False) otherwise

    """
    max_value = []
    min_value = []
    CommandString = 'gdalinfo -json  -stats ' + imagepath
    imginfo = basic.exec_command_string_output_string(CommandString)
    if imginfo is False:
        return False
    imginfo_obj = json.loads(imginfo)
    try:
        bands_info = imginfo_obj['bands']
        for band_info in bands_info:
            max_value.append(band_info['maximum'])
            min_value.append(band_info['minimum'])
        return (max_value,min_value)
    except KeyError:
        basic.outputlogMessage(str(KeyError))
        pass
    return (False, False)


def get_image_mean_value(imagepath):
    """
    get image first band max vlaue and min value
    Args:
        imagepath: image path

    Returns:(mean value list for each band) if successful, (False) otherwise

    """
    mean_value = []
    cmd_list = ['gdalinfo','-json','-stats', '-mm', imagepath]   # Force computation
    imginfo = basic.exec_command_args_list_one_string(cmd_list)
    # print imginfo
    if imginfo is False:
        return False
    imginfo_obj = json.loads(imginfo)
    try:
        bands_info = imginfo_obj['bands']
        for band_info in bands_info:
            mean_value.append(band_info['mean'])

        return mean_value
    except KeyError:
        basic.outputlogMessage(str(KeyError))
        pass
    return False

def get_image_histogram_oneband(image_path, band_idx=1):
    """
    get historgram of one band
    Args:
        image_path: image path
        band_idx: band index, start from 1

    Returns: hist_count (bucket count) ,hist_min, hist_max,hist_buckets

    """
    # -stats: Force computation if no statistics are stored in an image
    # -mm: Force computation of the actual min/max values for each band in the dataset.
    CommandString = 'gdalinfo -json -hist -mm -stats ' + image_path
    imginfo = basic.exec_command_string_output_string(CommandString)
    if imginfo is False:
        return False
    imginfo_obj = json.loads(imginfo)

    try:
        bands_info = imginfo_obj['bands']
        band_info = bands_info[band_idx -1 ]  # only care one band one
        histogram_info = band_info["histogram"]

        hist_count = histogram_info["count"]
        hist_min = histogram_info["min"]
        hist_max = histogram_info["max"]
        hist_buckets = histogram_info["buckets"]

        return hist_count,hist_min, hist_max,hist_buckets

        # hist_array = np.array(hist_buckets)
        # hist_x = np.linspace(hist_min, hist_max, hist_count)
        # hist_percent = 100.0 * hist_array / np.sum(hist_array)
        #
        # print(np.sum(hist_array))

    except KeyError:
        raise KeyError('parse keys failed')

def get_valid_pixel_count(image_path):
    """
    get the count of valid pixels (exclude no_data pixel)
    assume that the nodata value already be set
    Args:
        image_path: path

    Returns: the count

    """
    bucket_count, hist_min, hist_max, hist_buckets = get_image_histogram_oneband(image_path, 1)

    # make sure no_data already set in img_path
    valid_pixel_count = 0
    for count in hist_buckets:
        valid_pixel_count += count
    return valid_pixel_count

def get_valid_pixel_percentage(image_path,total_pixel_num=None):
    """
    get the percentage of valid pixels (exclude no_data pixel)
    assume that the nodata value already be set
    Args:
        image_path: path
        total_pixel_num: total pixel count, for example, the image only cover a portion of the area

    Returns: the percentage (%)

    """
    valid_pixel_count = get_valid_pixel_count(image_path)
    if total_pixel_num is None:
        # get image width and height
        img_obj = RSImageclass()
        if img_obj.open(image_path):
            width = img_obj.GetWidth()
            height = img_obj.GetHeight()
            valid_per = 100.0 * valid_pixel_count/ (width * height)
            return valid_per
    else:
        valid_per = 100.0 * valid_pixel_count / total_pixel_num
        return valid_per
    return False

def get_image_location_value(imagepath,x,y,xy_srs,bandindex):
    """
    get the image value of given location(x,y) in bandindex
    Args:
        imagepath:the image path which the information query
        x:x value
        y:y value
        xy_srs:the coordinate system of (x,y), the value is :pixel ,prj or lon_lat_wgs84
        bandindex:the bandindex of image want to query

    Returns:the certain value (float) of given location

    """
    coordinate = ''
    if xy_srs == 'pixel':
        coordinate = ' '
    elif xy_srs == 'prj':
        coordinate = ' -geoloc '
    elif xy_srs == 'lon_lat_wgs84':
        coordinate = ' -wgs84 '
    else:
        basic.outputlogMessage('input error: %s is not right'%xy_srs)
        assert  False


    command_str = 'gdallocationinfo  -valonly' + ' -b ' + str(bandindex) + coordinate \
    + ' ' +imagepath + ' ' + str(x) + ' ' + str(y)
    result = basic.exec_command_string_output_string(command_str)
    if result == "":
        raise ValueError('the command output is empty')
    try :
        result = float(result)
    except ValueError:
        raise ValueError('cannot convert: %s to float'%result)
    return result

def get_image_location_value_list(imagepath,x,y,xy_srs):
    """
    get the image value of given location(x,y) of all bands
    Args:
        imagepath:the image path which the information query
        x:x value
        y:y value
        xy_srs:the coordinate system of (x,y), the value is :pixel ,prj or lon_lat_wgs84

    Returns: a list containing values (string format) of all the bands

    """
    coordinate = ''
    if xy_srs == 'pixel':
        coordinate = ' '
    elif xy_srs == 'prj':
        coordinate = ' -geoloc '
    elif xy_srs == 'lon_lat_wgs84':
        coordinate = ' -wgs84 '
    else:
        basic.outputlogMessage('input error: %s is not right'%xy_srs)
        assert  False

    command_str = 'gdallocationinfo  -valonly '  + coordinate \
    + '  '+imagepath + ' ' + str(x) +' '+ str(y)
    result = basic.exec_command_string_output_string(command_str)
    if result == "":
        raise ValueError('the command output is empty')

    return result.split('\n')


def get_image_proj_extent(imagepath):
    """
    get the extent of a image
    Args:
        imagepath:image path

    Returns:(ulx:Upper Left X,uly: Upper Left Y,lrx: Lower Right X,lry: Lower Right Y)

    """
    ulx = False
    uly = False
    lrx = False
    lry = False
    CommandString = 'gdalinfo -json ' + imagepath
    imginfo = basic.exec_command_string_output_string(CommandString)
    if imginfo is False:
        return False
    imginfo_obj = json.loads(imginfo)
    # print imginfo_obj
    # print type(imginfo_obj)
    # print imginfo_obj.keys()
    try:
        cornerCoordinates = imginfo_obj['cornerCoordinates']
        upperLeft_value = cornerCoordinates['upperLeft']
        lowerRight_value = cornerCoordinates['lowerRight']
        ulx = upperLeft_value[0]
        uly = upperLeft_value[1]
        lrx = lowerRight_value[0]
        lry = lowerRight_value[1]
    except KeyError:
        basic.outputlogMessage(str(KeyError))
        pass

    return (ulx,uly,lrx,lry)

def get_image_latlon_centre(imagepath):
    centre_lat = False
    centre_lon = False
    (CornerLats, CornerLons) = GetCornerCoordinates(imagepath)
    centre_lat = CornerLats[4]
    centre_lon = CornerLons[4]
    return (centre_lat,centre_lon)

#codes from http://gis.stackexchange.com/
def GetCornerCoordinates(FileName):
    GdalInfo = subprocess.check_output('gdalinfo {}'.format(FileName), shell=True)
    # to string, not byte
    GdalInfo = GdalInfo.decode()
    GdalInfo = GdalInfo.splitlines() #split('/n') # Creates a line by line list.
    CornerLats, CornerLons = numpy.zeros(5), numpy.zeros(5)
    GotUL, GotUR, GotLL, GotLR, GotC = False, False, False, False, False
    for line in GdalInfo:
        if line[:10] == 'Upper Left':
            CornerLats[0], CornerLons[0] = GetLatLon(line)
            GotUL = True
        if line[:10] == 'Lower Left':
            CornerLats[1], CornerLons[1] = GetLatLon(line)
            GotLL = True
        if line[:11] == 'Upper Right':
            CornerLats[2], CornerLons[2] = GetLatLon(line)
            GotUR = True
        if line[:11] == 'Lower Right':
            CornerLats[3], CornerLons[3] = GetLatLon(line)
            GotLR = True
        if line[:6] == 'Center':
            CornerLats[4], CornerLons[4] = GetLatLon(line)
            GotC = True
        if GotUL and GotUR and GotLL and GotLR and GotC:
            break
    return CornerLats, CornerLons

def GetLatLon(line):
    coords = line.split(') (')[1]
    coords = coords[:-1]
    LonStr, LatStr = coords.split(',')
    # Longitude
    LonStr = LonStr.split('d')    # Get the degrees, and the rest
    LonD = int(LonStr[0])
    LonStr = LonStr[1].split('\'')# Get the arc-m, and the rest
    LonM = int(LonStr[0])
    LonStr = LonStr[1].split('"') # Get the arc-s, and the rest
    LonS = float(LonStr[0])
    Lon = LonD + LonM/60. + LonS/3600.
    if LonStr[1] in ['W', 'w']:
        Lon = -1*Lon
    # Same for Latitude
    LatStr = LatStr.split('d')
    LatD = int(LatStr[0])
    LatStr = LatStr[1].split('\'')
    LatM = int(LatStr[0])
    LatStr = LatStr[1].split('"')
    LatS = float(LatStr[0])
    Lat = LatD + LatM/60. + LatS/3600.
    if LatStr[1] in ['S', 's']:
        Lat = -1*Lat
    return Lat, Lon

def save_numpy_2d_array_to_image_tif(imagepath,array,datatype,geotransform,projection,nodata):
    if array.ndim !=2:
        basic.outputlogMessage('input error, only support 2-dimensional array')
        return False
    (height,width) = array.shape
    bandindex = 1
    bandcount = 1
    imagenew = RSImageclass()
    if not imagenew.New(imagepath,width,height,bandcount ,datatype):
        return False
    # templist = array.tolist()
    # band_str = struct.pack('%sf'%width*height,*templist)
    band_str = array.astype('f').tostring()
    if imagenew.WritebandData(bandindex,0,0,width,height,band_str,imagenew.GetGDALDataType()) is False:
        return False
    imagenew.SetBandNoDataValue(bandindex,nodata)
    imagenew.SetGeoTransform(geotransform)
    imagenew.SetProjection(projection)
    return True


def test_get_image_max_min_value():
    image_path = '/Users/huanglingcao/Data/getVelocityfromRSimage_test/pre_processing_saved/LC81400412015065LGN00_B8.TIF'
    # image_path = '/Users/huanglingcao/Data/getVelocityfromRSimage_test/pre_processing_saved/19900624_19900710_abs_m.jpg'
    get_image_max_min_value(image_path)


if __name__=='__main__':
    # test_error_handler()
    # open dataset
    # ds = gdal.Open('test.tif')
    #
    # # close dataset
    # ds = None
    print ('begin test')

    test_get_image_max_min_value()
    # sys.exit(0)


    rsimg = RSImageclass()

    # print rsimg.open('LE700801120083KS00_B8.TIF')

    if rsimg.open('LE70080112000083KIS00_B8.TIF'):
        metadata = rsimg.GetMetadata()
         # print metadata

        # print rsimg.Getband(1)
        # rsimg.testgetinfo()
        print (rsimg.GetGeoTransform())
        prj = rsimg.GetProjection()
        # src = osr.SpatialReference()
        # src.ImportFromWkt(prj)

        print (prj)
        # srs = osr.SpatialReference(wkt=prj)
        # print srs.GetUTMZone()
        #
        # if srs.IsProjected:
        #     print srs.GetAttrValue('projcs')
        #     print srs.GetAttrValue('geogcs')
        print (rsimg.GetUTMZone())
        print (rsimg.GetGEOGCS())
        print (rsimg.GetPROJCS())

        # result = rsimg.GetGetDriverShortName()
        result = rsimg.GetGetDriverLongName()
        if not result is False:
            result = result.upper()
            print (result)

    print ('end test')

