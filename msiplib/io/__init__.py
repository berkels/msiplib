from skimage.io import imread, imsave
import netCDF4 as nc4
import numpy as np
import scipy.sparse as sparse
import time
import sys
import os
import warnings
import traceback
import bz2
import re
import errno


class Tee:
    """Context manager that copies stdout and any exceptions to a log file

    Adapted from https://stackoverflow.com/a/57008553
    """
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def saveArrayAsNetCDF(data, filename, optimizeDataType: bool = False):
    # If the input data type is non-integer, check whether the
    # actual data is integer. If so, save as integer.
    if(optimizeDataType and (not np.issubdtype(data.dtype, np.integer)) and (not np.iscomplexobj(data)) and (np.all(np.mod(data, 1) == 0))):
        dataMin = np.amin(data)
        dataMax = np.amax(data)
        if(dataMin >= 0):
            if(dataMax <= np.iinfo(np.uint8).max):
                data = data.astype(np.uint8)
            elif(dataMax <= np.iinfo(np.uint16).max):
                data = data.astype(np.uint16)
            elif(dataMax <= np.iinfo(np.uint32).max):
                data = data.astype(np.uint32)
            elif(dataMax <= np.iinfo(np.uint64).max):
                data = data.astype(np.uint64)
        else:
            if((dataMin >= np.iinfo(np.int8).min) and (dataMax <= np.iinfo(np.int8).max)):
                data = data.astype(np.int8)
            elif((dataMin >= np.iinfo(np.int16).min) and (dataMax <= np.iinfo(np.int16).max)):
                data = data.astype(np.int16)
            elif((dataMin >= np.iinfo(np.int32).min) and (dataMax <= np.iinfo(np.int32).max)):
                data = data.astype(np.int32)
            elif((dataMin >= np.iinfo(np.int64).min) and (dataMax <= np.iinfo(np.int64).max)):
                data = data.astype(np.int64)
        print("Converted data type to {} before saving.".format(data.dtype))

    if (data.dtype == np.float64):
        dataType = 'd'
    elif (data.dtype == np.float32):
        dataType = 'f4'
    elif (data.dtype == np.int64):
        dataType = 'i8'
    elif (data.dtype == np.int32):
        dataType = 'i4'
    elif (data.dtype == np.int16):
        dataType = 'i2'
    elif (data.dtype == np.int8):
        dataType = 'i1'
    elif (data.dtype == np.uint16):
        dataType = 'u2'
    elif (data.dtype == np.uint8):
        dataType = 'u1'
    else:
        warnings.warn("Data type {} not explicitly handled, saving as double.".format(data.dtype))
        dataType = 'd'
    a = nc4.Dataset(filename, 'w', format='NETCDF4')
    if (data.ndim == 1):
        a.createDimension('x', data.shape[0])
        temp = a.createVariable('data', dataType, ('x'), zlib=True, complevel=5)
    elif (data.ndim == 2):
        a.createDimension('x', data.shape[0])
        a.createDimension('y', data.shape[1])
        temp = a.createVariable('data', dataType, ('x', 'y'),
                                zlib=True, complevel=5)
    elif (data.ndim == 3):
        a.createDimension('x', data.shape[0])
        a.createDimension('y', data.shape[1])
        a.createDimension('z', data.shape[2])
        temp = a.createVariable(
            'data', dataType, ('x', 'y', 'z'), zlib=True, complevel=5)
    else:
        raise NotImplementedError("Unsupported dimension")
    temp[:] = data
    a.history = 'Created ' + time.ctime(time.time())
    a.args = sys.argv
    a.close()


def readArrayFromNetCDF(filename, return_attributes=False):
    a = nc4.Dataset(filename, 'r')
    arr = a['data'][:]

    # arr is a masked array. If no entries in array are masked by the mask attribute,
    # remove the mask and return the raw numpy array
    return_array = arr if np.ma.is_masked(arr) else arr.data

    if return_attributes:
        attributes = {}
        for name in a.ncattrs():
            attributes[name] = getattr(a, name)
        return return_array, attributes
    else:
        return return_array


def append2DFloatArrayToNetCDF(filename, array_data, array_name):
    a = nc4.Dataset(filename, 'a')
    a.createDimension(array_name + '_dim_x', array_data.shape[0])
    a.createDimension(array_name + '_dim_y', array_data.shape[1])
    array_data_out = a.createVariable(array_name, float, (array_name + '_dim_x', array_name + '_dim_y'))
    array_data_out[:] = array_data
    a.close()


def appendLabelsToNetCDF(filename, label_array):
    a = nc4.Dataset(filename, 'a')
    a.createDimension('labels_dim', label_array.shape[0])
    labels = a.createVariable('labels', str, ('labels_dim',))
    labels[:] = label_array
    a.close()


def append_global_attribute_to_NetCDF(filename, attribute_name, attribute_value):
    a = nc4.Dataset(filename, 'a')
    setattr(a, attribute_name, attribute_value)
    a.close()


def loadFromQ2bz(path):
    filename, file_extension = os.path.splitext(path)

    if(file_extension == '.q2bz' or file_extension == '.q3bz' or file_extension == '.bz2'):
        fid = bz2.open(path, 'rb')
    else:
        fid = open(path, 'rb')
    # Read magic number
    line = fid.readline().rstrip().decode('ascii')
    if(line[0] == 'P'):
        dim = 2
    elif(line[0] == 'Q'):
        dim = 3
    else:
        raise ValueError("Invalid array header, doesn't start with 'P' or 'Q'")
    if(line[1] == '9'):
        dtype = np.float64
    elif(line[1] == '8'):
        dtype = np.float32
    elif(line[1] == '5'):
        dtype = np.uint8
    elif(line[1] == '2'):
        dtype = str
    else:
        dtype = None

    if not dtype:
        raise NotImplementedError("Invalid data type ({}), only float and double are supported currently".format(line[1]))
    # Skip comment lines (these either start with # or are empty, which means they start with \n)
    while chr(fid.peek()[0]) in ('#', '\n'):
        fid.readline().rstrip()

    # Read width and height
    arr = fid.readline().rstrip().split()
    width = int(arr[0])
    height = int(arr[1])
    depth = int(arr[2]) if dim == 3 else 1

    # Read max, but be careful not to read more than one "new line" character after max.
    # The binary data could start with a value that is equivalent to a
    # new line.
    max = ""
    while True:
        c = fid.read(1)
        if c == b'\n':
            break
        max = max + c.decode("utf-8")

    # Convert the string max to a number.
    if max.isdecimal():
        max = int(max)
    else:
        max = float(max)

    # Special handling for 16bit PGM files.
    if (dtype == np.uint8) and (max > 255):
        dtype = np.dtype(np.uint16)
        # Read data as big-endian.
        dtype = dtype.newbyteorder('>')

    # Read image to vector
    if dtype is str:
        s = fid.read()
        x = np.array(re.findall(b'[0-9P]+', s), dtype=np.uint8)
    else:
        x = np.frombuffer(fid.read(), dtype)
    img = x.reshape(depth, height, width) if dim == 3 else x.reshape(height, width)
    return img


def read_image(filename: str, as_gray: bool = False) -> np.ndarray:
    '''Reads an image file, i.e. a 2D array of values, while the format is determined from the filename suffix.
       Suffixes not explicitly handled are passed to `skimage.io.imread`'''

    filename = os.path.expandvars(filename)

    if filename.endswith('.nc'):
        if as_gray:
            raise NotImplementedError('as_gray reading not implemented for .nc files')
        return readArrayFromNetCDF(filename)
    elif filename.endswith('.q2') or filename.endswith('.q2bz') or filename.endswith('.dat.bz2'):
        if as_gray:
            raise NotImplementedError('as_gray reading not implemented for .q2 files')
        return loadFromQ2bz(filename)
    else:
        return imread(filename, as_gray=as_gray)


def write_image(filename: str, image: np.ndarray) -> None:
    '''Writes an image, i.e. a 2D array of values, to a file while the format is determined from the filename suffix.
       Suffixes not explicitly handled are passed to `skimage.io.imsave`'''
    if filename.endswith('.nc'):
        saveArrayAsNetCDF(image, filename)
    elif filename.endswith('.q2'):
        # saveAsQ2 automatically appends the suffix, so we have to remove it from the filename here.
        saveAsQ2(image, filename[:-3])
    else:
        imsave(filename, image)


def saveAsQ2(img, pathWithoutFileExtension):
    fid = open(pathWithoutFileExtension + ".q2", "wb")
    max_value = int(np.round(np.amax(img)))
    if img.dtype == np.uint8:
        type_char = "5"
        type_string = "BINARY"
        max_value = 255
        type_out = np.uint8
    elif img.dtype == np.uint16:
        type_char = "a"
        type_string = "UNSIGNED SHORT BINARY"
        max_value = 65535
        type_out = np.uint16
    elif img.dtype == np.float32:
        type_char = "8"
        type_string = "RAW FLOAT"
        type_out = np.float32
    else:
        type_char = "9"
        type_string = "RAW DOUBLE"
        type_out = np.float64

    fid.write(bytes("P{}\n".format(type_char), encoding="utf-8"))
    fid.write(
        bytes(
            "# This is a QuOcMesh file of type {} (={}) written {} by Python\n".format(
                type_char, type_string, time.ctime(time.time())
            ),
            encoding="utf-8",
        )
    )
    fid.write(bytes("{} {}\n".format(img.shape[1], img.shape[0]), encoding="utf-8"))
    fid.write(bytes("{}\n".format(max_value), encoding="utf-8"))
    img.astype(type_out).tofile(fid)
    fid.close()


def printSparseMatLikeMatlab(A):
    coo = sparse.coo_matrix(A)
    for i in range(coo.data.shape[0]):
        row = coo.row[i] + 1
        col = coo.col[i] + 1
        print('{}({},{}){}{:9.4f}'.format(' '*max((4-len(str(row))), 0), row, col,
                                          ' '*max((5-len(str(col))), 0), coo.data[i]))


def printVectorLikeMatlab(v):
    for i in range(v.shape[0]):
        print('{:10.4f}'.format(v[i]))


def check_path(path):
    """
    checks and creates path (borrowed from StackOverflow)
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path
