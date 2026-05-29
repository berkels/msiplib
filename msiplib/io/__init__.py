import netCDF4 as nc4
import numpy as np
import time
import sys
import os
import warnings
import traceback
import bz2
import re
import errno


class Tee:
    """
    Context manager that copies stdout and any exceptions to a log file.

    Adapted from https://stackoverflow.com/a/57008553
    """

    def __init__(self, filename):
        self.file = open(filename, "w")
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


def saveArrayAsNetCDF(data, filename: str | os.PathLike, optimizeDataType: bool = False):
    # If the input data type is non-integer, check whether the
    # actual data is integer. If so, save as integer.
    if (
        optimizeDataType
        and (not np.issubdtype(data.dtype, np.integer))
        and (not np.iscomplexobj(data))
        and (np.all(np.mod(data, 1) == 0))
    ):
        dataMin = np.amin(data)
        dataMax = np.amax(data)
        if dataMin >= 0:
            if dataMax <= np.iinfo(np.uint8).max:
                data = data.astype(np.uint8)
            elif dataMax <= np.iinfo(np.uint16).max:
                data = data.astype(np.uint16)
            elif dataMax <= np.iinfo(np.uint32).max:
                data = data.astype(np.uint32)
            elif dataMax <= np.iinfo(np.uint64).max:
                data = data.astype(np.uint64)
        else:
            if (dataMin >= np.iinfo(np.int8).min) and (dataMax <= np.iinfo(np.int8).max):
                data = data.astype(np.int8)
            elif (dataMin >= np.iinfo(np.int16).min) and (dataMax <= np.iinfo(np.int16).max):
                data = data.astype(np.int16)
            elif (dataMin >= np.iinfo(np.int32).min) and (dataMax <= np.iinfo(np.int32).max):
                data = data.astype(np.int32)
            elif (dataMin >= np.iinfo(np.int64).min) and (dataMax <= np.iinfo(np.int64).max):
                data = data.astype(np.int64)
        print(f"Converted data type to {data.dtype} before saving.")

    if data.dtype == np.float64:
        dataType = "d"
    elif data.dtype == np.float32:
        dataType = "f4"
    elif data.dtype == np.int64:
        dataType = "i8"
    elif data.dtype == np.int32:
        dataType = "i4"
    elif data.dtype == np.int16:
        dataType = "i2"
    elif data.dtype == np.int8:
        dataType = "i1"
    elif data.dtype == np.uint16:
        dataType = "u2"
    elif data.dtype == np.uint8:
        dataType = "u1"
    else:
        warnings.warn(f"Data type {data.dtype} not explicitly handled, saving as double.")
        dataType = "d"
    try:
        a = nc4.Dataset(filename, "w", format="NETCDF4")
        # If fill_value is not set to False below, entries that stay at fill_value after the data
        # is written to the variable will become masked. For instance, if we save as uint8,
        # fill_value is 255, so any "255"s in the input data would become masked in the stored file.
        if data.ndim == 1:
            a.createDimension("x", data.shape[0])
            temp = a.createVariable("data", dataType, ("x"), zlib=True, complevel=5, fill_value=False)
        elif data.ndim == 2:
            a.createDimension("x", data.shape[0])
            a.createDimension("y", data.shape[1])
            temp = a.createVariable("data", dataType, ("x", "y"), zlib=True, complevel=5, fill_value=False)
        elif data.ndim == 3:
            a.createDimension("x", data.shape[0])
            a.createDimension("y", data.shape[1])
            a.createDimension("z", data.shape[2])
            temp = a.createVariable("data", dataType, ("x", "y", "z"), zlib=True, complevel=5, fill_value=False)
        else:
            raise NotImplementedError("Unsupported dimension")
        temp[:] = data
        a.history = "Created " + time.ctime(time.time())
        a.args = sys.argv
        a.close()
    except PermissionError:
        print(f"Error writing to {filename}.")


def readArrayFromNetCDF(filename: str | os.PathLike | bytes, return_attributes=False):
    if isinstance(filename, str):
        a = nc4.Dataset(filename, "r")
    elif isinstance(filename, bytes):
        a = nc4.Dataset("inmemory.nc", memory=filename)
    else:
        raise ValueError(f"Unexpected type of argument 'filename': {type(filename)}")
    arr = a["data"][:]

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


def append2DFloatArrayToNetCDF(filename: str | os.PathLike, array_data, array_name):
    a = nc4.Dataset(filename, "a")
    a.createDimension(array_name + "_dim_x", array_data.shape[0])
    a.createDimension(array_name + "_dim_y", array_data.shape[1])
    array_data_out = a.createVariable(array_name, float, (array_name + "_dim_x", array_name + "_dim_y"))
    array_data_out[:] = array_data
    a.close()


def appendLabelsToNetCDF(filename: str | os.PathLike, label_array):
    a = nc4.Dataset(filename, "a")
    a.createDimension("labels_dim", label_array.shape[0])
    labels = a.createVariable("labels", str, ("labels_dim",))
    labels[:] = label_array
    a.close()


def append_global_attribute_to_NetCDF(filename: str | os.PathLike, attribute_name, attribute_value):
    a = nc4.Dataset(filename, "a")
    try:
        setattr(a, attribute_name, attribute_value)
    except AttributeError as exception:
        print(f"AttributeError: {exception}")
    except TypeError as exception:
        print(f"TypeError: {exception}")
    a.close()


def loadFromQ2bz(path: str | os.PathLike) -> np.ndarray:
    filename, file_extension = os.path.splitext(path)

    if file_extension == ".q2bz" or file_extension == ".q3bz" or file_extension == ".bz2":
        fid = bz2.open(path, "rb")
    else:
        fid = open(path, "rb")
    # Read magic number
    line = fid.readline().rstrip().decode("ascii")
    if line[0] == "O":
        dim = 1
    elif line[0] == "P":
        dim = 2
    elif line[0] == "Q":
        dim = 3
    else:
        raise ValueError("Invalid array header, doesn't start with 'O', 'P' or 'Q'")
    if line[1] == "9":
        dtype = np.float64
    elif line[1] == "8":
        dtype = np.float32
    elif line[1] == "5":
        dtype = np.uint8
    elif line[1] == "2":
        dtype = str
    else:
        dtype = None

    if not dtype:
        raise NotImplementedError(f"Invalid data type ({line[1]}), only float and double are supported currently")
    # Skip comment lines (these either start with # or are empty, which means they start with \n)
    while chr(fid.peek()[0]) in ("#", "\n"):
        fid.readline().rstrip()

    # Read width and height
    arr = fid.readline().rstrip().split()
    width = int(arr[0])
    height = int(arr[1]) if dim >= 2 else 1
    depth = int(arr[2]) if dim == 3 else 1

    # Read max, but be careful not to read more than one "new line" character after max.
    # The binary data could start with a value that is equivalent to a
    # new line.
    max = ""
    while True:
        c = fid.read(1)
        if c == b"\n":
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
        dtype = dtype.newbyteorder(">")

    # Read image to vector
    if dtype is str:
        s = fid.read()
        x = np.array(re.findall(b"[0-9P]+", s), dtype=np.uint8)
    else:
        x = np.frombuffer(fid.read(), dtype)
    if dim == 1:
        return x
    img = x.reshape(depth, height, width) if dim == 3 else x.reshape(height, width)
    return img


def read_image(filename: str | os.PathLike, as_gray: bool = False) -> np.ndarray:
    """
    Reads an image file, i.e. a 2D array of values, while the format is determined from
    the filename suffix.

    Suffixes not explicitly handled are passed to `skimage.io.imread`
    """
    filename=os.fspath(filename)
    filename = os.path.expanduser(os.path.expandvars(filename))

    if filename.endswith(".nc"):
        data = readArrayFromNetCDF(filename)
        if as_gray and data.ndim != 2:
            raise NotImplementedError("as_gray reading not implemented for .nc files")
        else:
            return data
    elif filename.endswith(".q2") or filename.endswith(".q2bz") or filename.endswith(".dat.bz2") or (filename.endswith(".dat") and (open(filename, 'rb').read(2) == b"P9")):
        data = loadFromQ2bz(filename)
        if as_gray and data.ndim != 2:
            raise NotImplementedError("as_gray reading not implemented for .q2 files")
        else:
            return data
    else:
        from skimage.io import imread
        return imread(filename, as_gray=as_gray)


def write_image(filename: str | os.PathLike, image: np.ndarray) -> None:
    """
    Writes an image, i.e. a 2D array of values, to a file while the format is determined
    from the filename suffix.

    Parameters:
        filename : str or os.PathLike
            The path to the file to write.
        image : np.ndarray
            The image data to write.

    Suffixes not explicitly handled are passed to `skimage.io.imsave`
    """
    
    filename=os.fspath(filename)
    if filename.endswith(".nc"):
        saveArrayAsNetCDF(image, filename)
    elif filename.endswith(".q2"):
        # saveAsQ2 automatically appends the suffix, so we have to remove it from the filename here.
        saveAsQ2(image, filename[:-3])
    else:
        from skimage.io import imsave
        imsave(filename, image)


def saveAsQ2(img, pathWithoutFileExtension: str):
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

    fid.write(bytes(f"P{type_char}\n", encoding="utf-8"))
    fid.write(
        bytes(
            f"# This is a QuOcMesh file of type {type_char} (={type_string}) written {time.ctime(time.time())} by Python\n",
            encoding="utf-8",
        )
    )
    fid.write(bytes(f"{img.shape[1]} {img.shape[0]}\n", encoding="utf-8"))
    fid.write(bytes(f"{max_value}\n", encoding="utf-8"))
    img.astype(type_out).tofile(fid)
    fid.close()


def printSparseMatLikeMatlab(A):
    from scipy.sparse import coo_matrix
    coo = coo_matrix(A)
    for i in range(coo.data.shape[0]):
        row = coo.row[i] + 1
        col = coo.col[i] + 1
        print(f"{' ' * max((4 - len(str(row))), 0)}({row},{col}){' ' * max((5 - len(str(col))), 0)}{coo.data[i]:9.4f}")


def printVectorLikeMatlab(v):
    for i in range(v.shape[0]):
        print(f"{v[i]:10.4f}")


def check_path(path: str | os.PathLike):
    """
    Ensures that the specified path exists, cf.

    https://stackoverflow.com/a/22718321
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path
