""" metadata.py

File containing implementation of MetaData object for SIGMOID.

MetaData objects are a critical component of data handling: they
store column information in such a way that it can be read/written
to/fropm HDF5 files, which is the default format to load data into
the machine learning pipeline of SIGMOID.

MetaData object contain several key attributes:

 - original index of column (as in the raw dataframe)
 - type of column (numerical, categorical, etc)
 - length of the transformed column (e.g. for one-hot encoding)
 - id of associated transformation
 - transformation arguments and parameters

This information is represented as numpy.ndarrays which allows for
persistent storage within the HDF5 file itself, making the Cache
(see coordinate_files.py) a self-contained representation of the data.
"""

import copy
from collections import OrderedDict


class MetaData(OrderedDict):
    """ Object to handle column metadata.

        Metadata is stored using the attributes capability of HDF5
        files, and as such it must be stored as vectorized representation.
        This implementation takes care of "translating" this vectorized
        format to a "human readable" representation.

        MetaData inherits from OrderedDict.
    """
    def __init__(self, max_buffer_size: int = 1024):
        """ Initializer of meta data object.

            @param max_buffer_size: int (defaults to 100)
                maximum *combined* number of arguments and parameters
                used by a column transformation routine.
        """
        super(MetaData, self).__init__()
        self.buffer_size_ = max_buffer_size
        self.pos_col_index_ = 0
        self.pos_col_type_ = 1
        self.pos_trafo_col_size_ = 2
        self.pos_trafo_n_params_ = 3
        self.pos_trafo_n_args_ = 4
        self.pos_begin_data_ = 10

    def from_hdf5_attributes(self, attr):
        """ Loads metadata information from h5py attribute object.

            @param attr: h5py.File.attr instance.
        """
        self.update(attr)

    def to_dict(self) -> OrderedDict:
        """ Returns deep-copy of object as OrderedDict.
        """
        return copy.deepcopy(self)

    def get_columns(self):
        """ Returns available column names.
        """
        return list(self.keys())

    def add_column(self,
                   col_name: str, col_index: int, col_type: str,
                   col_transform=None):
        """ Adds column information to metadata object.

            @param col_name: str
                name of the column being added
            @param col_index: int
                position of column in the original representation,
                usually a pandas.DataFrame.
            @param col_type: str
                type of column. Valid types are 'float', 'integer',
                'categorical', 'binary' and 'datetime'.
            @param col_transform: ColumnTransformation (defaults to None)
                instance of a ColumnTransformation object
        """
        # arguments passed to column transformation
        trafo_args = {}
        # parameters passed to column transformation
        trafo_params = {}
        # size of transformed column
        trafo_size = 1
        if col_transform is not None:
            # transformation length
            trafo_size = col_transform.get_length()
            # transformation parameters as numpy.ndarray
            trafo_params = col_transform.get_parameters()
            # transformation arguments as numpy.ndarray
            trafo_args = col_transform.get_arguments()

        # buffer to store column information
        data = [''] * self.buffer_size_
        data[self.pos_col_index_] = str(col_index)
        data[self.pos_col_type_] = str(col_type)
        data[self.pos_trafo_col_size_] = str(trafo_size)
        data[self.pos_trafo_n_params_] = str(len(trafo_params))
        data[self.pos_trafo_n_args_] = str(len(trafo_args))

        # store transformation arguments
        offset = self.pos_begin_data_
        offset += 1
        for arg_name, arg_data in trafo_args.items():
            data[offset] = str(len(arg_name))
            offset += 1
            data[offset] = str(len(arg_data))
            offset += 1
            for c in arg_name:
                data[offset] = str(c)
                offset += 1
            for d in arg_data:
                if type(d) == str:
                    data[offset] = d
                else:
                    data[offset] = "{:.6f}".format(float(arg_data))
                offset += 1
        # store transformation parameters
        for param_name, param_data in trafo_params.items():
            param_size = len(param_data)
            data[offset] = str(len(param_name))
            offset += 1
            data[offset] = str(len(param_data))
            offset += 1
            for c in param_name:
                data[offset] = str(c)
                offset += 1
            for i in range(param_size):
                data[offset] = str(param_data[i])
                offset += 1
        # build buffer
        bytebuffer = ''
        for d in data:
            bytebuffer += d + '\t'
        # consolidate into OrderedDict as byte-buffer
        self[col_name] = bytebuffer  #.encode()

    def get_column_index(self, col_name: str):
        """ Returns index of column.
        """
        r = self[col_name].split('\t')[self.pos_col_index_]
        return int(r)

    def get_column_type(self, col_name: str):
        """ Returns column type as string.
        """
        data = self[col_name].split('\t')
        r = data[self.pos_col_type_]
        return r

    def get_transformed_column_length(self, col_name: str):
        """ Returns column "length".

            Here, length is used to refer to the number of
            elements in a column. For instance, a categorical
            column that is encoded has a one-hot vector and has
            n categories will have a length of n.
        """
        r = self[col_name].split('\t')[self.pos_trafo_col_size_]
        return int(r)

    def get_parameter_data(self, col_name: str, return_offset: bool = False) -> dict:
        """ Return list with transformation parameter.
            @param col_name: str
                name of the column
            @param return_offset: optional, bool
                whether to return the offset in the buffer.
        """
        data = self[col_name].split('\t')
        n_param = int(data[self.pos_trafo_n_params_])
        # get offset to load parameter data
        args, offset = self.get_argument_data(col_name, return_offset=True)
        # reconstruct parameters
        parameters = {}
        for _ in range(0, n_param):
            param_name_len = int(data[offset])
            offset += 1
            param_size = int(data[offset])
            offset += 1

            param_chars = []
            for c in data[offset:offset + param_name_len]:
                param_chars.append(c)
                offset += 1
            param_name = ''.join(param_chars)

            param_data = []
            for __ in range(param_size):
                r = None
                try:
                    r = float(data[offset])
                except ValueError:
                    r = data[offset]
                param_data.append(r)
                offset += 1
            parameters[param_name] = param_data

        if return_offset:
            return parameters, offset
        return parameters

    def get_argument_data(self, col_name: str, return_offset: bool = False) -> dict:
        """ Returns list with transformation arguments.
        """
        data = self[col_name].split('\t')
        offset = self.pos_begin_data_
        n_args = int(data[self.pos_trafo_n_args_])

        offset += 1
        arguments = {}
        for _ in range(0, n_args):
            arg_name_len = int(data[offset])
            offset += 1
            arg_size = int(data[offset])
            offset += 1

            arg_chars = []
            for c in data[offset:offset + arg_name_len]:
                arg_chars.append(c)
                offset += 1
            arg_name = ''.join(arg_chars)

            arg_data = []
            for __ in range(arg_size):
                r = None
                try:
                    r = float(data[offset])
                except ValueError:
                    r = data[offset]
                arg_data.append(r)
                offset += 1
            arguments[arg_name] = arg_data

        if return_offset:
            return arguments, offset

        return arguments

