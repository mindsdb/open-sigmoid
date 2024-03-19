"""
coordinate_files.py

File containing the implementation of data cache and dataset.

A cache is an abstraction that enables data to be stored
in binary format so as to be read efficiently by data loaders
(e.g. PyTorch DataLoader) in both local and distributed environments.
It basically serves the same purpose of a DataFrame, with the added
capabililty of performing and storing data transformations and
information about the data types. As such, cache objects contain
a purely numerical representation of the data, and hold information
about what is being stored, in particular, the type of data and the
way it is presented to the data loader.
"""
import os
import copy
import json

import h5py
import numpy
import pandas

import torch

from torch.utils.data import Dataset as torch_Dataset
from torch.utils.data import DataLoader as torch_DataLoader
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from sigmoid.local.preprocessing.metadata import MetaData
from sigmoid.local.preprocessing.cleaners import ColumnCleaner
from sigmoid.local.preprocessing.transformations import ColumnTransform

from sigmoid.local.preprocessing.transformations import NormalizeToOneZero
from sigmoid.local.preprocessing.transformations import CategoricalAsOneHot


class LocalCache:
    """
    Cache object for non-distributed environments.
    """

    def __init__(self, dataframe: pandas.DataFrame,
                 target: str,
                 ignore: list) -> None:
        """ Initializes cache object.

        @param dataframe: pandas.DataFrame
            instance of pandas.DataFrame with data.
        @param target: str
            name of target column
        @param ignore: list (defalts to empty)
            list of column names from dataframe that must be ignored
        """
        self.target_ = target
        self.ignore_ = ignore
        self.data_ = dataframe.drop(ignore, axis=1)
        self.x_data_ = pandas.DataFrame()
        self.y_data_ = pandas.DataFrame()

        self.n_rows = dataframe.shape[0]

        self.x_meta_data_ = MetaData()
        self.y_meta_data_ = MetaData()
        self.trafos_ = {}
        self.cleaners_ = {}
        self.col_types_ = {}
        self.train_splits_ = []
        self.test_splits_ = []

    def load_column_types(self, path: str) -> None:
        """ Reads column types from JSON.

            :param path (str)
                system path to JSON file with column information.

            :note
                This routine must be called before any transformation
                is attached to the cache.
        """
        column_type_info = {}
        with open(path, 'r', encoding='utf-8') as f:
            column_type_info = json.load(f)

        for col_name, info in column_type_info.items():
            self.col_types_[col_name] = info['type']

    def attach_type_transformation(self,
                                   apply_to_type: str,
                                   trafo: ColumnTransform,
                                   cleaner: ColumnCleaner,
                                   **trafo_args) -> None:
        """ Set transform for specified data type.

            @param trafo: column transformation type
                A column transformation (see transformations.py)
                This argument must be passed as a type, NOT as an instance.
            @param apply_to_type: str
                Column type to apply the transformation to.
                See `type_infer` for supported types.
            @param trafo_args: dict (defaults to empty dict)
                dictionary with transformation arguments.
        """
        for colname, coltype in self.col_types_.items():
            # ignore forced transformations
            if colname in self.trafos_:
                continue
            if coltype == apply_to_type:
                self.trafos_[colname] = trafo(**trafo_args)
                self.cleaners_[colname] = cleaner()

    def attach_column_transformation(self, apply_to_col: str,
                                     trafo: ColumnTransform,
                                     cleaner: ColumnCleaner,
                                     **trafo_args) -> None:
        """ Set transform for specified column.

            @param apply_to_col: str
                name of column to apply transformation to.
            @param trafo: ColumnTransform
                type of column transformation to apply.
                Argument must be passed as a type, NOT as instance.
            @param trafo_args:
                positional arguments of specific transform.

            This function overrides `attach_type_transformation` for
            the specific column.
        """
        for colname in self.col_types_:
            if colname == apply_to_col:
                self.trafos_[colname] = trafo(**trafo_args)
                self.cleaners_[colname] = cleaner()

    def get_column_transform(self, col_name: str) -> ColumnTransform:
        """ Returns transformation associated to a column.

            @param col_name: str
                name of the column to retrieve transformation from.

            @raises:
                ValueError if column `col_name` does not have a transform
                attached to it.
        """
        if col_name not in self.trafos_:
            raise ValueError(f"column {col_name} does not have a transform.")

        return self.trafos_[col_name]

    def transform(self) -> None:
        """ Applies transformations to specified columns.

            This function must be called *after* `load_types()`.

            The transformed columns are stored into internal DataFrame
            `x_data` and `y_data`. While the order is preserved, names
            might vary depending on how the transform renames the columns.
        """
        # transform column-wise
        trafo_dataframes = []
        for col_name in self.data_.columns:
            col_data = self.data_[col_name]
            if col_name in self.trafos_:
                # clean
                col_data = self.cleaners_[col_name](col_data)
                # transform
                col_data = self.trafos_[col_name](col_data)
            trafo_dataframes.append(col_data)

        trafo_data = pandas.concat(trafo_dataframes, axis=1)

        y_cols = [c for c in trafo_data.columns if c.startswith(self.target_)]

        self.y_data_ = trafo_data.filter(regex=f"^{self.target_}_.*")
        self.x_data_ = trafo_data.drop(y_cols, axis=1)

    def populate_metadata(self) -> None:
        """ Writes column information to internal MetaData object.

            This routine must be called *after* `transform()` in order
            for the MetaData object to track transformed types correctly.
        """
        # input meta-data
        running_index = 0
        for col_name in self.data_.columns:
            if col_name == self.target_:
                continue
            # get basic column information
            coltype = self.col_types_[col_name]
            # get column transformation (None is valid)
            coltrafo = self.trafos_.get(col_name, None)
            self.x_meta_data_.add_column(col_name,
                                         running_index,
                                         coltype, coltrafo)
            running_index += 1
        # output (target) meta-data
        coltype = self.col_types_[self.target_]
        coltrafo = self.trafos_.get(self.target_, None)
        self.y_meta_data_.add_column(self.target_, 0, coltype, coltrafo)

    def build_splits(self,
                     splits: list = [{'global_fraction': 1.0,
                                      'val_fraction': 0.2}],
                     random_state: int = 42) -> None:
        """ Performs stratified splitting of the data.

            @param splits: list of dictionaries
                must have at least one entry with keywords

                'val_fraction':
                    floating point number between 0.0 and 1.0
                    to specify the fraction of the dataset used
                    for validation. Defaults to 0.2

                'global_fraction':
                    as 'val_fraction', but specifying the fraction
                    of the dataset used in global terms (for training
                    and validation) Defaults to 1.0

            This routine takes into account different class weights when
            performing the split by applying a stratified sampling scheme
            that preserves the probability distribution of the data for
            categorical and binary targets. For numerical targets, uniform
            sampling is used.
        """
        X = self.x_data_
        Y = self.y_data_
        target = self.y_meta_data_.get_columns()[0]
        target_type = self.y_meta_data_.get_column_type(target)
        if target_type == 'categorical':
            for split_info in splits:
                val_fraction = split_info.get('val_fraction', 0.2)
                global_fraction = split_info.get('global_fraction', 1.0)

                idx_cutoff = int(len(X) * global_fraction)
                Xs = X[:idx_cutoff]
                Ys = Y[:idx_cutoff]

                msss = \
                    MultilabelStratifiedShuffleSplit(n_splits=1,
                                                     test_size=val_fraction,
                                                     random_state=random_state)
                for train_index, test_index in msss.split(Xs, Ys):
                    self.train_splits_.append(train_index)
                    self.test_splits_.append(test_index)
        elif target_type == 'binary':
            for split_info in splits:
                val_fraction = split_info.get('val_fraction', 0.2)
                global_fraction = split_info.get('global_fraction', 1.0)

                idx_cutoff = int(len(X) * global_fraction)
                Xs = X[:idx_cutoff]
                Ys = Y[:idx_cutoff]
                mss = StratifiedShuffleSplit(n_splits=1,
                                             test_size=val_fraction,
                                             random_state=random_state)
                for train_index, test_index in mss.split(Xs, Ys):
                    self.train_splits_.append(train_index)
                    self.test_splits_.append(test_index)
        else:
            for split_info in splits:
                val_fraction = split_info.get('val_fraction', 0.2)
                global_fraction = split_info.get('global_fraction', 1.0)
                idx_cutoff = int(len(X) * global_fraction)
                idx = numpy.arange(0, idx_cutoff)
                train_index, test_index = train_test_split(
                                            idx,
                                            test_size=val_fraction,
                                            random_state=random_state)
                self.train_splits_.append(train_index)
                self.test_splits_.append(test_index)

    def to_hdf5(self, path: str) -> None:
        """ Writes data to HDF5 file.

            @param path: str
                system path to file.

            Warning: this routine overwrites the file if it already exists.

            Resulting HDF5 file has two datasets called 'x' and 'y'.
            Dataset 'x' holds input data, and dataset 'y' holds target data.
            The metadata for each dataset is attached to `attrs`.
            as well.
        """
        if os.path.isfile(path):
            h5file = h5py.File(path, mode='r')
            h5file.close()
        h5file = h5py.File(path, mode='w')
        # write raw data
        x_data = self.x_data_.values
        y_data = self.y_data_.values
        h5file.create_dataset('x', x_data.shape, data=x_data, track_order=True)
        h5file.create_dataset('y', y_data.shape, data=y_data, track_order=True)
        # write meta-data
        h5file['x'].attrs.update(self.x_meta_data_)
        h5file['y'].attrs.update(self.y_meta_data_)
        # write splits
        i = 0
        for train_idx, test_idx in zip(self.train_splits_, self.test_splits_):
            h5file.create_dataset(f'train_split_{i}', train_idx.shape,
                                  data=train_idx)
            h5file.create_dataset(f'test_split_{i}', test_idx.shape,
                                  data=test_idx)
            i = i + 1
        h5file.attrs['n_splits'] = numpy.asarray([i, ])
        # close file
        h5file.close()

    @classmethod
    def from_hdf5(cls, path: str):
        """ Reads cache from HDF5 file.

            Internal dataframe reference is lost when loading a cache from
            HDF5, as saving the data frame in this format is not supported
            given the possibility of having heterogenous data types.

            @param path: str
                system path to file.
        """
        h5file = h5py.File(path, mode='r', swmr=True)
        # instantiate dummy cache
        self = cls(pandas.DataFrame(), 'unknown', ignore=[])
        # write raw data
        self.x_data_ = h5file['x']
        self.y_data_ = h5file['y']
        # read meta-data
        self.x_meta_data_.from_hdf5_attributes(h5file['x'].attrs)
        self.y_meta_data_.from_hdf5_attributes(h5file['y'].attrs)
        # recover target name
        self.target_ = next(iter(self.y_meta_data_))
        # read splits
        n_splits = int(h5file.attrs['n_splits'][0])
        for i in range(n_splits):
            self.train_splits_.append(h5file[f'train_split_{i}'])
            self.test_splits_.append(h5file[f'test_split_{i}'])
        # load transformations applied to input data
        for col_name in self.x_meta_data_:
            trafo_args = self.x_meta_data_.get_argument_data(col_name)
            trafo_params = self.x_meta_data_.get_parameter_data(col_name)
            trafo_name = ''.join(trafo_args['name'])
            # TODO: calling `eval` is dangerous. This must be changed
            #       by a safer method!
            trafo = eval(f'{trafo_name}()')
            trafo.from_metadata(trafo_args, trafo_params)
            #for arg_name, arg_value in trafo_args.items():
            #    trafo.set_argument(arg_name, arg_value)
            # for param_name, param_value in trafo_params.items():
            #     trafo.set_parameter(param_name, param_value)
            self.trafos_[col_name] = trafo
        # load transformations applied to target data
        for col_name in self.y_meta_data_:
            trafo_args = self.y_meta_data_.get_argument_data(col_name)
            trafo_params = self.y_meta_data_.get_parameter_data(col_name)
            trafo_name = ''.join(trafo_args['name'])
            # TODO: calling `eval` is dangerous. This must be changed
            #       by a safer method!
            trafo = eval(f'{trafo_name}()')
            trafo.from_metadata(trafo_args, trafo_params)
            # for arg_name, arg_value in trafo_args.items():
            #     trafo.set_argument(arg_name, arg_value)
            # for param_name, param_value in trafo_params.items():
            #     trafo.set_parameter(param_name, param_value)
            self.trafos_[col_name] = trafo

        return self


class LocalDataset(torch_Dataset):
    """ Wrapper around torch.utils.data.Dataset to read Cache files.

        This particular object handles LocalCache objects, i.e.,
        works for non-distributed environments.

        `LocalDataset` can be thought as a DataFrame with some added
        functionality to keep track of column types. This is needed
        by `sigmoid` in the essenziehen step. The most important
        functionality is that it "knows" which columns correspond
        to a given type. Supported types are

        - numerical
        - categorical (as one-hot encoded vectors)
        - binary (zero and ones)
        - datetime (encoded as trigonometric series)
    """
    def __init__(self, path: str) -> None:
        """ Initializer for `LocalDataset`.

            @param path: str
                system path to HDF5 file with data.
        """
        self.binary_ = []
        self.categorical_ = []
        self.numerical_ = []
        self.datetime_ = []

        # swmr = True allows multiple workers to read from the same file
        # concurrently. handy!
        self.h5file_ = h5py.File(path, 'r', swmr=True)
        # use meta data to tell which columns are for training and
        # which columns are for target
        self.x_meta_data_ = MetaData()
        self.y_meta_data_ = MetaData()

        self.x_meta_data_.from_hdf5_attributes(self.h5file_['x'].attrs)
        self.y_meta_data_.from_hdf5_attributes(self.h5file_['y'].attrs)

        # assemble column ranges for different input types
        offset = 0
        for col_name in self.x_meta_data_.get_columns():
            idx = self.x_meta_data_.get_column_index(col_name)
            size = self.x_meta_data_.get_transformed_column_length(col_name)
            col_type = self.x_meta_data_.get_column_type(col_name)
            col_range = numpy.arange(idx + offset,
                                     idx + offset + size,
                                     dtype=int)
            col_range = col_range.tolist()
            offset += size - 1
            # assign column types
            if col_type == 'binary':
                self.binary_.append(col_range)
            elif col_type == 'categorical':
                self.categorical_.append(col_range)
            elif col_type == 'datetime':
                self.datetime_.append(col_range)
            # the rest is numerical
            else:
                self.numerical_.append(col_range)

        self.x_data_ = numpy.asarray(self.h5file_['x'][:])
        self.y_data_ = numpy.asarray(self.h5file_['y'][:])

        self.n_rows_ = self.x_data_.shape[0]
        self.n_input_ = self.x_data_.shape[1]
        self.n_output_ = self.y_data_.shape[1]

    def get_input_dim(self) -> int:
        """ Returns number of features in X data.
        """
        return self.n_input_

    def get_output_dim(self) -> int:
        """ Returns number of features in Y data.
        """
        return self.n_output_

    def get_input_metadata(self) -> MetaData:
        """ Returns reference to internal metadata of X.
        """
        return self.x_meta_data_

    def get_output_metadata(self) -> MetaData:
        """ Returns reference to internal metadata of Y.
        """
        return self.y_meta_data_

    def get_input_categorical_columns(self) -> list:
        """ Returns list of column ranges with one-hot encoded data.
        """
        in_onehot = copy.copy(self.categorical_)

        return in_onehot

    def get_input_binary_columns(self) -> list:
        """ Returns list of column ranges with binary data.
        """
        return self.binary_

    def get_input_numerical_columns(self) -> list:
        """ Returns list of column ranges with numerical data.
        """
        return self.numerical_

    def get_input_datetime_columns(self) -> list:
        """ Returns list of column ranges with datetime data
        """
        return self.datetime_

    def get_input_class_weights(self) -> list:
        """ Returns list of numpy.ndarray with relative class weights.

            Weights are calculated as normalized relative frequencies
            of occurrences. It is assumed that the transformation that
            encoded categories has a 'class_weights' parameter.
'
            @raises: ValueError if transformation does not have a
            parameter called 'class_weights'.
        """
        weights = []
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                p = self.x_meta_data_.get_parameter_data(col_name)
                if 'class_weights' not in p:
                    raise ValueError(
                        "Transformation needs 'class_weights' parameter.")
                w = p['class_weights']
                w = numpy.asarray(w).ravel()
                w = w / numpy.sum(w)
                weights.append(w)

        return weights

    def get_output_class_weights(self) -> list:
        """ Returns list of numpy.ndarray with relative class weights
            for the target variable.

            Weights are calculated as normalized relative frequencies
            of occurrences. It is assumed that the transformation that
            encoded categories has a 'class_weights' parameter.
'
            @raises: ValueError if transformation does not have a
            parameter called 'class_weights'.
        """
        weights = []
        for col_name in self.y_meta_data_.get_columns():
            col_type = self.y_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                p = self.y_meta_data_.get_parameter_data(col_name)
                if 'class_weights' not in p:
                    raise ValueError(
                        "Transformation needs 'class_weights' parameter.")
                w = p['class_weights']
                w = numpy.asarray(w).ravel()
                w = w / numpy.sum(w)
                weights.append(w)

        return weights

    def get_train_split(self, split_id: int = 0) -> numpy.ndarray:
        """ Returns training split
        """
        split = self.h5file_[f'train_split_{split_id}'][:]

        return split

    def get_test_split(self, split_id: int = 0) -> numpy.ndarray:
        """ Returns training split
        """
        split = self.h5file_[f'test_split_{split_id}'][:]

        return split

    def __len__(self):
        """ Returns number of rows in dataset.
        """
        return self.n_rows_

    def __getitem__(self, index):
        """ Returns tensors X and Y.
        """
        x = numpy.atleast_1d(self.x_data_)[index]
        y = numpy.atleast_1d(self.y_data_)[index]

        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)

        return X, Y


class LocalLoader(torch_DataLoader):
    """ Wrapper around torch.utils.data.DataLoader with
        support for distributed data loading.
    """

    def __init__(self, dataset: LocalDataset) -> None:
        """ Initializer.

            @param dataset: LocalDataset
                An instance of LocalDataset.
        """
        super(LocalLoader, self).__init__(dataset)

        self.dataset_ = dataset
        self.train_set_ = torch_Dataset
        self.test_set_ = torch_Dataset
        self.train_loader_ = torch_DataLoader
        self.test_loader_ = torch_DataLoader

        self.test_fraction_ = 0

        self.train_batch_size_ = 0
        self.test_batch_size_ = 0
        self.num_workers_ = 0
        # use persistence workers by default
        self.persistent_workers_ = True

    def get_train_loader(self,
                         split_id: int,
                         batch_size: int, num_workers: int = 0):
        """ Returns DataLoader for training.

            @param split_id: integer
                id of split to load.
            @param batch_size: integer
                batch size, passed to `batch_size` when
                building the `DataLoader`
            @param num_workers: integer (defaults to 0)
                number of workers, passed as to `num_workers`
                when building the `DataLoader`
        """
        train_split = self.dataset_.get_train_split(split_id).tolist()
        train_dataset = Subset(self.dataset_, train_split)

        if self.num_workers_ > 0 and self.num_workers_ is not None:
            self.train_loader_ = torch_DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=True,
                drop_last=True)
        else:
            self.train_loader_ = torch_DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True)

        return self.train_loader_

    def get_test_loader(self,
                        split_id: int,
                        batch_size: int, num_workers: int = 0):
        """ Returns DataLoader for training.

            @param split_id: integer
                id of split to load.
            @param batch_size: integer
                batch size, passed to `batch_size` when building
                the `DataLoader`.
            @param num_workers: integer (defaults to 0)
                number of workers, passed as to `num_workers` when
                building the `DataLoader`.
        """
        test_split = self.dataset_.get_test_split(split_id).tolist()
        test_dataset = Subset(self.dataset_, test_split)

        if self.num_workers_ > 0 and self.num_workers_ is not None:
            self.test_loader_ = torch_DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=True,
                drop_last=True)
        else:
            self.test_loader_ = torch_DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True)

        return self.test_loader_
