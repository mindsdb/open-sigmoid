""" transformations.py

Implementation of various data transformation for different types of data.

The base class of this file, ColumnTransform, defines an abstraction to perform
column-wise transformation on pandas.Series objects. In a nutshell, a column
transformation takes an input array (as pandas.Series) and outputs a matrix
(represented as pandas.DataFrame).

Transformations can have arguments and parameters; arguments are quantities
required to perform the transformation, while parameters are derived from
the transformation itself. For example, a transformation that takes an array
of classes (as text) and outputs a one-hot encoded vector might require a
parameter called `smooth` to perform label smoothing. Similarly, such
transformation could output a parameter called `class_weights` holding the
relative frequency of every class.

Transformations are stored in MetaData objects (see metadata.py) and may be
loaded from every-day Python dictionaries.

New column transforms might be added by specifying three methods:
    - __init__ (required to set a name for the transform)
    - forward_transform (i.e. categories to one-hot)
    - backwards transform (i.e. one-hot to categories)
    - from_metadata (to load transformation from dictionaries)
"""
import copy

from typing import Any
from typing import Union
from collections import OrderedDict

import numpy
import pandas

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class ColumnTransform:
    """ Base class for column transformations.

        Transformations operate on a single column and may have a
        number of arguments in their "apply" function. When building
        new transformations, the developer must add those arguments
        to the params_ dictionary so that they can be tracked by
        the metadata object and added to the coordinate file.
    """

    __MAX_TRANSFORM_PARAMETERS__ = 50
    __TRANSFORM_IDS__ = {'DateAsTrigonometric': 10,
                         'DatetimeAsTrigonometric': 11,
                         'CategoricalAsOrdinal': 20,
                         'CategoricalAsOneHot': 21,
                         'NumericalNormalize': 31,
                         'BinaryAsZeroOne': 41}

    def __init__(self, **kwargs):
        """ Initialiazes column transform.

            :param kwargs (keyword arguments)
                keyword arguments passed to ColumnTransform instance.
        """
        self.length_ = -1
        self.args_ = OrderedDict()
        self.params_ = OrderedDict()
        self.direction_ = 'forward'

        for arg_name, arg_value in kwargs.items():
            self.set_argument(arg_name, arg_value)

    def toggle_direction(self):
        """ Switches direction of transform.
        """
        if self.direction_ == 'forward':
            self.direction_ = 'backwards'
        elif self.direction_ == 'backwards':
            self.direction_ = 'forward'

    def set_length(self, length: int):
        """ Sets length of transformed column data.

            :param length (int)
                New "length" of column (see notes)

            :note
                While every entry in a table contains a single element
                (scalar data), the transformed element might have more
                than one element. A common example corresponds to the
                case of using one-hot encoding to represent categorical
                variables: in that case, the "length" of the transformed
                column would be equal to the number of categories.
        """
        self.length_ = length

    def get_length(self):
        """ Returns length of transformed column data.

            :returns
                Number of elements of transformed column data
                in the forward direction.
        """
        return self.length_

    def set_parameter(self,
                      parameter_name: str,
                      parameter_value: Any):
        """ Adds or sets parameters to the transform.

            :param parameter_name (str)
                Name of parameter
            :param parameter_value
                Value of parameter. If scalar, it must have a `__repr__ `
                implementation. If vector, every element must be convertible
                to `str`.
        """
        self.params_[parameter_name] = parameter_value

    def get_parameters(self) -> dict:
        """ Returns transform parameters.
        """
        d = copy.deepcopy(self.params_)
        return d

    def get_parameter_names(self) -> list:
        """ Returns list of parameter names.
        """
        return list(self.params_.keys())

    def set_argument(self,
                     argument_name: str,
                     argument_value: Any):
        """ Adds or sets arguments of the transform.

            :param argument_name (str)
                Name of argument.
            :param argument_value (scalar)
                Value of argument. If scalar, it must have a `__repr__ `
                implementation. If vector, every element must be convertible
                to `str`.

            :note
                `argument_value` must have a `__repr__` implementation.
        """
        self.args_[argument_name] = argument_value

    def get_arguments(self) -> dict:
        """ Returns dictionary of transform arguments.
        """
        d = copy.deepcopy(self.args_)
        return d

    def get_argument_names(self):
        """ Returns list of argument names.
        """
        return list(self.args_.keys())

    def __call__(self, col_data: Union[pandas.Series, pandas.DataFrame]):
        """ Executes transform on provided data.

            :param col_data (pandas.Series or pandas.DataFrame)
                If the transform direction is `forward`, this routine
                expects `col_data` to be a pandas.Series. When the
                transform direction is `backwards`, the routine expects
                `col_data` to be a pandas.DataFrame.
        """
        # while this routine never returns `None`,`
        # I just really don't like unbounded variable declarations
        # floating around
        transformed = None
        if self.direction_ == 'forward':
            transformed = self.forward_transform(col_data, **self.args_)
            self.set_length(transformed.shape[1])
        elif self.direction_ == 'backwards':
            transformed = self.backwards_transform(col_data, **self.args_)
            self.set_length(1)
        else:
            raise RuntimeError(f"{self.direction_} is not a valid direction.")

        return transformed

    def forward_transform(self,
                          col_data: pandas.Series,
                          **kwargs) -> pandas.DataFrame:
        """ Applies formward transformation.

            :param col_data (pandas.Series)
                pandas.Series containing a single column.
            :param kwargs (keyword arguments)
                keyword arguments of column transform routine.
        """
        raise NotImplementedError("forward_transform not implemented.")

    def backwards_transform(self,
                            col_data: pandas.DataFrame,
                            **kwargs) -> pandas.Series:
        """ Applies backwards transformation.

            :param col_data (pandas.DataFrame)
                pandas.DataFrame containing transformed representation of
                a single column.
            :param kwargs (keyword arguments)
                keyword arguments of backwards column transform routine.
        """
        raise NotImplementedError("backwards_transform not implemented.")

    def from_metadata(self, arguments: dict, parameters: dict):
        """ Loads transformation arguments and parameters.

            :param arguments (dict)
                dictionary with column transform arguments.
            :param parameters (dict)
                dictionary with column transform parameters.
        """
        raise NotImplementedError("from_metadata not implemented.")


class NormalizeToOneZero(ColumnTransform):
    """ Normalizes a sequence of numbers to the range [0, 1].
    """
    def __init__(self):
        """ Initializes ColumnTransform with name (required).
        """
        super(NormalizeToOneZero, self).__init__(
            name='NormalizeToOneZero')
        # set length of transformed data to 1 (scalar output)

    def forward_transform(self,
                          col_data: pandas.Series,
                          **kwargs) -> pandas.DataFrame:
        """ Executes forward transform.

            :param col_data (pandas.Series)
                pandas.Series containing data to be transformed. Each entry
                of this sequence must be convertible to `float`.
            :param kwargs (keyword arguments)
                Not used, kept for compatibility.
        """
        # transformed column is of length 1
        self.set_length(1)
        scaler = MinMaxScaler()
        col_name = col_data.name
        # fit MinMax scaler
        x = col_data.to_numpy().reshape((-1, 1))
        y1 = scaler.fit_transform(x)
        # create DataFrame to store transformed data.
        transformed = pandas.DataFrame()
        transformed[f'{col_name}_transformed'] = y1.ravel()

        self.set_parameter('scale', scaler.scale_)
        self.set_parameter('min', scaler.min_)

        return transformed

    def backwards_transform(self,
                            col_data: pandas.DataFrame,
                            **kwargs) -> pandas.Series:
        """ Executes backwards transform.

            :param col_data (pandas.DataFrame)
                Transformed data.
            :kwargs (keyword arguments)
                Not used; kept for compatiblity.
        """
        if 'scale' not in self.params_:
            raise KeyError("parameter 'scale' is missing.")
        if 'min' not in self.params_:
            raise KeyError("parameter 'min' is missing")
        scaler = MinMaxScaler()
        scaler.scale_ = self.params_['scale']
        scaler.min_ = self.params_['min']

        orig_col_data = scaler.inverse_transform(
            col_data.values.reshape((-1, 1)))

        orig_col = pandas.Series(orig_col_data.ravel())

        return orig_col

    def from_metadata(self, arguments: dict, parameters: dict):
        """ Loads scaling information from arguments/parameters.

            @param arguments: dict
                Dictionary with at least a keyword named 'name'
            @param parameters: dict
                Dictionary with at least the keywords
                    - 'scale' (float)
                    - 'mean' (float)
                    - 'var' (float, positive number)
        """
        if 'name' not in arguments:
            raise KeyError("'name' argument is missing.")
        if 'scale' not in parameters:
            raise KeyError("'scale' parameter is missing.")
        if 'min' not in parameters:
            raise KeyError("'min' parameter is missing")

        name = arguments['name']
        scale = parameters['scale']
        dmin = parameters['min']

        self.set_argument('name', name)
        self.set_parameter('scale', scale)
        self.set_parameter('min', dmin)


class CategoricalAsOneHot(ColumnTransform):
    """ Transformation for categorical features.
    """
    def __init__(self):
        """ Initializes transform with name (required).
        """
        super(CategoricalAsOneHot, self).__init__(
            name='CategoricalAsOneHot')

    def forward_transform(self,
                          col_data: pandas.Series,
                          **kwargs) -> pandas.DataFrame:
        """ Executes forward transform.

            :param col_data (pandas.Series)
                pandas.Series containing data to be transformed.
            :param kwargs (keyword arguments)
                Not used, kept for compatibility.
        """
        col_name = col_data.name
        lbl_encoder = LabelEncoder()
        # transform categories into ordinal labels
        lbl_encoded = lbl_encoder.fit_transform(col_data)
        n_categories = len(lbl_encoder.classes_)
        # set length of data output to n_categories
        self.set_length(n_categories)
        # labels go from 0 to n_categories - 1
        labels = numpy.asarray(lbl_encoded, dtype=int)
        # one-hot encode "a la numpy"
        idx = numpy.arange(labels.size)
        ohmatrix = numpy.zeros((labels.size, n_categories), dtype=float)
        ohmatrix[idx, labels] = 1.0
        # convert to dataframe
        col_names = []
        new_col_name = "{}_oh_{:d}"
        for lbl in range(0, n_categories):
            col_names.append(new_col_name.format(col_name, lbl))
        transformed = pandas.DataFrame(data=ohmatrix, columns=col_names)
        # compute class weights
        weights = transformed.sum(axis=0)
        weights = weights / weights.sum()
        # store class weights
        self.set_parameter('class_weights', weights.values)
        # store original class names
        self.set_parameter('class_names', lbl_encoder.classes_)

        return transformed

    def backwards_transform(self,
                            col_data: pandas.DataFrame,
                            **kwargs) -> pandas.Series:
        """ Executes backwards transform.

            :param col_data (pandas.DataFrame)
                Transformed data.
            :kwargs (keyword arguments)
                Not used; kept for compatiblity.
        """
        if 'class_names' not in self.params_:
            raise KeyError("'class_names' parameter is missing.")
        # create new dataframe with column names = class names
        class_names = self.params_['class_names']
        as_df = pandas.DataFrame(
            index=col_data.index,
            columns=class_names,
            data=col_data.to_numpy()
        )
        # one-hot -> original classes
        as_labels = as_df.idxmax(axis=1)

        return as_labels

    def from_metadata(self, arguments: dict, parameters: dict):
        """ Loads scaling information from arguments/parameters.

            @param arguments: dict
                Dictionary with at least a keyword named 'name'
            @param parameters: dict
                Dictionary with at least the keywords
                    - 'class_names' (float)
        """
        if 'name' not in arguments:
            raise KeyError("'name' argument is missing.")
        if 'class_names' not in parameters:
            raise KeyError("'class_names' parameter is missing.")

        name = arguments['name']
        class_names = parameters['class_names']

        self.set_argument('name', name)
        self.set_parameter('class_names', class_names)

'''
class DateAsTrigonometric(ColumnTransform):
    """ Transformation for dates (year-month-day)

        Transforms dates to trigonometric data points using
        the cosine and sine of the day of the year, month and week.
    """
    def __init__(self, year_period: float = 1.0, year_start: int = 1970):
        """ Initializes encoder.
        """
        super(DateAsTrigonometric, self).__init__(
            name='DateAsTrigonometric',
            year_period=year_period,
            year_start=year_start)

    def _apply(self, col_data: pandas.Series):
        """ Applies trigonometric encoding of dates.
        """
        col_name = col_data.name
        transformed = pandas.DataFrame()
        as_datetime = pandas.to_datetime(col_data)

        twopi = 2 * numpy.pi
        # years encoded with speficied period (default to 1)
        # starting from 1970
        yp = self.args_['year_period']
        ys = self.args_['year_start']
        twopiyear = twopi * (as_datetime.dt.year - ys) / yp
        twopidoy = twopi * as_datetime.dt.day_of_year / 365.0
        # january = 1 december = 12
        twopimon = twopi * as_datetime.dt.month / 12.0
        # monday = 0 sunday = 6
        twopidow = twopi * as_datetime.dt.day_of_week / 6.0

        transformed[f'{col_name}_cos_year'] = numpy.cos(twopiyear)
        transformed[f'{col_name}_sin_year'] = numpy.sin(twopiyear)
        transformed[f'{col_name}_cos_doy'] = numpy.cos(twopidoy)
        transformed[f'{col_name}_sin_doy'] = numpy.sin(twopidoy)
        transformed[f'{col_name}_cos_month'] = numpy.cos(twopimon)
        transformed[f'{col_name}_sin_month'] = numpy.sin(twopimon)
        transformed[f'{col_name}_cos_dow'] = numpy.cos(twopidow)
        transformed[f'{col_name}_sin_dow'] = numpy.sin(twopidow)

        return transformed


class DatetimeAsTrigonometric(ColumnTransform):
    """ Transformation for datetimes (year-month-day hour minute sec)

        Tranforms date-time values to trigonometric ones by computing
        the cosine and sine of the day of the year, the month, day of
        the week and seconds since start of the day.
    """
    def __init__(self):
        """ Initializes transformation.
            @param date_columns: list of columns that are dates.
        """
        super(DatetimeAsTrigonometric, self).__init__(
            name='DatetimeAsTrigonometric')

    def _apply(self, col_data: pandas.Series):
        """ Applies transformation to specified columns.
        """
        col_name = col_data.name
        transformed = pandas.DataFrame()
        as_datetime = pandas.to_datetime(col_data)

        twopi = 2 * numpy.pi

        twopidoy = twopi * as_datetime.dt.day_of_year / 365.0
        twopimon = twopi * as_datetime.dt.month / 12.0
        twopidow = twopi * as_datetime.dt.day_of_week / 6.0
        twopihrs = twopi * as_datetime.dt.hour / 24.0
        twopimin = twopi * as_datetime.dt.minute / 60.0
        twopisec = twopi * as_datetime.dt.second / 60.0
        # number of seconds in a day
        twopiday = \
            3600 * (twopihrs * 24) \
            + 60 * (twopimin * 60.0) \
            + (twopisec * 60)
        twopiday = twopiday / 86400.0

        transformed[f'{col_name}_cos_doy'] = numpy.cos(twopidoy)
        transformed[f'{col_name}_sin_doy'] = numpy.sin(twopidoy)
        transformed[f'{col_name}_cos_month'] = numpy.cos(twopimon)
        transformed[f'{col_name}_sin_month'] = numpy.sin(twopimon)
        transformed[f'{col_name}_cos_dow'] = numpy.cos(twopidow)
        transformed[f'{col_name}_sin_dow'] = numpy.sin(twopidow)
        transformed[f'{col_name}_cos_day'] = numpy.cos(twopiday)
        transformed[f'{col_name}_sin_day'] = numpy.sin(twopiday)
        transformed[f'{col_name}_cos_hrs'] = numpy.cos(twopihrs)
        transformed[f'{col_name}_sin_hrs'] = numpy.sin(twopihrs)
        transformed[f'{col_name}_cos_min'] = numpy.cos(twopimin)
        transformed[f'{col_name}_sin_min'] = numpy.sin(twopimin)
        transformed[f'{col_name}_cos_sec'] = numpy.cos(twopisec)
        transformed[f'{col_name}_sin_sec'] = numpy.sin(twopisec)

        return transformed


class BinaryAsZeroOne(ColumnTransform):
    """ Transformation for binary features.
    """
    def __init__(self):
        """ Initializer for transformation.
        """
        super(BinaryAsZeroOne, self).__init__(
            name='BinaryAsZeroOne')

    def _apply(self, col_data: pandas.Series) -> pandas.DataFrame:
        """ Encodes categorical columns as one-hot vectors.
        """
        col_name = col_data.name
        lbl_encoder = LabelEncoder()
        # transform categories into integer labels with
        # for binary data is just 0 or 1
        lbl_encoded = lbl_encoder.fit_transform(col_data)

        # convert to dataframe
        new_col_name = "{}_bin_zo".format(col_name)
        transformed = pandas.DataFrame(data=lbl_encoded,
                                       columns=[new_col_name, ])

        # compute class weights
        weights = transformed.sum(axis=0)
        weights = weights / weights.sum()
        self.set_parameter('class_weights', weights.values)
        self.set_parameter('class_names', lbl_encoder.classes_)

        return transformed
'''