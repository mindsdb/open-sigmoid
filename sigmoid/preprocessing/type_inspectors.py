""" type_inspectors.py

    Module that performs inference of column type for tabular data.

    Supported types are

    - categorical: categorical data
    - integer: natural numbers (0 and negative are included)
    - string: text-like data
    - float: real numbers
    - date-iso-8601: timestamps in ISO format (YYYY-M-dTH:m:S)
    - date-eu: timestamps in EU format (YYYY/MM/DD H:m:S)
    - date-non-std-subtype: incomplete timestamps (YYYY-MM-DD or H:m:S)
    - date-non-std: complete timestamps in non standard format
"""
import pandas

from ptype.PtypeCat import PtypeCat
from type_infer.infer import infer_types


class Inspector:
    """ Base class for type inspectors.
    """
    # valid column data types
    ___TYPES__ = (
        'categorical',
        'integer',
        'float',
        'string',
        'date-iso-8601',
        'date-eu',
        'date-non-std-subtype',
        'date-non-std'
    )

    def __init__(self, data_path: str, **kwargs):
        """ Initializer.

            :param data_path (str)
                system path to CSV file.
            :param kwargs (keyword arguments as expanded dict)
                keyword arguments passed to `pandas.read_csv`.

            :note
            The call to `pandas.read_csv()` is performed with
            `engine='python'`.`
        """
        # :note
        # local implementation is just the distributed one
        # using rank = 0 and world_size = 1
        self.rank_ = 0
        self.world_size_ = 1

        self.path_ = data_path
        # engine = 'python' is slower but more flexible for complex types
        self.raw_df_ = pandas.read_csv(self.path_, engine='python',
                                       **kwargs)

        self.type_inference_engine_ = None
        self.column_info_ = {}
        self.type_votes_ = {}

    def get_column_info(self):
        """ Returns column information.

            :returns `self.column_info_` (dict)
                A dictionary with keys equal to column names. Each entry
                is associated with another dictionary of keys `type` and
                `probability`; `type` corresponds to the inferred column
                type, while `probability` to the *relative* probability
                of that column corresponding to `type`.
        """
        r = {}
        r.update(self.column_info_)

        return r

    def infer_data_types(self, **kwargs):
        """ Performs type inference on loaded data.
        """
        raise NotImplementedError("Cannot call from base class.")


class PtypeInspector(Inspector):
    """ Performs probabilistic type inference using PType

        https://link.springer.com/article/10.1007/s10618-020-00680-1
    """
    def __init__(self, data_path: str, **kwargs):
        """ Initializer.

            :param data_path (str)
                system path to CSV file.
            :param kwargs (keyword arguments)
                keyword arguments passed to `pandas.read_csv()`.
        """
        super(PtypeInspector, self).__init__(data_path, **kwargs)

        self.type_inference_engine_ = PtypeCat()

    def infer_data_types(self, **kwargs):
        """ Performs type inference on loaded data.
        """
        schema = self.type_inference_engine_.schema_fit(self.raw_df_)
        # convert schema into common data structure
        for column in schema.cols:
            info = {}
            info['type'] = schema.cols[column].type
            info['probability'] = schema.cols[column].p_t.get(info['type'])
            self.column_info_[column] = info


class TypeInferInspector(Inspector):
    """ Performs type inference using MindsDB's type_infer package.
    """
    def __init__(self, data_path: str, **kwargs):
        """ Initializer.

            :param data_path (str)
                system path to CSV file.
            :param kwargs (keyword arguments)
                keyword arguments passed to `pandas.read_csv()`.
        """
        super(TypeInferInspector, self).__init__(data_path, **kwargs)

        self.type_inference_engine_ = infer_types

    def infer_data_types(self, **kwargs):
        """ Performs type inference on loaded data.
        """
        schema = self.type_inference_engine_(self.raw_df_, 0.0)
        # convert schema into common data structure
        for col_name, col_type in schema.dtypes.items():
            print(col_name, col_type)
            info = {}
            # type_infer -> sigmoid types
            if col_type in ['binary', ]:
                col_type = 'categorical'
            if col_type in ['short_text', 'rich_text', ]:
                col_type = 'string'
            if col_type in ['datetime']:
                col_type = 'date-non-std'
            if col_type in ['date']:
                col_type = 'date-non-std-subtype'
            info['type'] = col_type
            info['probability'] = 1.0
            self.column_info_[col_name] = info
