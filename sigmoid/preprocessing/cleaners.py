""" cleaners.py

Implementation of cleansing routines for data columns.

Cleansing is the process of doing the best possible effort in
transforming an entry into its corresponding data type. For
example, an entry in the form

$456

is clearly numerical, but the '$' sign will prevent packages
like `pandas` or `numpy` to properly parse it to a numerical
representation.
"""
import pandas


class ColumnCleaner:
    """ Base class for type cleaner.
    """
    def __init__(self, name: str):
        """ Empty initializer.
        """
        self.name_ = name

    def clean(self, col_data: pandas.Series) -> pandas.Series:
        """ Base clean method.
        """
        raise NotImplementedError("Base class does not have clean().")

    def __call__(self, col_data: pandas.Series) -> pandas.Series:
        """ Cleaner calling function.
        """
        return self.clean(col_data)


class DummyCleaner(ColumnCleaner):
    """ Implementation of cleaner that does nothing.
    """
    def __init__(self):
        """ Initializer.
        """
        super(DummyCleaner, self).__init__('DummyCleaner')

    def clean(self, col_data: pandas.Series) -> pandas.Series:
        """ Returns data unchanged.
        """
        return col_data


class FloatingPointCleaner(ColumnCleaner):
    """ Cleaning routines for floating point data.
    """
    def __init__(self):
        """ Initializer
        """
        super(FloatingPointCleaner, self).__init__('FloatingPointCleaner')
        self.currency_tokens_ = ['$', ]

    def clean_trailing_spaces(self,
                              col_as_str: pandas.Series) -> pandas.Series:
        """ Removes trailing spaces from entries.

            Converts an entry in the form '  567 ' to '567'.
        """
        return col_as_str.apply(lambda x: x.strip(' '))

    def handle_commas(self, col_as_str: pandas.Series) -> pandas.Series:
        """ Remove or replace commas in numerical entries.

            This routine checks if '.' are present in the column to
            decide what to do. If '.' are found, the routine assumes
            that the data has numbers in the form

              '123,456.789'

            which shall be transformed to

              '123456.789'

            If '.' are not present, the routines assumes commas are
            being used as separators for decimal, and so an entry
            in the form

              '123,456'

            gets transformed to

              '123.456'
        """
        # check for commas, exit if not present
        if col_as_str.str.count(',').sum() == 0:
            return col_as_str
        # entries have commas, perform analysis
        comma_use = 'unknown'
        # comma used to separate thousands
        if col_as_str.str.count('.').sum() > 0:
            comma_use = 'thousands_separator'
        # comma used to separate decimals
        else:
            comma_use = 'decimal_separator'

        # remove commas as they confuse the parser
        if comma_use == 'thousands_separator':
            col_as_str = col_as_str.apply(lambda x: x.replace(',', ''))
        if comma_use == 'decimal_separator':
            col_as_str = col_as_str.apply(lambda x: x.replace(',', '.'))

        return col_as_str

    def handle_currency_tokens(self,
                               col_as_str: pandas.Series) -> pandas.Series:
        """ Removes currency tokens from entries.

            Entries that look

              '$ 123,123,123.456'

             are transformed to
               '123,123,123.456'.
        """
        for token in self.currency_tokens_:
            t = str(token)
            col_as_str = col_as_str.apply(lambda x: x.replace(t, ''))
        return col_as_str

    def clean(self, col_data: pandas.Series) -> pandas.Series:
        """ Returns clean version of data.
        """
        col_as_str = col_data.astype(str)
        col_as_str = self.handle_currency_tokens(col_as_str)
        col_as_str = self.clean_trailing_spaces(col_as_str)
        col_as_str = self.handle_commas(col_as_str)
        col_as_float = col_as_str.astype(float)

        return col_as_float
