Type inference
--------------------
Sigmoid leverages `type_infer`, another Python package in the MindsDB ecosystem, to estimate what is the optimal data type to use for interpreting each feature in any given dataset.

TypeInfer supports several data types used in standard machine learning pipelines. Its ``dtype`` class is used to label columns of information as the right input format. The type inference procedure affects what feature engineering methodology is used on a labeled column.

For more information, please refer to the TypeInfer documentation `here <https://mindsdb.github.io/type_infer/>`_.
