""" home_rental_prices.py

    Runs sigmoid on the 'home_rental_prices' dataset using a pipeline-like
    execution flow.
"""
import sys
import glob
import json
import random
import logging

import csv
import yaml

import torch
import numpy
import pandas

# database broker
from sigmoid.local.data_brokers.sql import SimpleSQLTableBroker
# cache object
from sigmoid.local.preprocessing.coordinate_files import LocalCache
# dataset object
from sigmoid.local.preprocessing.coordinate_files import LocalDataset
# dataloader object
from sigmoid.local.preprocessing.coordinate_files import LocalLoader
# for type inference
from sigmoid.local.preprocessing.type_inspectors import PtypeInspector
# data transformation routines
from sigmoid.local.preprocessing.transformations import NormalizeToOneZero
from sigmoid.local.preprocessing.transformations import CategoricalAsOneHot
# data cleaning routines
from sigmoid.local.preprocessing.cleaners import FloatingPointCleaner
from sigmoid.local.preprocessing.cleaners import DummyCleaner
# dimensionality reduction
from sigmoid.local.analysis.dimensionality_reduction import LocalPCAEstimator
# auto-encoding step
from sigmoid.local.auto_encoding.convolutional import MixedTypesAutoEncoder


def configure_logger(level=logging.INFO) -> logging.Logger:
    """ Configured main logger to play nice with Prefect.

        :param level: logging level (defaults to logging.DEBUG)
            logging level. Allowed values are discribed in the `logging`
            module documentation.

        :returns main_logger: logging.Logger
            a logging.Logger instance that has been configured to
            redirect all mesages to `stdout`.
    """
    main_logger = logging.getLogger(__file__)
    main_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = \
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    main_logger.addHandler(handler)

    return main_logger


def read_config(path_to_config: str, logger: logging.Logger) -> dict:
    """ Reads YAML configuration file.

        :param path_to_config: str
            file system path to YAML file with configuration.

            The file must have at least the following entries

                uri: <URI of database>
                schema: <name of schema where table lives>
                table: <name of table to fetch>
        :param logger: logging.Logger
            Instance of `logging.Logger` to print messages. It is
            recommended that this instance comes from calling the
            `configure_logger` routine so that all logging messages
            are redirected to `stdout` (for Prefect to be able to
            fetch them and show them)

        :returns config: dict
            Python dictionary with at least 3 entries with keys
                - uri
                - schema
                - table
        :note
            For simplicity, the user and password are included in the URI.
            THIS IS A BAD SECURITY PRACTICE AND WILL BE CHANGED.
    """
    config = {}
    with open(path_to_config, "r", encoding='utf-8') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.critical("An error as occured.")
            raise exc

    return config


def run_database_broker(config: dict, logger: logging.Logger) -> str:
    """ Dumps database into CSV files.

        :param config: dict
            Python dictionary with at least 3 entries with keys
                - uri
                - schema
                - table
        :param logger: logging.Logger
            Instance of `logging.Logger` to print messages. It is
            recommended that this instance comes from calling the
            `configure_logger` routine so that all logging messages
            are redirected to `stdout` (for Prefect to be able to
            fetch them and show them)

        :returns `uid` (str)
            unique identifier of CSV files written to disk.
    """
    # get configuration parameters to connect to DB
    uri = config['uri']
    schema = config['schema']
    table_name = config['table']
    dbinfo = f"{schema}/{table_name}"
    logger.info(f"Reading database {dbinfo}")
    # generate unique identifier and distribute across workers
    r = random.randint(10000, 1000000)
    uid = str(r)
    logger.info(f"CSV uid = {uid}")

    broker = SimpleSQLTableBroker(uri, table_name, uid)
    broker.begin(schema=schema)
    broker.configure()

    df_path = broker.broke('./')
    logger.info(f"CSV file written to {df_path}")
    broker.terminate()

    return uid


def run_type_analysis(path: str):
    """ Runs type inference on provided CSV file.
    """
    inspector = PtypeInspector(path)
    inspector.infer_data_types()
    info = inspector.get_column_info()

    return info


def run_create_cache(csv_file: str):
    """ Creates Cache object from CSV file.
    """
    df = pandas.read_csv(csv_file, quoting=csv.QUOTE_ALL)
    cache = LocalCache(df, 'rental_price', ignore=['Unnamed: 0'])

    cache.load_column_types('column_info.json')

    cache.attach_type_transformation('float', NormalizeToOneZero, FloatingPointCleaner)
    cache.attach_type_transformation('integer', NormalizeToOneZero, FloatingPointCleaner)
    cache.attach_type_transformation('categorical', CategoricalAsOneHot, DummyCleaner)
    cache.transform()
    cache.populate_metadata()
    cache.build_splits()
    cache_path = csv_file.replace('.csv', '.h5')
    cache.to_hdf5(cache_path)

    return cache_path


def run_dim_analysis(cache_path: str):
    """ Calculates dimension for auto-encoder.
    """
    cache = LocalCache.from_hdf5(cache_path)
    dim_red = LocalPCAEstimator(cache)
    opt_dim = dim_red.get_optimal_dimension(threshold=0.75)

    return opt_dim


def run_auto_encoding(cache_path: str, codec_dim: int):
    """ Performs auto-encoding of dataset.
    """
    batch_size = 128
    dataset = LocalDataset(cache_path)
    dataloader = LocalLoader(dataset)
    train_dataloader = dataloader.get_train_loader(split_id=0,
                                                   batch_size=batch_size)
    test_dataloader = dataloader.get_test_loader(split_id=0,
                                                 batch_size=batch_size)

    ae = MixedTypesAutoEncoder(dataset.get_input_dim(), codec_dim)
    ae.set_device(torch.device('cpu'))
    ae.set_global_encoder_parameters(64)
    ae.set_binary_encoder_parameters(128)
    ae.set_categorical_encoder_parameters(128)
    ae.set_numerical_encoder_parameters(128)
    ae.set_binary_columns(dataset.get_input_binary_columns())
    ae.set_categorical_columns(dataset.get_input_categorical_columns(),
                               dataset.get_input_class_weights())
    ae.set_numerical_columns(dataset.get_input_numerical_columns())
    ae.set_batch_size(batch_size)
    ae.build()
    print("n parameters:", ae.get_nparams())
    train_log = ae.fit(train_dataloader, test_dataloader, n_epochs=100)

    ae.to_disk('./')

    return train_log


def run_dataset_encoding(cache_path: str, codec_dim: int):
    """ Performs encoding of input dataset.
    """
    batch_size = 128
    dataset = LocalDataset(cache_path)
    data_loader = LocalLoader(dataset)
    test_loader = data_loader.get_test_loader(split_id=0,
                                              batch_size=batch_size)
    ae = MixedTypesAutoEncoder(dataset.get_input_dim(), codec_dim)
    ae.set_device(torch.device('cpu'))
    ae.set_global_encoder_parameters(64)
    ae.set_binary_encoder_parameters(128)
    ae.set_categorical_encoder_parameters(128)
    ae.set_numerical_encoder_parameters(128)
    ae.set_binary_columns(dataset.get_input_binary_columns())
    ae.set_categorical_columns(dataset.get_input_categorical_columns(),
                               dataset.get_input_class_weights())
    ae.set_numerical_columns(dataset.get_input_numerical_columns())
    ae.set_batch_size(batch_size)
    ae.build()
    ae.from_disk('./')
    encoded = ae.encode(test_loader)
    encoded = encoded.cpu().numpy()
    print(encoded[0])

    numpy.save("dataset.npy", encoded)

    return True


def execute_pipeline():
    """ Runs simple pipeline.

        Currently, steps are
        - database broker
        - type inference

    """
    proc_logger = configure_logger()

    db_config = read_config(
        "./database_broker.yaml", proc_logger)

    # begin of step 1
    csv_id = run_database_broker(db_config, proc_logger)
    csv_files = glob.glob(f'./id_{csv_id}*.csv')
    column_info = run_type_analysis(csv_files[0])
    with open('./column_info.json', 'w', encoding='utf-8') as f:
        json.dump(column_info, f, indent=4)
    cache_path = run_create_cache(csv_files[0])
    codec_dim = run_dim_analysis(cache_path)
    # end of step 1

    # begin step 2
    run_auto_encoding(cache_path, codec_dim)
    run_dataset_encoding(cache_path, codec_dim)
    # end step 2

if __name__ == '__main__':

    execute_pipeline()

