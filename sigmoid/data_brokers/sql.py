#!/usr/bin/env python
import os
import logging
from math import floor

import csv
import pandas

from sqlalchemy import Table
from sqlalchemy import MetaData
from sqlalchemy import select
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class SimpleSQLTableBroker:
    """ Implementation of simple SQL-to-pandas.DataFrame broker.
    """

    def __init__(self, uri: str, table_name: str, id: str):
        """ Initializes broker.

        @param uri: str
            URL of database. Must be compatible with SQLAlchemy.
        @param table_name: str
            Name of the table to connect.
        """

        # database parameters
        self.uri_ = uri
        self.table_name_ = table_name
        self.conn_ = None
        self.engine_ = None
        self.mdata_ = None
        self.session_ = None
        self.table_ = None
        self.dbrows_ = -1

        logger = logging.getLogger(__file__)
        self.logger_ = self._configure_logging(logger)

        self.id_ = id
        self.dataframe_ = pandas.DataFrame()
        self.rows_per_rank_ = 0
        self.num_workers_ = 1
        self.rank_ = 0
        self.offset_ = 0
        self.rs_ = 0
        self.re_ = 0

    def begin(self, schema: str = ''):
        """ Connect to database
        """
        self.logger_.info(f"Process {self.rank_}: starting connection to database.")

        # engine accepts commsize + 1 connections, with an overflow of 1
        self.engine_ = create_engine(self.uri_, echo=False, pool_size=self.num_workers_ + 1, max_overflow=1)
        # reflect table (magic)
        self.logger_.info(f'Process {self.rank_}: using schema named {schema}')
        self.mdata_ = MetaData(schema=schema)
        self.mdata_.reflect(bind=self.engine_, schema=schema)
        self.table_ = Table(self.table_name_, self.mdata_, autoload_with=self.engine_, schema=schema)
        self.session_ = sessionmaker(bind=self.engine_)()
        self.conn_ = self.engine_.connect()

        self.logger_.info(f"Process {self.rank_}: connection to database established.")

    def terminate(self):
        """ Gracefully closes database connection.
        """
        self.logger_.info(f"Process {self.rank_}: closing database session.")
        self.session_.close()

        self.logger_.info(f"Process {self.rank_}: disposing database engine.")
        self.engine_.dispose()

        self.logger_.info(f"Process {self.rank_}: terminated.")

    def configure(self):
        """ Configured broker based on the number of rows and workers.
        """
        # get number of rows in the database
        self.logger_.debug(f"Process {self.rank_}: fetching number of rows in the database")
        self.dbrows_ = self.session_.query(self.table_).count()
        self.logger_.debug(f"Process {self.rank_}: Database has {self.dbrows_} rows.")

        # calculate offsets to write data
        self.rows_per_rank_ = floor(self.dbrows_ / self.num_workers_)
        self.rs_ = self.rank_ * self.rows_per_rank_
        self.re_ = self.rs_ + self.rows_per_rank_

        return self.dbrows_

    def broke(self, path: str) -> str:
        """ Fetches data and writes it to CSV file.

            @param path: str
                base path to write the CSV file.

            @returns df_path: str
                system path to CSV file.
        """
        self.logger_.info(f"Process {self.rank_}: go go broker do your thing")

        # retrieve rows from database
        self.logger_.debug(f"[RANK {self.rank_}]: fetching rows from database")
        self._fetch(self.rs_, self.re_)
        df_path = self._dataframe_to_csv(path)

        return df_path

    def _configure_logging(self, logger: logging.Logger, level=logging.INFO):
        """ Provide automagic logger configuration.
        """
        logger.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # create console handler and set level to debug
        self.logStreamHandler_ = logging.StreamHandler()
        self.logStreamHandler_.setLevel(logging.DEBUG)
        self.logStreamHandler_.setFormatter(formatter)
        logger.addHandler(self.logStreamHandler_)

        return logger

    def _fetch(self, start: int, end: int):
        """ Fetches rows from database.
            Notes
            =====
            - datachunks contains an iterator that reads every 1024 columns.
        """
        self.logger_.info(f'Process {self.rank_}: fetching rows from {self.rs_} to {self.re_}.')
        try:
            batchsel = select(self.table_).slice(start, end)
            self.dataframe_ = pandas.read_sql(batchsel, self.conn_)
        except Exception as error:
            errmsg = "{:s}".format(Exception(error))
            self.logger_.critical(f"Process {self.rank_}: cannot fetch data from database: {errmsg}")
            self.terminate()
            raise RuntimeError(f"Process {self.rank_}: unable to fetch data.")

    def _dataframe_to_csv(self, path: str) -> str:
        """ Writes dataframe to CSV file.

            Output file has naming convention:
                id_{id}.start_{starting row}.end_{ending_row}.rank_{process rank}.csv

            @param path: str
                base path to store CSV file.

            @returns df_path: str
                system path to CSV file.
        """
        df_name = f'id_{self.id_}.start_{self.rs_}.end_{self.re_}.rank_{self.rank_}.csv'
        df_path = os.path.join(path, df_name)
        self.dataframe_.to_csv(df_path, quoting=csv.QUOTE_ALL)

        return df_path
