""" Script to try SIGMOID on the dielectron dataset.

    This script will use *a single* GPU. Run with

    > PYTHONPATH=$(pwd) python scripts/local/dielectron.py
"""
import time

import numpy
import pandas
import torch

from torchmetrics.regression import MeanSquaredError

# data files
from sigmoid.preprocessing.coordinate_files import LocalCache
from sigmoid.preprocessing.coordinate_files import LocalDataset
from sigmoid.preprocessing.coordinate_files import LocalLoader
# column transforms
from sigmoid.preprocessing.transformations import CategoricalAsOneHot
from sigmoid.preprocessing.transformations import NumericalNormalize
from sigmoid.preprocessing.transformations import BinaryAsZeroOne
# first-pass dimensionality reduction (for estimation purposes)
from sigmoid.analysis.dimensionality_reduction import LocalPCAEstimator
# soul capturing
from sigmoid.auto_encoding.autoencoders import MixedTypesAutoEncoder
# clustering to find optimal number of experts
from sigmoid.analysis.clustering import OptimalClustering
# switch-MoE
from sigmoid.switching.models import Convolutional1DSwitch
from sigmoid.model_scaling.pools import StochasticPool
from sigmoid.nn.convolutional import EfficientBackbone1D


def make_model():
    """ Returns instance of skill.
    """

    model = EfficientBackbone1D(16, 1,
                                n_filters=32, alpha=4.0)
    model.set_final_activation(torch.nn.Sigmoid())
    model.build()

    return model


def train_model():
    """ Trains simple model on sub-set of data.
    """
    dataframe = read_data(
        '/srv/storage/ml/datasets/dielectron/data.csv',
        10000)
    dcache = LocalCache(dataframe, 'M', ignore=['Run', 'Event'])
    dcache.analyze_types()
    dcache.attach_type_transformation('categorical', CategoricalAsOneHot)
    dcache.attach_type_transformation('binary', BinaryAsZeroOne)
    dcache.attach_type_transformation('float', NumericalNormalize)
    dcache.attach_type_transformation('integer', NumericalNormalize)
    dcache.transform()
    dcache.populate_metadata()
    dcache.build_splits()
    print(dcache.x_data_.head())
    dcache.to_hdf5('temp_dielectron_small.h5')

    # create local dataset and data loaders
    ddataset = LocalDataset('./temp_dielectron_small.h5')
    ddataloader = LocalLoader(ddataset)
    ddataloader.set_train_batch_size(128)
    ddataloader.set_test_batch_size(128)
    ddataloader.set_num_workers(8)
    ddataloader.load_split()
    dtrain_dataloader = ddataloader.get_train_loader()
    dtest_dataloader = ddataloader.get_test_loader()

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in list(range(100)):
        model.train(True)
        epoch_loss = 0.0
        for x_data, y_data in dtrain_dataloader:
            # every batch starts with zero gradients
            optimizer.zero_grad()
            # run autoencoder on batch
            y_hat = model(x_data)
            # compute mean squared loss
            loss = torch.nn.functional.l1_loss(y_hat, y_data)
            loss.backward()
            # adjust learning weights
            optimizer.step()
            # update running (training) loss
            epoch_loss += loss.item()

        eval_loss = 0.0
        with torch.no_grad():
            for x_data, y_data in dtest_dataloader:
                # run autoencoder on batch
                y_hat = model(x_data)
                # compute mean squared loss
                loss = torch.nn.functional.l1_loss(y_hat, y_data)
                # update running (training) loss
                eval_loss += loss.item()
        print(f"epoch = {epoch + 1} train_loss = {epoch_loss:4.4f} eval_loss = {eval_loss:4.4f}")
        model.train(False)

    return model


def read_data(path, nrows: int):
    """ Returns dataframe with training data.
    """
    dframe = pandas.read_csv(path, nrows=nrows,
                             low_memory=False)
    dframe.reset_index(inplace=True, drop=True)

    return dframe


if __name__ == '__main__':

    tic = time.time()

    # train skill on a subset of the dataset
    skill = train_model()
    torch.save(skill.cpu().state_dict(), 'skill.pkt')

    # read, transform and write raw data into cache
    data_frame = read_data(
        '/srv/storage/ml/datasets/dielectron/data.csv',
        50000)

    # create local cache
    cache = LocalCache(data_frame, 'M', ignore=['Run', 'Event'])
    cache.analyze_types()
    cache.attach_type_transformation('categorical', CategoricalAsOneHot)
    cache.attach_type_transformation('binary', BinaryAsZeroOne)
    cache.attach_type_transformation('float', NumericalNormalize)
    cache.attach_type_transformation('integer', NumericalNormalize)
    cache.transform()
    cache.populate_metadata()
    cache.build_splits()
    print(cache.x_data_.head())

    # estimate codec dimensions
    est = LocalPCAEstimator(cache)
    opt_codec_size = est.get_optimal_dimension(threshold=0.80)
    print("original dimensions:", cache.x_data_.shape[1])
    print("optimal codec dimension:", opt_codec_size)
    cache.to_hdf5('./temp_dielectron.h5')

    # create local dataset and data loaders
    dataset = LocalDataset('./temp_dielectron.h5')
    dataloader = LocalLoader(dataset)
    dataloader.set_train_batch_size(1024)
    dataloader.set_test_batch_size(1024)
    dataloader.set_num_workers(12)
    dataloader.load_split()
    train_dataloader = dataloader.get_train_loader()
    test_dataloader = dataloader.get_test_loader()

    # fit auto-encoder for dimensionality reduction
    ae = MixedTypesAutoEncoder(dataset.get_input_dim(), opt_codec_size)
    ae.set_global_encoder_parameters(128)
    ae.set_binary_encoder_parameters(128)
    ae.set_categorical_encoder_parameters(128)
    ae.set_numerical_encoder_parameters(512, depth_scale=2.0)
    ae.set_binary_columns(dataset.get_input_binary_columns())
    ae.set_categorical_columns(dataset.get_input_categorical_columns(),
                               dataset.get_input_class_weights())
    ae.set_numerical_columns(dataset.get_input_numerical_columns())
    ae.set_batch_size(512)
    ae.build()
    print("n parameters:", ae.get_nparams())
    ae.fit(train_dataloader, test_dataloader, n_epochs=50)

    # encode training set
    encoded = ae.encode_training_data(train_dataloader, max_samples=40000)
    encoded = encoded.cpu().numpy()
    # make some room for K-Means to run on GPU
    torch.cuda.empty_cache()

    # clustering analysis supports single GPU for now
    ekmn = OptimalClustering(encoded, 2, 20)

    # find optimal clustering
    clustered, labels = ekmn.get_optimal_clustering()
    labels = numpy.expand_dims(labels, axis=1)
    n_clusters = len(numpy.unique(labels))

    # create dummy data frame with synthetic labels
    col_names = []
    for i in range(opt_codec_size):
        col_names.append(f'codec_{i}')
    col_names.append('__cluster_id__')
    data_enc = numpy.concatenate([clustered, labels], axis=1)
    new_data_frame = pandas.DataFrame(
        data=data_enc,
        columns=col_names)
    # compute cluster weights
    switch_weights = new_data_frame['__cluster_id__'].value_counts(
        normalize=True, sort=False).values
    switch_weights = numpy.asarray(switch_weights)

    # setuo cache to train switch
    switch_cache = LocalCache(new_data_frame, '__cluster_id__', ignore=[])
    switch_cache.analyze_types()
    switch_cache.attach_column_transformation('__cluster_id__', CategoricalAsOneHot)
    switch_cache.transform()
    switch_cache.populate_metadata()
    switch_cache.build_splits()
    switch_cache.to_hdf5('./temp_dielectron_switch_cache.h5')

    # setup dataset and data loaders to train switch
    switch_dataset = LocalDataset('./temp_dielectron_switch_cache.h5')
    switch_loader = LocalLoader(switch_dataset)
    switch_loader.set_train_batch_size(512)
    switch_loader.set_test_batch_size(512)
    switch_loader.set_num_workers(8)
    switch_loader.load_split()
    switch_train_loader = switch_loader.get_train_loader()
    switch_test_loader = switch_loader.get_test_loader()

    # build and train switch
    switch = Convolutional1DSwitch(ae.get_codec_dim(), n_clusters)
    switch.build()
    switch.fit(switch_train_loader, switch_test_loader, 100)

    # build and load skill
    skill = make_model()
    skill.load_state_dict(torch.load('skill.pkt'))

    # build specialist and set switch
    specialist = StochasticPool(ae, 'regression')
    specialist.set_switch(switch)
    specialist.set_switch_balancing(switch_weights)
    # set specialist skill
    specialist.set_skills(skill, n_clusters, torch.nn.L1Loss())
    specialist.set_skill_metric(MeanSquaredError)
    # train specialist
    specialist.fit(train_dataloader, test_dataloader, n_epochs=100)

    # report output on test data
    gt, pr, rt = specialist.evaluate(test_dataloader)

    pred_df = pandas.DataFrame()
    pred_df['ground_truth'] = gt.ravel()
    pred_df['prediction'] = pr.ravel()
    pred_df['routing'] = rt

    pred_df.to_csv('dielectron_predictions.csv')

    toc = time.time()
    print("# WALL TIME IN SECONDS: ", toc - tic)