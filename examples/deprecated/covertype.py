""" Script to try SIGMOID on the covertype dataset.

    This script will use *every* GPU. Run with

    > PYTHONPATH=$(pwd) torchrun --nnodes=1 --nproc_per_node=<number of gpus> scripts/covertype.py
"""
import time
import os

import numpy
import pandas
import torch
import torch.distributed as dist
from torch.distributed import FileStore

from torchmetrics.classification import Accuracy

# data files
from sigmoid.vorbereiter.local.coordinate_files import LocalCache
from sigmoid.vorbereiter.local.coordinate_files import LocalDataset
from sigmoid.vorbereiter.local.coordinate_files import LocalLoader
# column transforms
from sigmoid.vorbereiter.local.transformations import CategoricalAsOneHot
from sigmoid.vorbereiter.local.transformations import NumericalNormalize
from sigmoid.vorbereiter.local.transformations import BinaryAsZeroOne
# first-pass dimensionality reduction (for estimation purposes)
from sigmoid.analyse.local.dimensionality_reduction import LocalPCAEstimator
# soul capturing
from sigmoid.essenzziehen.distributed.autoencoders import MixedTypesAutoEncoder
# clustering to find optimal number of experts
from sigmoid.fluglotse.local.clustering import OptimalClustering
# switch-MoE
from sigmoid.fluglotse.distributed.switches import Convolutional1DSwitch
from sigmoid.spezialist.distributed.pools import StocasticPool
from sigmoid.nn.convolutional import EfficientBackbone1D


def make_model():
    """ Returns instance of skill.
    """

    model = EfficientBackbone1D(53, 7,
                                n_filters=128, alpha=4.0)
    model.set_final_activation(torch.nn.Softmax(dim=-1))
    model.build()

    return model


def train_model():
    """ Trains simple model on sub-set of data.
    """
    dataframe = read_data(
        '/srv/storage/ml/datasets/covertype/data.csv',
        10000)
    dcache = LocalCache(dataframe, 'Cover_Type', ignore=[])
    dcache.analyze_types()
    dcache.attach_type_transformation('categorical', CategoricalAsOneHot)
    dcache.attach_type_transformation('binary', BinaryAsZeroOne)
    dcache.attach_type_transformation('float', NumericalNormalize)
    dcache.attach_type_transformation('integer', NumericalNormalize)
    dcache.transform()
    dcache.populate_metadata()
    dcache.build_splits()
    print(dcache.x_data_.head())
    dcache.to_hdf5('temp_covertype_small.h5')

    # create local dataset and data loaders
    ddataset = LocalDataset('./temp_covertype_small.h5')
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
            loss = torch.nn.functional.cross_entropy(y_hat, y_data)
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
                loss = torch.nn.functional.cross_entropy(y_hat, y_data)
                # update running (training) loss
                eval_loss += loss.item()
        print(f"epoch = {epoch + 1} train_loss = {epoch_loss:4.4f} eval_loss = {eval_loss:4.4f}")
        model.train(False)

    return model


def read_data(path, nrows: int):
    """ Returns dataframe with training data.
    """
    dframe = pandas.read_csv(path, nrows=nrows,
                             low_memory=False,
                             index_col=0)
    dframe.reset_index(inplace=True, drop=True)

    return dframe


if __name__ == '__main__':

    tic = time.time()
    store = FileStore('./temp_store')

    dist.init_process_group('gloo')
    if dist.get_rank() == 0:
        skill = train_model()
        torch.save(skill.cpu().state_dict(), 'skill.pkt')
    else:
        pass
    dist.barrier()

    # read, transform and write raw data into cache
    if dist.get_rank() == 0:
        data_frame = read_data(
            '/srv/storage/ml/datasets/covertype/data.csv',
            600000)

        # create local cache
        cache = LocalCache(data_frame, 'Cover_Type', ignore=[])
        cache.analyze_types()
        cache.attach_type_transformation('categorical', CategoricalAsOneHot)
        cache.attach_type_transformation('binary', BinaryAsZeroOne)
        cache.attach_type_transformation('float', NumericalNormalize)
        cache.attach_type_transformation('integer', NumericalNormalize)
        cache.transform()
        cache.populate_metadata()
        cache.build_splits()

        # estimate codec dimensions
        est = LocalPCAEstimator(cache)
        opt_codec_size = est.get_optimal_dimension(threshold=0.50)
        print("original dimensions:", cache.x_data_.shape[1])
        print("optimal codec dimension:", opt_codec_size)
        store.set('opt_codec_size', str(opt_codec_size))
        cache.to_hdf5('./temp_covertype.h5')
    else:
        pass
    dist.barrier()

    opt_codec_size = int(store.get('opt_codec_size'))

    # create local dataset and data loaders
    dataset = LocalDataset('./temp_covertype.h5')
    dataloader = LocalLoader(dataset)
    dataloader.set_train_batch_size(1024)
    dataloader.set_test_batch_size(1024)
    dataloader.set_num_workers(8)
    dataloader.load_split()
    train_dataloader = dataloader.get_train_loader()
    test_dataloader = dataloader.get_test_loader()

    # fit auto-encoder for dimensionality reduction
    ae = MixedTypesAutoEncoder(dataset.get_input_dim(), opt_codec_size)
    ae.set_global_encoder_parameters(128)
    ae.set_binary_encoder_parameters(256)
    ae.set_categorical_encoder_parameters(128)
    ae.set_numerical_encoder_parameters(256, depth_scale=2.0)
    ae.set_binary_columns(dataset.get_input_binary_columns())
    ae.set_categorical_columns(dataset.get_input_categorical_columns(),
                               dataset.get_input_class_weights())
    ae.set_numerical_columns(dataset.get_input_numerical_columns())
    ae.set_batch_size(512)
    ae.build()
    if dist.get_rank() == 0:
        print("n parameters:", ae.get_nparams())
    dist.barrier()
    ae.fit(train_dataloader, test_dataloader, n_epochs=100)

    # encode training set
    encoded = ae.encode_training_data(train_dataloader, max_samples=40000)
    encoded = encoded.cpu().numpy()
    # make some room for K-Means to run on GPU
    torch.cuda.empty_cache()

    # clustering analysis supports single GPU for now
    if dist.get_rank() == 0:
        ekmn = OptimalClustering(encoded, 2, 20,
                                 device_id=(dist.get_rank() % torch.cuda.device_count()))

        # find optimal number of clusters
        clustered, labels = ekmn.get_optimal_clustering()
        labels = numpy.expand_dims(labels, axis=1)
        n_clusters = len(numpy.unique(labels))

        col_names = []
        for i in range(opt_codec_size):
            col_names.append(f'codec_{i}')
        col_names.append('__cluster_id__')
        data_enc = numpy.concatenate([clustered, labels], axis=1)
        new_data_frame = pandas.DataFrame(
            data=data_enc,
            columns=col_names)
        # new_data_frame['__cluster_id__'] = -1
        # new_data_frame.loc[samp_idx, '__cluster_id__'] = labels
        cluster_weights = new_data_frame['__cluster_id__'].value_counts(
            normalize=True, sort=False).values
        cluster_weights = numpy.asarray(cluster_weights)
        print('cluster weights', cluster_weights)
        store.set('n_clusters', str(n_clusters))
        store.set('cluster_weights', cluster_weights.tobytes())

        router_cache = LocalCache(new_data_frame, '__cluster_id__', ignore=[])
        router_cache.analyze_types()
        router_cache.attach_column_transformation('__cluster_id__', CategoricalAsOneHot)
        router_cache.transform()
        router_cache.populate_metadata()
        router_cache.build_splits()
        router_cache.to_hdf5('./temp_covertype_router_cache.h5')
    else:
        pass
    dist.barrier()

    n_clusters = int(store.get('n_clusters'))
    cluster_weights = numpy.frombuffer(store.get('cluster_weights'))

    router_dataset = LocalDataset('./temp_covertype_router_cache.h5')
    router_loader = LocalLoader(router_dataset)
    router_loader.set_train_batch_size(512)
    router_loader.set_test_batch_size(512)
    router_loader.set_num_workers(8)
    router_loader.load_split()
    router_train_loader = router_loader.get_train_loader()
    router_test_loader = router_loader.get_test_loader()

    torch.cuda.empty_cache()
    switch = Convolutional1DSwitch(ae.get_codec_dim(), n_clusters)
    switch.build()
    switch.fit(router_train_loader, router_test_loader, 100)

    torch.cuda.empty_cache()
    skill = make_model()
    skill.load_state_dict(torch.load('skill.pkt'))

    specialist = StocasticPool(ae, 'classification')
    specialist.set_switch(switch)
    specialist.set_switch_balancing(cluster_weights)

    w = dataset.get_output_class_weights()[0]
    w = numpy.asarray(w)
    weights = torch.tensor(w, device=dist.get_rank() % torch.cuda.device_count())
    specialist.set_skills(skill, n_clusters,
                          torch.nn.CrossEntropyLoss(weight=weights))
    specialist.set_skill_metric(Accuracy, task='multiclass',
                                num_classes=dataset.get_output_dim())

    specialist.fit(train_dataloader, test_dataloader, n_epochs=100)

    gt, pr, rt = specialist.evaluate(test_dataloader)

    if dist.get_rank() == 0:
        pred_df = pandas.DataFrame()
        pred_df['ground_truth'] = numpy.argmax(gt, axis=1).ravel()
        pred_df['prediction'] = numpy.argmax(pr, axis=1).ravel()
        pred_df['routing'] = rt

        pred_df.to_csv('covertype_predictions.csv')

        toc = time.time()

        print("# WALL TIME IN SECONDS: ", toc - tic)
    else:
        pass
    dist.barrier()

    dist.destroy_process_group()
