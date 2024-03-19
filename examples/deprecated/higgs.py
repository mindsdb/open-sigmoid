""" Script to try SIGMOID on the higgs dataset.

    This script will use *every* GPU. Run with

    > PYTHONPATH=$(pwd) torchrun --nnodes=1 --nproc_per_node=<number of gpus> scripts/higgs.py
"""
import time
import os

import numpy
import pandas
import torch
import torch.distributed as dist
from torch.distributed import FileStore

from torchmetrics.classification import Accuracy
from torchmetrics.classification import Precision
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import Recall

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
from sigmoid.spezialist.distributed.pools import SwitchMOE
from sigmoid.nn.convolutional import EfficientBackbone1D


def read_data(path):
    """ Returns dataframe with training data.
    """
    dframe = pandas.read_csv(path, low_memory=False, nrows=10000000)
    dframe.reset_index(inplace=True, drop=True)

    return dframe


if __name__ == '__main__':

    tic = time.time()
    store = FileStore('./temp_store')
    dist.init_process_group('nccl')

    if dist.get_rank() == 0:
        # assume data comes from elsewhere as a dataframe
        data_frame = read_data('/srv/storage/ml/datasets/higgs/data.csv')
        print(data_frame.shape)
        # create local cache
        cache = LocalCache(data_frame, 'class_label', ignore=[])
        cache.analyze_types()
        cache.build_splits()
        cache.attach_type_transformation('categorical', CategoricalAsOneHot)
        cache.attach_type_transformation('binary', BinaryAsZeroOne)
        cache.attach_type_transformation('float', NumericalNormalize)
        cache.attach_type_transformation('integer', NumericalNormalize)
        cache.transform()
        cache.populate_metadata()
        print(cache.x_data_.head())
        cache.to_hdf5('./temp_higgs.h5')
        # estimate codec dimensions
        est = LocalPCAEstimator(cache)
        opt_codec_size = est.get_optimal_dimension(threshold=0.5)
        print("original dimensions:", cache.x_data_.shape[1])
        print("optimal codec dimension:", opt_codec_size)
        store.set('opt_codec_size', str(opt_codec_size))
    else:
        pass
    dist.barrier()
    opt_codec_size = int(store.get('opt_codec_size'))

    # create local dataset
    dataset = LocalDataset('./temp_higgs.h5')
    dataloader = LocalLoader(dataset)
    dataloader.set_train_batch_size(2048)
    dataloader.set_test_batch_size(2048)
    dataloader.set_num_workers(4)
    dataloader.load_split()
    train_dataloader = dataloader.get_train_loader()
    test_dataloader = dataloader.get_test_loader()
    # # fit auto-encoder for dimensionality reduction
    ae = MixedTypesAutoEncoder(dataset.get_input_dim(),
                               # master encoder
                               codec_dim=opt_codec_size,
                               n_hidden=0,
                               hidden_dim=128,
                               # encoder for binary data
                               bin_n_filters=128,
                               # encoder for categorical data
                               cat_n_filters=128,
                               # encoder for numerical data
                               num_n_filters=512)
    ae.set_binary_columns(dataset.get_input_binary_columns())
    ae.set_categorical_columns(dataset.get_input_categorical_columns(),
                               dataset.get_input_class_weights())
    ae.set_numerical_columns(dataset.get_input_numerical_columns())
    ae.set_batch_size(1024)
    ae.build()
    ae.fit(train_dataloader, test_dataloader, n_epochs=100)
    # encode training set
    encoded = ae.encode_training_data(train_dataloader, max_samples=20000)
    torch.cuda.empty_cache()
    if dist.get_rank() == 0:
        ekmn = OptimalClustering(encoded, 2, 20,
                                 device_id=(dist.get_rank() % torch.cuda.device_count()))

        # find optimal number of clusters
        labels = ekmn.get_optimal_clustering_labels()
        labels = numpy.expand_dims(labels, axis=1)
        n_clusters = len(numpy.unique(labels))

        col_names = []
        for i in range(opt_codec_size):
            col_names.append(f'codec_{i}')
        col_names.append('__cluster_id__')
        data_enc = numpy.concatenate([encoded.cpu().numpy(), labels], axis=1)
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
        router_cache.build_splits()
        router_cache.attach_column_transformation('__cluster_id__', CategoricalAsOneHot)
        router_cache.transform()
        router_cache.populate_metadata()
        router_cache.to_hdf5('./temp_higgs_router_cache.h5')
    else:
        cache = None
    dist.barrier()

    n_clusters = int(store.get('n_clusters'))
    cluster_weights = numpy.frombuffer(store.get('cluster_weights'))
    print(cluster_weights)

    router_dataset = LocalDataset('./temp_higgs_router_cache.h5')
    router_loader = LocalLoader(router_dataset)
    router_loader.set_train_batch_size(2048)
    router_loader.set_test_batch_size(8192)
    router_loader.set_num_workers(24)
    router_loader.load_split()
    router_train_loader = router_loader.get_train_loader()
    router_test_loader = router_loader.get_test_loader()

    router = EfficientBackbone1D(ae.get_codec_dim(), n_clusters,
                                 kernel_size=3,
                                 n_filters=64)
    router.build()

    skill = EfficientBackbone1D(dataset.get_input_dim(),
                                dataset.get_output_dim(),
                                kernel_size=3,
                                n_filters=64)
    skill.set_final_activation(torch.nn.Sigmoid())
    skill.build()

    specialist = SwitchMOE(ae, 'binary_classification')
    specialist.set_router(router)
    specialist.set_router_balancing(cluster_weights)
    specialist.attach_router_metric('accuracy', Accuracy, task='multiclass',
                                    num_classes=n_clusters)
    specialist.attach_router_metric('precision', Precision, task='multiclass',
                                    num_classes=n_clusters)
    specialist.attach_router_metric('recall', Recall, task='multiclass',
                                    num_classes=n_clusters)

    specialist.pretrain_router(router_train_loader, router_test_loader,
                               n_epochs=100)

    specialist.set_skills(skill, n_clusters, torch.nn.BCELoss())
    specialist.set_skill_metric(BinaryPrecision)
    specialist.fit(train_dataloader, test_dataloader, n_epochs=100)

    gt, pr, rt = specialist.evaluate(test_dataloader)

    if dist.get_rank() == 0:
        pred_df = pandas.DataFrame()
        pr[pr < 0.5] = 0
        pr[pr >= 0.5] = 1
        pred_df['ground_truth'] = gt.ravel()
        pred_df['prediction'] = pr.ravel()
        pred_df['routing'] = rt
        pred_df.to_csv('higgs_predictions.csv')

        toc = time.time()

        print("# WALL TIME IN SECONDS: ", toc - tic)
    else:
        pass
    dist.barrier()

    dist.destroy_process_group()