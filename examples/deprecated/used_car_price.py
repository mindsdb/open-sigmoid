""" Script to try SIGMOID on the used car prices dataset.
"""
import time

import numpy
import pandas
import torch

from torchmetrics.classification import Accuracy
from torchmetrics.classification import Precision
from torchmetrics.classification import Recall

from torchmetrics.regression import MeanSquaredError

from torchsummary import summary

# data files
from sigmoid.preprocessing.local.coordinate_files import LocalCache
from sigmoid.preprocessing.local.coordinate_files import LocalDataset
# column transforms
from sigmoid.preprocessing.local.transformations import CategoricalAsOneHot
from sigmoid.preprocessing.local.transformations import NumericalNormalize
from sigmoid.preprocessing.local.transformations import BinaryAsZeroOne
# first-pass dimensionality reduction (for estimation purposes)
from sigmoid.analysis.local.dimensionality_reduction import LocalPCAEstimator
# soul capturing
from sigmoid.auto_encoding.local.autoencoders import MixedTypesAutoEncoder
# clustering to find optimal number of experts
from sigmoid.switching.local.clustering import OptimalClustering
# switch-MoE
from sigmoid.model_scalers.local.pools import StochasticPool
from sigmoid.nn.convolutional import EfficientBackbone1D


def read_data(path):
    """ Returns dataframe with training data.
    """
    dframe = pandas.read_csv(path, low_memory=False)
    dframe.reset_index(inplace=True, drop=True)

    return dframe


if __name__ == '__main__':

    tic = time.time()

    dev = torch.device('cuda')
    # for debugging
    torch.set_printoptions(precision=2)
    torch.set_printoptions(threshold=50)
    torch.set_printoptions(sci_mode=False)

    # assume data comes from elsewhere as a dataframe
    data_frame = read_data('/srv/storage/ml/datasets/used_car_price/data.csv')
    print(data_frame.shape)
    # create local cache
    cache = LocalCache(data_frame, 'price', ignore=[])
    cache.build_splits()
    cache.analyze_types()
    cache.attach_type_transformation('categorical', CategoricalAsOneHot)
    cache.attach_type_transformation('binary', BinaryAsZeroOne)
    cache.attach_type_transformation('float', NumericalNormalize)
    cache.attach_type_transformation('integer', NumericalNormalize)
    cache.transform()
    cache.populate_metadata()
    print(cache.x_data_.head())
    cache.to_hdf5('./temp_usp.h5')

    # estimate codec dimensions
    est = LocalPCAEstimator(cache)
    opt_codec_size = est.get_optimal_dimension(threshold=0.95)
    print("original dimensions:", cache.x_data_.shape[1])
    print("optimal codec dimension:", opt_codec_size)

    # create local dataset
    dataset = LocalDataset('./temp_usp.h5')
    print(dataset.get_output_dim())
    # # fit auto-encoder for dimensionality reduction
    ae = MixedTypesAutoEncoder(dataset.get_input_dim(),
                               # master encoder
                               codec_dim=opt_codec_size,
                               n_hidden=0,
                               hidden_dim=64,
                               # encoder for binary data
                               bin_n_filters=64,
                               # encoder for categorical data
                               cat_n_filters=64,
                               # encoder for numerical data
                               num_n_filters=64,
                               device=dev)
    ae.set_binary_columns(dataset.get_input_binary_columns())
    ae.set_categorical_columns(dataset.get_input_categorical_columns(),
                               dataset.get_input_class_weights())
    ae.set_numerical_columns(dataset.get_input_numerical_columns())
    ae.set_batch_size(512)
    ae.build()
    ae.fit(dataset,
           n_epochs=100, batch_size=ae.get_batch_size(),
           n_workers=24)

    # encode training set
    encoded = ae.encode_training_data()
    # sample training set (10% or 100000 rows)
    idx = numpy.arange(len(encoded))
    samp_idx = numpy.random.choice(idx,
                                   size=min(int(0.5 * len(encoded)), 100000),
                                   replace=False)
    # find optimal number of clusters
    ekmn = OptimalClustering(encoded[samp_idx], 3, 20)
    labels = ekmn.get_optimal_clustering_labels()
    labels = numpy.expand_dims(labels, axis=1)
    n_clusters = len(numpy.unique(labels))

    col_names = []
    for i in range(opt_codec_size):
        col_names.append(f'codec_{i}')
    col_names.append('__cluster_id__')
    print(encoded[samp_idx].shape)
    print(labels.shape)
    data_enc = numpy.concatenate([encoded[samp_idx], labels], axis=1)
    print(data_enc.shape)
    new_data_frame = pandas.DataFrame(data=data_enc,
                                      columns=col_names)
    #new_data_frame['__cluster_id__'] = -1
    #new_data_frame.loc[samp_idx, '__cluster_id__'] = labels
    cluster_weights = new_data_frame['__cluster_id__'].value_counts(
        normalize=True, sort=False).values
    print('cluster weights', cluster_weights)
    cache2 = LocalCache(new_data_frame, '__cluster_id__', ignore=[])
    cache2.analyze_types()
    cache2.attach_column_transformation('__cluster_id__', CategoricalAsOneHot)
    cache2.attach_type_transformation('categorical', CategoricalAsOneHot)
    cache2.attach_type_transformation('binary', BinaryAsZeroOne)
    cache2.attach_type_transformation('float', NumericalNormalize)
    cache2.attach_type_transformation('integer', NumericalNormalize)
    cache2.transform()
    cache2.populate_metadata()
    cache2.to_hdf5('./temp_usp_encoded.h5')

    router = EfficientBackbone1D(ae.get_codec_dim(), n_clusters,
                                 kernel_size=3,
                                 n_filters=32)
    router.build()

    skill = EfficientBackbone1D(dataset.get_input_dim(),
                                dataset.get_output_dim(),
                                kernel_size=3,
                                n_filters=32)
    skill.build()

    specialist = SwitchMOE(ae, ae.get_device(), 'regression')
    specialist.set_router(router)
    specialist.set_router_balancing(cluster_weights)
    specialist.attach_router_metric('accuracy', Accuracy, task='multiclass',
                                    num_classes=n_clusters)
    specialist.attach_router_metric('precision', Precision, task='multiclass',
                                    num_classes=n_clusters)
    specialist.attach_router_metric('recall', Recall, task='multiclass',
                                    num_classes=n_clusters)

    dataset2 = LocalDataset('./temp_usp_encoded.h5')
    specialist.pretrain_router(dataset2,
                               batch_size=256, n_workers=24,
                               n_epochs=100)

    specialist.set_skills(skill, n_clusters, torch.nn.L1Loss())
    specialist.set_skill_metric(MeanSquaredError)
    specialist.fit(ae.get_train_loader(),
                   ae.get_validation_loader(),
                   n_epochs=100)

    v_loader = ae.get_validation_loader()
    gt = []
    preds = []
    routes = []
    with torch.no_grad():
        for x, y in v_loader:
            x = x.to(dev)
            y = y.to(dev)
            y_hat = specialist(x)
            r = specialist.get_routes()
            preds.append(y_hat)
            gt.append(y)
            routes.append(r)

    predictions = torch.cat(preds).cpu().numpy()
    ground_truth = torch.cat(gt).cpu().numpy()
    skill_route = torch.cat(routes).cpu().numpy()

    pred_df = pandas.DataFrame()
    pred_df['ground_truth'] = ground_truth.ravel()
    pred_df['prediction'] = predictions.ravel()
    pred_df['routing'] = skill_route

    target_trafo = cache.get_column_transform('price')
    target_trafo.toggle_direction()
    pred_df['ground_truth'] = target_trafo(pred_df['ground_truth'])
    pred_df['prediction'] = target_trafo(pred_df['prediction'])

    pred_df.to_csv('used_car_price_predictions.csv')

    toc = time.time()

    print("# WALL TIME IN SECONDS: ", toc - tic)