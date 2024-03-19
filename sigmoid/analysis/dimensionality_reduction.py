""" dimensionality_reduction.py

This module contains tools to estimate the an optimal dimensionality
reduction scheme for data that has been already prepared for ML.

Available methods are:
- PCA

"""
import numpy
from sklearn.decomposition import PCA

from sigmoid.local.preprocessing.coordinate_files import LocalCache


class LocalPCAEstimator:
    """ Estimates optimal dimensionality reduction using Principal Component
        Analysis. This estimator works locally, that, in a non-distributed
        compute environment.
    """

    def __init__(self, cache: LocalCache):
        """ Initializer for PCAEstimator.

        @param cache: LocalCache object.
        """
        self.k_comp = -1
        self.data_ = cache.x_data_
        print(cache.x_data_.shape)
        self.estimator_ = PCA(n_components=2)
        self.var_ = {}

    def build_estimator(self):
        """ Initializes PCA estimator.
        """
        self.estimator_ = PCA(n_components=self.k_comp)

    def set_estimator_components(self, k: int):
        """ Sets the number of components for PCA.
        """
        self.k_comp = k

    def get_explained_variance(self):
        """ Runs PCA and returns the explained variance.
        """
        self.estimator_.fit(self.data_)
        exp_var = numpy.sum(
            self.estimator_.explained_variance_ratio_)

        return exp_var

    def get_optimal_dimension(self, threshold: float = 0.50):
        """ Returns number of components to recover specified variance.

            @param threshold (optional)
                float between 0 and 1 that specifies the minimum recovered
                variance of PCA. Defaults to 0.5 (50%).
        """
        ks = numpy.linspace(2, self.data_.shape[1], 5, endpoint=True)
        vs = []
        for k in ks:
            self.set_estimator_components(int(k))
            self.build_estimator()
            var = self.get_explained_variance()
            vs.append(var)
        exp_var = numpy.asarray(vs, dtype=float)

        all_ks = numpy.arange(2, self.data_.shape[1], 1)
        all_vars = numpy.interp(all_ks, ks, exp_var)
        # number of components that recovers
        if numpy.max(all_vars) < threshold:
            raise RuntimeError("PCA not able to recover specified variance.")
        k_opt = int(all_ks[all_vars > threshold][0])

        return k_opt
