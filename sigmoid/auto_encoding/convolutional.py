import os
import time

import numpy

import torch
from torch.nn import MSELoss
from torch.nn import BCELoss
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from torcheval.metrics import MeanSquaredError

from sigmoid.nn.base import Base
from sigmoid.nn.convolutional import EfficientBackbone1D
from sigmoid.nn.convolutional import EfficientFrontbone1D
from sigmoid.local.preprocessing.coordinate_files import LocalLoader


class MixedTypesAutoEncoder(Base):
    """ Auto-encoder that trains itself on mixed types of data.
    """
    def __init__(self, input_dim: int, codec_dim: int):
        """ Initializes auto-encoder.
        """
        super(MixedTypesAutoEncoder, self).__init__(
            (input_dim, ), (input_dim, ),
            'mixed_types_autoencoder')

        self.codec_dim_ = codec_dim
        self.codec_nf_ = -1
        self.codec_ds_ = 0.0
        self.codec_ws_ = 0.0

        self.bin_nf_ = -1
        self.bin_ds_ = 0.0
        self.bin_ws_ = 0.0

        self.cat_nf_ = -1
        self.cat_ds_ = 0.0
        self.cat_ws_ = 0.0

        self.num_nf_ = -1
        self.num_ds_ = 0.0
        self.num_ws_ = 0.0

        self.metric_ = {}
        self.metric_['mse'] = []
        self.metricfn_ = MeanSquaredError()

        self.enc_ = Base
        self.dec_ = Base
        self.encodings_ = torch.Tensor

        self.bin_loss_ = BCELoss(reduction='sum')
        self.cat_loss_ = []
        self.num_loss_ = MSELoss(reduction='sum')

        self.bin_cols_ = []
        self.bin_mask_ = torch.zeros(input_dim, dtype=torch.bool)
        self.bin_enc_ = Base
        self.bin_dec_ = Base

        self.cat_cols_ = []
        self.cat_col_ranges_ = []
        self.cat_enc_ = Base
        self.cat_dec_ = Base

        self.num_cols_ = []
        self.num_mask_ = torch.zeros(input_dim, dtype=torch.bool)
        self.num_enc_ = Base
        self.num_dec_ = Base

        self.class_weights_ = []

        self.seed_ = 42
        self.train_size_ = -1
        self.val_size_ = -1
        self.train_loader_ = LocalLoader
        self.val_loader_ = LocalLoader

        self.device_id_ = torch.device

    def to_disk(self, path: str):
        """ Writes model to disk.
        """
        # save global encoder/decoder
        torch.save(self.enc_.cpu().state_dict(),
                   os.path.join(path, 'global_encoder.pkt'))
        torch.save(self.dec_.cpu().state_dict(),
                   os.path.join(path, 'global_decoder.pkt'))
        # save categorical encoder/decoder
        torch.save(self.cat_enc_.cpu().state_dict(),
                   os.path.join(path, 'cat_encoder.pkt'))
        torch.save(self.cat_dec_.cpu().state_dict(),
                   os.path.join(path, 'cat_decoder.pkt'))
        torch.save(self.num_enc_.cpu().state_dict(),
                   os.path.join(path, 'num_encoder.pkt'))
        torch.save(self.num_dec_.cpu().state_dict(),
                   os.path.join(path, 'num_decoder.pkt'))

    def from_disk(self, path: str):
        """ Loads model from disk.
        """
        # save global encoder/decoder
        self.enc_.load_state_dict(
            torch.load(
                os.path.join(path, 'global_encoder.pkt')
            )
        )
        self.dec_.load_state_dict(
            torch.load(
                os.path.join(path, 'global_decoder.pkt')
            )
        )
        # categorical encoder/decoder
        self.cat_enc_.load_state_dict(
            torch.load(
                os.path.join(path, 'cat_encoder.pkt')
            )
        )
        self.cat_dec_.load_state_dict(
            torch.load(
                os.path.join(path, 'cat_decoder.pkt')
            )
        )
        # numerical encoder/decoder
        self.num_enc_.load_state_dict(
            torch.load(
                os.path.join(path, 'num_encoder.pkt')
            )
        )
        self.num_dec_.load_state_dict(
            torch.load(
                os.path.join(path, 'num_decoder.pkt')
            )
        )

    def set_device(self, device: torch.device):
        """ Sets the device to run training/inference.

            @param device: torch.device
                instance of torch.device (ex: torch.device('cpu'))
        """
        self.device_id_ = device

    def set_binary_encoder_parameters(self, n_filters,
                                      depth_scale: float = 1.0,
                                      width_scale: float = 1.0):
        """ Sets EfficientNet parameters of binary encoder/decoder.

            @param n_filters: int
                number of output filters.
            @param depth_scale: float
                controls the depth (in filter space) of the network.
            @param width_scale: float
                controls the depth (in data space) of the network.
        """
        self.bin_nf_ = n_filters
        self.bin_ds_ = depth_scale
        self.bin_ws_ = width_scale

    def set_categorical_encoder_parameters(self, n_filters,
                                           depth_scale: float = 1.0,
                                           width_scale: float = 1.0):
        """ Sets EfficientNet parameters of categorical encoder/decoder.

            @param n_filters: int
                number of output filters.
            @param depth_scale: float
                controls the depth (in filter space) of the network.
            @param width_scale: float
                controls the depth (in data space) of the network.
        """
        self.cat_nf_ = n_filters
        self.cat_ds_ = depth_scale
        self.cat_ws_ = width_scale

    def set_numerical_encoder_parameters(self, n_filters,
                                         depth_scale: float = 1.0,
                                         width_scale: float = 1.0):
        """ Sets EfficientNet parameters of numerical encoder/decoder.

            @param n_filters: int
                number of output filters.
            @param depth_scale: float
                controls the depth (in filter space) of the network.
            @param width_scale: float
                controls the depth (in data space) of the network.
        """
        self.num_nf_ = n_filters
        self.num_ds_ = depth_scale
        self.num_ws_ = width_scale

    def set_global_encoder_parameters(self, n_filters,
                                      depth_scale: float = 1.0,
                                      width_scale: float = 1.0):
        """ Sets EfficientNet parameters of global encoder/decoder.

            @param n_filters: int
                number of output filters.
            @param depth_scale: float
                controls the depth (in filter space) of the network.
            @param width_scale: float
                controls the depth (in data space) of the network.
        """
        self.codec_nf_ = n_filters
        self.codec_ds_ = depth_scale
        self.codec_ws_ = width_scale

    def get_train_loader(self) -> LocalLoader:
        """ Returns data loader for training.
        """
        return self.train_loader_

    def get_validation_loader(self) -> LocalLoader:
        """ Returns data loader for validation.
        """
        return self.val_loader_

    def get_device(self) -> torch.device:
        """ Returns device used for inference and training.
        """
        return self.device_id_

    def get_codec_dim(self) -> int:
        """ Return size of master enconder.
        """
        return self.codec_dim_

    def get_encodings(self) -> torch.Tensor:
        """ Returns encodings of last mini-batch pass.
        """
        return self.encodings_

    def get_test_mse(self, epoch) -> float:
        """ Returns MSE of validation batch for specified epoch.
        """
        return self.metric_['mse'][epoch]

    def set_binary_columns(self, bin_col_ranges: list):
        """ Sets column ranges of binary data.
        """
        self.bin_cols_ = []
        for range_idx in bin_col_ranges:
            col_range = range(range_idx[0], range_idx[-1] + 1)
            self.bin_mask_[col_range.start:col_range.stop] = True
            self.bin_cols_ += range_idx

    def set_categorical_columns(self, cat_col_ranges: list,
                                class_weights: list):
        """ Set column ranges of categorical data.
        """
        self.cat_cols_ = []
        for i, col_range in enumerate(cat_col_ranges):
            self.cat_cols_ += col_range
            s = col_range[0]
            e = col_range[-1] + 1
            self.cat_col_ranges_.append(range(s, e))
            if class_weights is None:
                w = torch.tensor(numpy.ones(e - s))
                w = w.to(self.device_id_)
            else:
                w = torch.tensor(1. / class_weights[i])
                w = w.to(self.device_id_)
            self.class_weights_.append(w)

    def set_numerical_columns(self, num_col_ranges: list):
        """ Set column ranges of numerical data.
        """
        self.num_cols_ = []
        for range_idx in num_col_ranges:
            col_range = range(range_idx[0], range_idx[-1] + 1)
            self.num_mask_[col_range.start:col_range.stop] = True
            self.num_cols_ += range_idx

    def build(self):
        """ Build encoder and decoder models.
        """
        self.n_params_ = 0
        # encoders
        n_g = 0
        n_b = len(self.bin_cols_)
        if n_b > 0:
            n_g += n_b
            self.bin_enc_ = EfficientBackbone1D(n_b, n_b,
                                                n_filters=self.bin_nf_,
                                                alpha=self.bin_ds_,
                                                beta=self.bin_ws_)
            self.bin_enc_.build()
            self.n_params_ += self.bin_enc_.get_nparams()
            self.bin_enc_ = self.bin_enc_.to(self.device_id_)

        n_c = len(self.cat_cols_)
        if n_c > 0:
            n_g += n_c
            self.cat_enc_ = EfficientBackbone1D(n_c, n_c,
                                                n_filters=self.cat_nf_,
                                                alpha=self.cat_ds_,
                                                beta=self.cat_ws_)
            self.cat_enc_.build()
            self.n_params_ += self.cat_enc_.get_nparams()
            self.cat_enc_ = self.cat_enc_.to(self.device_id_)

        n_n = len(self.num_cols_)
        if n_n > 0:
            n_g += n_n
            self.num_enc_ = EfficientBackbone1D(n_n, n_n,
                                                n_filters=self.num_nf_,
                                                alpha=self.num_ds_,
                                                beta=self.num_ws_)
            self.num_enc_.build()
            self.n_params_ += self.num_enc_.get_nparams()
            self.num_enc_ = self.num_enc_.to(self.device_id_)

        self.enc_ = EfficientBackbone1D(n_g, self.codec_dim_,
                                        n_filters=self.codec_nf_,
                                        alpha=self.codec_ds_,
                                        beta=self.codec_ws_)
        self.enc_.build()
        self.n_params_ += self.enc_.get_nparams()
        self.enc_ = self.enc_.to(self.device_id_)

        # decoders
        if n_b > 0:
            self.bin_dec_ = EfficientFrontbone1D(n_g, n_b,
                                                 n_filters=self.bin_nf_,
                                                 alpha=self.bin_ds_,
                                                 beta=self.bin_ws_)
            self.bin_dec_.build()
            self.n_params_ += self.bin_dec_.get_nparams()
            self.bin_dec_ = self.bin_dec_.to(self.device_id_)

        n_c = len(self.cat_cols_)
        if n_c > 0:
            self.cat_dec_ = EfficientFrontbone1D(n_g, n_c,
                                                 n_filters=self.cat_nf_,
                                                 alpha=self.cat_ds_,
                                                 beta=self.cat_ws_)
            self.cat_dec_.build()
            self.n_params_ += self.cat_dec_.get_nparams()
            self.cat_dec_ = self.cat_dec_.to(self.device_id_)

        n_n = len(self.num_cols_)
        if n_n > 0:
            self.num_dec_ = EfficientFrontbone1D(n_g, n_n,
                                                 n_filters=self.num_nf_,
                                                 alpha=self.num_ds_,
                                                 beta=self.num_ws_)
            self.num_dec_.build()
            self.n_params_ += self.num_dec_.get_nparams()
            self.num_dec_ = self.num_dec_.to(self.device_id_)

        self.dec_ = EfficientFrontbone1D(self.codec_dim_, n_g,
                                         n_filters=self.codec_nf_,
                                         alpha=self.codec_ds_,
                                         beta=self.codec_ws_)
        self.dec_.build()
        self.n_params_ += self.dec_.get_nparams()
        self.dec_ = self.dec_.to(self.device_id_)

        # masks
        self.bin_mask_ = self.bin_mask_.to(self.device_id_)
        self.num_mask_ = self.num_mask_.to(self.device_id_)

        # losses
        self.bin_loss_.to(self.device_id_)
        self.num_loss_.to(self.device_id_)
        for cw in self.class_weights_:
            c_loss = CrossEntropyLoss(reduction='sum', weight=cw)
            c_loss.to(self.device_id_)
            self.cat_loss_.append(c_loss)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """ Overloads forward method.
        """
        pre_encodings = []

        x_hat = torch.zeros((self.batch_size_, *self.in_),
                            dtype=torch.float32,
                            device=self.device_id_)

        # encode categorical data
        if len(self.cat_col_ranges_) > 0:
            c_c = self.cat_enc_(x_in[:, self.cat_cols_])
            pre_encodings.append(c_c)

        # encode numerical data
        if len(self.num_cols_) > 0:
            n_c = self.num_enc_(x_in[:, self.num_mask_])
            pre_encodings.append(n_c)

        # encode binary data
        if len(self.bin_cols_) > 0:
            b_c = self.bin_enc_(x_in[:, self.bin_mask_])
            pre_encodings.append(b_c)

        pre_enc = torch.cat(pre_encodings, dim=1)
        self.encodings_ = self.enc_(pre_enc)
        enc = self.encodings_
        post_enc = self.dec_(enc)

        # decode binary data
        if len(self.bin_cols_) > 0:
            b_d = self.bin_dec_(post_enc)
            x_hat[:, self.bin_mask_] = \
                torch.nn.functional.sigmoid(b_d)

        # decode categorical data
        if len(self.cat_cols_) > 0:
            c_d = self.cat_dec_(post_enc)
            # apply per-range softmax activation
            offset = 0
            for col_range in self.cat_col_ranges_:
                delta = col_range.stop - col_range.start
                real_range = torch.arange(col_range.start, col_range.stop, 1)
                offset_range = torch.arange(offset, offset + delta, 1)
                x_hat[:, real_range] = \
                    torch.nn.functional.softmax(c_d[:, offset_range], dim=-1)
                offset += delta

        # encode numerical data
        if len(self.num_cols_) > 0:
            n_d = self.num_dec_(post_enc)
            x_hat[:, self.num_mask_] = n_d

        return x_hat

    def compute_loss(self, y_hat, y_true) -> torch.Tensor:
        """ Compute reconstruction loss (MSE).
        """
        # loss for binary features
        bin_loss = torch.tensor(0, device=self.device_id_, dtype=torch.float32)
        # loss for numerical features
        num_loss = torch.tensor(0, device=self.device_id_, dtype=torch.float32)
        # loss for categorical features
        cat_loss = torch.tensor(0, device=self.device_id_, dtype=torch.float32)

        if len(self.bin_cols_) > 0:
            bin_loss = self.bin_loss_(y_hat[:, self.bin_mask_],
                                      y_true[:, self.bin_mask_])
        if len(self.cat_col_ranges_) > 0:
            for i, col_range in enumerate(self.cat_col_ranges_):
                real_range = torch.arange(col_range.start, col_range.stop, 1)
                cat_loss += self.cat_loss_[i](y_hat[:, real_range],
                                              y_true[:, real_range])
        if len(self.num_cols_) > 0:
            num_loss = self.num_loss_(y_hat[:, self.num_mask_],
                                      y_true[:, self.num_mask_])

        return bin_loss + cat_loss + num_loss

    def train_single_epoch(self,
                           dataloader: DataLoader,
                           optimizer: torch.optim.Optimizer):
        """ Train auto-encoder for a single epoch.

            @param
        """
        self.train(True)
        epoch_loss = 0.0
        for x_data, _ in dataloader:
            x_data = x_data.to(self.device_id_)
            # every batch starts with zero gradients
            optimizer.zero_grad()
            # run autoencoder on batch
            x_hat = self(x_data)
            # compute mean squared loss
            loss = self.compute_loss(x_hat, x_data)
            loss.backward(retain_graph=True)
            # adjust learning weights
            optimizer.step()
            # update running (training) loss
            epoch_loss += loss.item()
        self.train(False)

        return epoch_loss

    def eval_single_epoch(self, dataloader: DataLoader):
        """ Runs model in validation data.
        """
        self.eval()
        testing_loss = 0.0
        self.metricfn_.reset()
        with torch.no_grad():
            for x_data, _ in dataloader:
                x_data = x_data.to(self.device_id_)
                # run model
                x_hat = self(x_data)
                # update loss
                loss = self.compute_loss(x_hat, x_data)
                testing_loss += loss.item()
                self.metricfn_.update(x_hat.cpu(), x_data.cpu())
            # print(x_hat[0, :])
            # print(x_data[0, :])
        self.metric_['mse'].append(self.metricfn_.compute())

        return testing_loss

    def fit(self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            n_epochs: int) -> dict:
        """ Fit auto-encoder to data coming from `train_loader`, evaluating
            using data from `test_loader`.
        """
        training_log = {
            'epoch_wall_time': [],
            'train_loss': [],
            'test_loss': [],
            'reconstruction_error': []}

        logline = "{epoch:+04d},"
        logline += "{time:+04.4f},"
        logline += "{loss:+04.4f},"
        logline += "{test_loss:+04.4f},"
        logline += "{test_mse:+04.4f}"

        optimizer = torch.optim.Adam(self.parameters(), lr=1E-4)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        for epoch in range(0, n_epochs):
            # train/evaluate
            tic = time.time()
            train_loss = self.train_single_epoch(train_loader, optimizer)
            test_loss = self.eval_single_epoch(test_loader)
            toc = time.time()
            # print logging metrics
            log = logline.format(epoch=epoch+1,
                                 time=(toc - tic),
                                 loss=train_loss,
                                 test_loss=test_loss,
                                 test_mse=self.get_test_mse(epoch))
            print(log)
            # update training log
            training_log['epoch_wall_time'].append(toc - tic)
            training_log['train_loss'].append(train_loss)
            training_log['test_loss'].append(test_loss)
            training_log['reconstruction_error'].append(self.get_test_mse(epoch))
            # update learning rate
            scheduler.step()

        return training_log

    def encode(self, data_loader: DataLoader,
               max_samples: int = -1) -> torch.Tensor:
        """ Encode data from DataLoader.
        """
        counter = 0
        encodings = []
        self.eval()
        with torch.no_grad():
            for x_data, _ in data_loader:
                x_data = x_data.to(self.device_id_)
                _ = self(x_data)
                batch_encoding = self.get_encodings()
                counter += batch_encoding.shape[0]
                if counter > max_samples and max_samples > 0:
                    break
                encodings.append(batch_encoding)
                counter += 1
            encodings = torch.cat(encodings)

        return encodings
