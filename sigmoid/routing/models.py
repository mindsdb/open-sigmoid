import numpy

import torch

from torcheval.metrics import MulticlassAccuracy

from sigmoid.nn.base import Base
from sigmoid.nn.convolutional import EfficientBackbone1D
from sigmoid.preprocessing.local.coordinate_files import LocalLoader


class Convolutional1DSwitch(Base):
    """ Implementation of a convolutional 1D switch.

        This switch uses an EfficientNet backbone to classify codecs
        into appropiate routes, after codecs have been associated
        to clusters.
    """

    def __init__(self, input_dim: int, n_routes: int):
        """ Initializer

            @param input_dim: int
                number of features in 1D input, usually the codec size.
            @param n_routes: int
                number of routes for the switch.
        """
        super(Convolutional1DSwitch, self).__init__(
            (input_dim, ), (n_routes, ),
            'convolutional_1d_switch'
        )

        self.device_id_ = 0
        self.cluster_weights_ = numpy.ndarray
        self.loss_fn_ = torch.nn.Module

    def set_device(self, device_id: torch.device):
        """ Set device for computation.

            @param device_id: torch.device
                instance of torch.device (ex: torch.device('cpu'))
        """
        self.device_id_ = device_id

    def set_weights(self, class_weights):
        """ Sets class weights for classification.
        """
        w = torch.tensor(class_weights,
                         dtype=torch.float32,
                         device=self.device_id_)
        w = w / w.sum()
        self.cluster_weights_ = w
        self.loss_fn_ = torch.nn.CrossEntropyLoss(weight=w)
        self.loss_fn_.to(self.device_id_)

    def build(self,
              n_filters: int = 128,
              kernel_size: int = 5,
              depth_scale: float = 1.0):
        """ Builds model

            @param n_filters: int
                number of filters in the last convolutional layer.
            @param kernel_size: int
                size of kernel used in convolutional layers.
            @param depth_scale: float > 0
                parameter to control the depth of the convolutional
                backbone.
        """
        self.model_ = EfficientBackbone1D(self.in_[0], self.out_[0],
                                          n_filters=n_filters,
                                          kernel_size=kernel_size,
                                          alpha=depth_scale)
        self.model_.build()
        self.model_.to(self.device_id_)

        self.layerinfo_['backbone'] = {
            'input_c': 1,
            'input_h': 1,
            'input_w': self.in_[0],
            'output_c': 1,
            'output_h': 1,
            'output_w': self.out_[0]
        }

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """ Overloads forward method.

            @param x: torch.Tensor
                input tensor to run model on.
        """
        logits = self.model_(x_in)
        return logits

    def fit(self,
            train_loader: LocalLoader,
            test_loader: LocalLoader,
            n_epochs: int):
        """ Trains switch.

            @param train_loader: LocalLoader
                data loader for training data
            @param test_loader: LocalLoader
                data loader for testing data
            @param n_epochs: int
                number of epochs to run training for.
        """
        metric = MulticlassAccuracy(num_classes=self.out_[0])
        metric.to(self.device_id_)
        optimizer = torch.optim.Adam(self.parameters(), lr=1E-4)
        for epoch in range(n_epochs):
            self.train(True)
            epoch_loss = 0.0
            for x, y in train_loader:
                x = x.to(self.device_id_)
                y = y.to(self.device_id_)
                optimizer.zero_grad()
                y_hat = torch.nn.functional.softmax(self.model_(x), dim=-1)
                loss = torch.nn.functional.cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            self.eval()
            test_loss = 0.0
            metric.reset()
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(self.device_id_)
                    y = y.to(self.device_id_)
                    y_hat = torch.nn.functional.softmax(self.model_(x), dim=-1)
                    loss = torch.nn.functional.cross_entropy(y_hat, y)
                    test_loss += loss.item()
                    pred = torch.argmax(y_hat, dim=1)
                    target = torch.argmax(y, dim=1)
                    metric.update(pred, target)

            acc = metric.compute()
            println = "{:+04d} {:4.4f} {:4.4f} {:4.4f}"
            print(println.format(epoch + 1, epoch_loss, test_loss, acc))
