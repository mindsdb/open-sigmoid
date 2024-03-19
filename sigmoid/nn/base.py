"""
Base class for sigmoid deep learning models.
"""
from typing import Dict
from typing import Tuple
from collections import OrderedDict

import torch
from torch.nn import Module


class Base(Module):
    """ Base class for all SIGMOID models.
    """
    def __init__(self,
                 inputDims: Tuple,
                 outputDims: Tuple,
                 modelType: str):
        """ Base initializer.

            @param inputDims
                tuple of integers with input dimensions of model (not
                including mini-batch size)
            @param outputDims
                tuple of integers with output dimensions of model.
            @param modelType
                string with model type identifier, like "fc", "conv1d",
                "rnn" and so on.
        """
        super(Base, self).__init__()

        self.in_ = inputDims
        self.out_ = outputDims
        self.type_ = modelType
        self.layerinfo_ = OrderedDict()
        self.layers_ = OrderedDict()
        self.batch_size_ = -1
        self.n_params_ = 0
        self.model_ = Module

    def get_nparams(self):
        """ Returns number of *trainable* parameters in model.
        """
        return self.n_params_

    def get_input_shape(self) -> int:
        """ Returns number of dimensions of input.
        """
        return len(self.in_)

    def get_output_shape(self) -> int:
        """ Returns number of dimensions of input.
        """
        return len(self.out_)

    def get_input_dims(self) -> Tuple:
        """ Returns input dimensions.
        """
        return self.in_

    def get_output_dims(self) -> Tuple:
        """ Returns output dimension.
        """
        return self.out_

    def get_layer_info(self) -> Dict:
        """ Returns dictionary with layer information.

            Each entry in the returned list is a dictionary with keys

              'input_c': number of input channels
              'input_h': height of input
              'input_w': width of input
              'output_c': number of output channels
              'output_h': output height
              'output_w': output width
        """
        return self.layerinfo_

    def set_batch_size(self, batch_size: int):
        """ Sets batch size
        """
        self.batch_size_ = batch_size

    def get_batch_size(self) -> int:
        """ Returns batch size of model.
        """
        return self.batch_size_

    def load(self, path: str) -> None:
        """ Load model weights from checkpoint file.
        """
        self.model_.load_state_dict(torch.load(path))

    def save(self, path: str) -> None:
        """ Write model weights to checkpoint file.
        """
        torch.save(self.model_.cpu().state_dict(), path)

    def set_final_activation(self, act):
        """ Set final model activation.
        """
        self.layerinfo_['final_activation'] = act

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """ Performs single forward-pass of model (overloads __call__)
        """
        y_hat = self.model_(x_in)
        return y_hat

    def build(self) -> None:
        """ Creates sequential model from layer information.
        """
        raise NotImplementedError("Cannot call from base class.")
