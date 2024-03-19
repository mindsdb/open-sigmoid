""" convolutional.py

Module containing implementations of convolutional neural network
models in PyTorch.
"""
import math

import torch
from torch import nn
from torch import Tensor

from sigmoid.nn.base import Base
from sigmoid.nn.utils import conv1d_output_size
from sigmoid.nn.utils import conv1d_transposed_output_size


def round_value(value, scale):
    """ Returns nearest scaled value.
    """
    return int(math.ceil(value * scale))


class ConvBnRelu(Base):
    """ Basic block for batch normalized 1D convolution
        with LeakyReLU activation.
    """
    def __init__(self,
                 in_channel: int, in_width: int,
                 out_channel: int,
                 kernel_size: int, stride: int = 1):
        """ Initializer.

            Creates a single convolutional layer with batch normalization
            and LeakyReLU activation. Expects the input to be 1D in format
            (batch_size, in_channels, in_width).

            @param in_channel: int
                number of input channels (filters)
            @param in_width: int
                number of features.
            @param out_channel: int
                number of output channels (filters)
            @kernel_size: int
                size of the kernel used in the convolutional layer.
            @param stride: int (defaults to 1)
                stride of kernel windowing.
        """
        self.pad_size_ = kernel_size // 2
        self.kernel_size_ = kernel_size
        self.stride_ = stride
        out_c, out_l = conv1d_output_size(
            (in_channel, in_width), out_channel,
            padding=self.pad_size_,
            kernel_size=self.kernel_size_,
            stride=self.stride_)

        super(ConvBnRelu, self).__init__(
            (in_channel, in_width), (out_channel, out_l),
            "conv_bn_relu")

        self.layerinfo_['conv'] = {
            'input_c': in_channel,
            'input_h': 1,
            'input_w': in_width,
            'output_c': out_c,
            'output_h': 1,
            'output_w': 1,
        }

    def build(self) -> None:
        """ Builds convolutonal layer.
        """
        conv_params = self.layerinfo_['conv']
        self.layers_['conv'] = nn.Conv1d(
            conv_params.get('input_c'),
            conv_params.get('output_c'),
            kernel_size=self.kernel_size_,
            padding=self.pad_size_,
            stride=self.stride_
        )
        self.layers_['batch_normalization'] = \
            nn.BatchNorm1d(conv_params.get('output_c'))
        self.layers_['activation'] = nn.LeakyReLU()

        self.model_ = nn.Sequential(self.layers_)

        n = 0
        for p in self.model_.parameters():
            if p.requires_grad:
                n_n = 1
                for s in list(p.size()):
                    n_n = n_n * s
                n += n_n
            else:
                continue

        self.n_params_ = n


class TransposedConvBnRelu(Base):
    """ Basic block for batch normalized 1D transposed
        convolution with ReLU activation.
    """
    def __init__(self,
                 in_channel: int, in_width: int,
                 out_channel: int,
                 kernel_size: int, dilation: int = 1):
        """ Initializer.

            Creates a single transposed convolutional layer with batch
            normalization and LeakyReLU activation. Expects the input to
            be 1D in format (batch_size, in_channels, in_width).

            @param in_channel: int
                number of input channels (filters)
            @param in_width: int
                number of features.
            @param out_channel: int
                number of output channels (filters)
            @kernel_size: int
                size of the kernel used in the convolutional layer.
            @param dilation: int (defaults to 1)
                dilation used in kernel windowing.
        """
        self.pad_size_ = kernel_size // 2
        self.kernel_size_ = kernel_size
        self.dilation_ = dilation

        out_c, out_l = conv1d_transposed_output_size(
            (in_channel, in_width), out_channel,
            padding=self.pad_size_,
            kernel_size=self.kernel_size_,
            stride=1,
            dilation=self.dilation_)
        super(TransposedConvBnRelu, self).__init__(
            (in_channel, in_width), (out_channel, out_l),
            "transposed_conv_bn_relu")

        self.layerinfo_['transposed_conv'] = {
            'input_c': in_channel,
            'input_h': 1,
            'input_w': in_width,
            'output_c': out_c,
            'output_h': 1,
            'output_w': out_l
        }

    def build(self) -> None:
        """ Builds convolutonal layer.
        """
        conv_params = self.layerinfo_['transposed_conv']
        self.layers_['transposed_conv'] = nn.ConvTranspose1d(
            conv_params.get('input_c'),
            conv_params.get('output_c'),
            kernel_size=self.kernel_size_,
            padding=self.pad_size_,
            stride=1,
            dilation=self.dilation_
        )
        self.layers_['batch_normalization'] = \
            nn.BatchNorm1d(conv_params.get('output_c'))
        self.layers_['activation'] = nn.LeakyReLU()

        self.model_ = nn.Sequential(self.layers_)

        n = 0
        for p in self.model_.parameters():
            if p.requires_grad:
                n_n = 1
                for s in list(p.size()):
                    n_n = n_n * s
                n += n_n
            else:
                continue

        self.n_params_ = n


class ResidualConvModule(Base):
    """ Efficient-Net block with residual connection.
    """
    def __init__(self,
                 in_channel: int, in_width: int,
                 out_channel: int,
                 kernel_size, depth_scale, width_scale,
                 initial=False):
        """ Initializer.

            Creates a single convolutional block with batch normalization
            and LeakyReLU activation. Expects the input to be 1D in format
            (batch_size, in_channels, in_width).

            If in_channel == out_channel then a residual connection is added.

            depth_scale and width_scale control the number of parameters of
            the block by increasing (decreasing) the number of convolutional
            layers used (see EfficientNet paper)

            @param in_channel: int
                number of input channels (filters)
            @param in_width: int
                number of features.
            @param out_channel: int
                number of output channels (filters)
            @kernel_size: int
                size of the kernel used in the convolutional layer.
            @param depth_scale: float
                controls the depth (in filter space) of the block
            @param width_scale: float
                control the depth (in data space) of the block.
        """
        super(ResidualConvModule, self).__init__(
            (in_channel, in_width), (out_channel, ),
            'efficient_net_block'
        )

        self.kernel_size_ = kernel_size
        self.depth_ = round_value(2, depth_scale)
        self.width_scale_ = width_scale
        self.width_ = round_value(out_channel, width_scale)
        self.initial_ = initial
        self.residual_ = in_channel == out_channel

    def build(self):
        """ Builds single EfficientNet block.
        """
        lyr = None
        info = []
        if self.initial_:
            lyr = ConvBnRelu(
                self.in_[0], self.in_[1],
                self.width_,
                self.kernel_size_,
                stride=2)
        else:
            lyr = ConvBnRelu(
                round_value(self.in_[0], self.width_scale_), self.in_[1],
                self.width_,
                self.kernel_size_,
                stride=2)
        lyr.build()
        self.n_params_ += lyr.get_nparams()
        inf = lyr.get_layer_info()
        info.append(inf[next(reversed(inf))])
        self.layerinfo_['conv_bn_relu_in'] = info[-1]
        self.layers_['in'] = lyr

        for b_id in range(self.depth_ - 1):
            linfo = info[-1]
            lyr = ConvBnRelu(linfo['output_c'], linfo['output_w'],
                             self.width_,
                             self.kernel_size_)
            lyr.build()
            self.n_params_ += lyr.get_nparams()
            inf = lyr.get_layer_info()
            info.append(inf[next(reversed(inf))])
            self.layerinfo_[f'conv_bn_relu_{b_id}'] = info[-1]
            self.layers_[f'block_{b_id}'] = lyr

        self.model_ = nn.Sequential(self.layers_)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """ Forward method.
        """
        res = x_in
        y_hat = self.model_(x_in)
        if self.residual_:
            y_hat += res

        return y_hat


class TransposedResidualConvModule(Base):
    """ Efficient-Net block with inverse residual connection
        using transposed convolutions.
    """
    def __init__(self,
                 in_channel: int, in_width: int,
                 out_channel: int,
                 kernel_size, depth_scale, width_scale,
                 initial=False):
        """ Initializer.

            Creates a single transposed convolutional block with batch normalization
            and LeakyReLU activation. Expects the input to be 1D in format
            (batch_size, in_channels, in_width).

            If in_channel == out_channel then a residual connection is added.

            depth_scale and width_scale control the number of parameters of
            the block by increasing (decreasing) the number of convolutional
            layers used (see EfficientNet paper)

            @param in_channel: int
                number of input channels (filters)
            @param in_width: int
                number of features.
            @param out_channel: int
                number of output channels (filters)
            @kernel_size: int
                size of the kernel used in the convolutional layer.
            @param depth_scale: float
                controls the depth (in filter space) of the block
            @param width_scale: float
                control the depth (in data space) of the block.
        """
        super(TransposedResidualConvModule, self).__init__(
            (in_channel, in_width), (out_channel, ),
            'efficient_net_block_transposed'
        )

        self.kernel_size_ = kernel_size
        self.depth_ = round_value(2, depth_scale)
        self.width_scale_ = width_scale
        self.width_ = round_value(out_channel, width_scale)
        self.initial_ = initial
        self.residual_ = in_channel == out_channel

    def build(self):
        """ Builds block.
        """
        self.n_params_ = 0
        lyr = None
        info = []
        if self.initial_:
            lyr = TransposedConvBnRelu(
                self.in_[0], self.in_[1],
                self.width_,
                kernel_size=self.kernel_size_,
                dilation=2)
        else:
            lyr = TransposedConvBnRelu(
                round_value(self.in_[0], self.width_scale_), self.in_[1],
                self.width_,
                self.kernel_size_,
                dilation=2)
        lyr.build()
        self.n_params_ += lyr.get_nparams()
        inf = lyr.get_layer_info()
        info.append(inf[next(reversed(inf))])
        self.layerinfo_['conv_bn_relu_in'] = info[-1]
        self.layers_['in'] = lyr

        for b_id in range(self.depth_ - 1):
            linfo = info[-1]
            lyr = TransposedConvBnRelu(
                    linfo['output_c'], linfo['output_w'],
                    self.width_,
                    self.kernel_size_)
            lyr.build()
            self.n_params_ += lyr.get_nparams()
            inf = lyr.get_layer_info()
            info.append(inf[next(reversed(inf))])
            self.layerinfo_[f'trans_conv_bn_relu_{b_id}'] = info[-1]
            self.layers_[f'block_{b_id}'] = lyr

        self.model_ = nn.Sequential(self.layers_)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """ Forward method.

            Applies residual skip connection if input shape equals
            output shape.
        """
        res = x_in
        y_hat = self.model_(x_in)
        if self.residual_:
            y_hat += res

        return y_hat


class EfficientBackbone1D(Base):
    """ Implementation of "a la efficient-net" convolutional
        backbone for feature extraction.
    """
    def __init__(self, input_features, output_features,
                 kernel_size: int = 3,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 phi: float = 1.0,
                 n_filters: int = 128):
        """ Initializer for EfficientNet-ish backbone.
        """
        super(EfficientBackbone1D, self).__init__(
            (1, input_features),
            (output_features,),
            'efficientnet_backbone_1d')
        # number of filters
        self.nf_ = n_filters
        # kernel size for convolution
        self.ks_ = kernel_size
        # depth scale
        self.ds_ = alpha ** phi
        # width scale
        self.ws_ = beta ** phi
        # number of neurons in last channel
        self.lc_ = round_value(self.nf_, self.ws_)
        # feature extractor
        self.fe_ = nn.Sequential
        self.fc_ = nn.Sequential
        self.pool_ = nn.Module

    def build(self):
        """ Overloads method to build model.
        """
        self.n_params_ = 0

        info = []
        # build initial layer of feature extractor
        f = 16
        lyr = ResidualConvModule(
            self.in_[0], self.in_[1],
            f,
            self.ks_, self.ds_, self.ws_,
            initial=True)
        lyr.build()
        self.n_params_ += lyr.get_nparams()
        lyrinfo = lyr.get_layer_info()
        infitem = lyrinfo[next(reversed(lyrinfo))]
        info.append(infitem)
        self.layerinfo_['res_conv_module'] = info[-1]
        self.layers_[f'block_{1}_{f}'] = lyr

        while f < self.nf_:
            infitem = info[-1]
            lyr = ResidualConvModule(
                infitem['output_c'], infitem['output_w'],
                infitem['output_c'] * 2,
                self.ks_, self.ds_, self.ws_,
                initial=False)
            lyr.build()
            self.n_params_ += lyr.get_nparams()
            lyrinfo = lyr.get_layer_info()
            infitem = lyrinfo[next(reversed(lyrinfo))]
            info.append(infitem)
            self.layerinfo_['res_conv_module'] = info[-1]
            self.layers_[f'block_{f}_{2 * f}'] = lyr
            f = f * 2

        # lazy average pooling
        self.layers_['pooling'] = nn.AdaptiveAvgPool1d(1)
        # flatten along features
        self.layers_['flatten'] = nn.Flatten(start_dim=1)
        # fully connected layer
        self.layers_['fc'] = nn.Linear(self.lc_, self.out_[0])
        for p in self.layers_['fc'].parameters():
            if p.requires_grad:
                self.n_params_ += 1
        self.layers_['final_activation'] = \
            self.layerinfo_.get('final_activation', torch.nn.Identity())

        self.model_ = nn.Sequential(self.layers_)

    def forward(self, x_in: Tensor) -> torch.Tensor:
        """ Method to run model on input tensor.
        """
        x = x_in[:, None, :]
        y_hat = self.model_(x)

        return y_hat


class EfficientFrontbone1D(Base):
    """ Implementation of "a la efficient-net" convolutional
        backbone for feature extraction.

        The name "front-bone" is a reference to the fact that this
        network operates in transposed mode.
    """
    def __init__(self,
                 input_channels: int,
                 output_features: int,
                 kernel_size: int = 3,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 phi: float = 1.0,
                 n_filters: int = 128):
        """ Initializer for EfficientNet-ish backbone.
        """
        super(EfficientFrontbone1D, self).__init__(
            (input_channels, 1),
            (output_features,),
            'efficientnet_frontbone_1d')
        # number of filters
        self.nf_ = n_filters
        # kernel size for convolution
        self.ks_ = kernel_size
        # depth scale
        self.ds_ = alpha ** phi
        # width scale
        self.ws_ = beta ** phi
        # number of neurons in last channel
        self.lc_ = output_features
        # feature extractor
        self.fe_ = nn.Sequential
        self.fc_ = nn.Sequential
        self.pool_ = nn.Module

    def build(self):
        """ Overloads method to build model.
        """
        c = self.nf_
        info = []
        # build initial layer of feature extractor
        lyr = TransposedResidualConvModule(
            self.in_[0], self.in_[1],
            self.nf_,
            self.ks_, self.ds_, self.ws_,
            initial=True)
        lyr.build()
        self.n_params_ = lyr.get_nparams()
        lyrinfo = lyr.get_layer_info()
        infitem = lyrinfo[next(reversed(lyrinfo))]
        newinfo = {
            'input_c': self.in_[0],
            'input_h': 1,
            'input_w': self.in_[1],
            'output_c': self.nf_,
            'output_h': 1,
            'output_w': infitem['output_w']
        }
        info.append(newinfo)
        self.layerinfo_['res_conv_module'] = info[-1]
        self.layers_[f'block_{1}_{c}'] = lyr
        while c >= 16:
            infitem = info[-1]
            lyr = TransposedResidualConvModule(
                c, infitem['output_w'],
                c // 2,
                self.ks_, self.ds_, self.ws_,
                initial=False)
            lyr.build()
            self.n_params_ = lyr.get_nparams()
            lyrinfo = lyr.get_layer_info()
            infitem = lyrinfo[next(reversed(lyrinfo))]
            newinfo = {
                'input_c': c,
                'input_h': 1,
                'input_w': infitem['input_w'],
                'output_c': c // 2,
                'output_h': 1,
                'output_w': infitem['output_w']
            }
            info.append(newinfo)
            self.layerinfo_['res_conv_module'] = info[-1]
            self.layers_[f'block_{c}_{c // 2}'] = lyr
            c = c // 2

        # average pooling on channels
        self.layers_['pooling'] = nn.Conv1d(c, 1, kernel_size=1)
        # flatten along features
        self.layers_['flatten'] = nn.Flatten(start_dim=1)
        # fully connected layer
        info = self.layerinfo_[next(reversed(self.layerinfo_))]
        self.layers_['fc'] = nn.Linear(info['output_w'], self.lc_)
        for p in self.layers_['fc'].parameters():
            if p.requires_grad:
                self.n_params_ += 1
        self.layers_['final_activation'] = \
            self.layerinfo_.get('final_activation', torch.nn.Identity())

        self.model_ = nn.Sequential(self.layers_)

    def forward(self, x_in: Tensor) -> torch.Tensor:
        """ Method to run model on input tensor.
        """
        x = x_in[:, :, None]
        y_hat = self.model_(x)

        return y_hat
