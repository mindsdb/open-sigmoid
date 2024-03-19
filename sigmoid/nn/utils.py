# https://github.com/tuttelikz/conv_output_size
import unittest

import numpy as np

import torch
import torch.nn as nn


def conv1d_output_size(input_size, out_channels, padding, kernel_size, stride, dilation=None):
    """According to https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    """
    if dilation is None:
        dilation = 1

    output_size = (
        out_channels,
        np.floor((input_size[1] + 2 * padding - dilation *
                  (kernel_size - 1) - 1) / stride + 1).astype(int)
    )
    return output_size


def conv1d_transposed_output_size(input_size, out_channels, padding, kernel_size, stride, dilation = 1, output_padding = 0):
    """ According to https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
    """
    Lin = input_size[1]
    Lout = (Lin - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    return (int(out_channels), int(Lout))


def conv2d_output_size(input_size, out_channels, padding, kernel_size, stride, dilation=None):
    """According to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    if dilation is None:
        dilation = (1, ) * 2
    if isinstance(padding, int):
        padding = (padding, ) * 2
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 2
    if isinstance(stride, int):
        stride = (stride, ) * 2

    output_size = (
        out_channels,
        np.floor((input_size[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) /
                 stride[0] + 1).astype(int),
        np.floor((input_size[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) /
                 stride[1] + 1).astype(int)
    )
    return output_size


def conv3d_output_size(input_size, out_channels, padding, kernel_size, stride, dilation=None):
    """According to https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
    """
    if dilation is None:
        dilation = (1, ) * 3
    if isinstance(padding, int):
        padding = (padding, ) * 3
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 3
    if isinstance(stride, int):
        stride = (stride, ) * 3

    output_size = (
        out_channels,
        np.floor((input_size[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) /
                 stride[0] + 1).astype(int),
        np.floor((input_size[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) /
                 stride[1] + 1).astype(int),
        np.floor((input_size[3] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) /
                 stride[2] + 1).astype(int)
    )
    return output_size


class ConvOutputSizeTest(unittest.TestCase):

    def test_conv1d_output_size(self):
        c_i, c_o = 3, 16
        k, s, p = 4, 1, 0

        sample_tensor = torch.ones((c_i, 128))
        c1d = nn.Conv1d(in_channels=c_i, out_channels=c_o, kernel_size=k,
                        stride=s, padding=p)

        output_size = conv1d_output_size(
            sample_tensor.shape, out_channels=c_o, kernel_size=k, stride=s, padding=p)

        self.assertEqual(
            c1d(sample_tensor).detach().numpy().shape, output_size)

    def test_conv2d_output_size(self):
        c_i, c_o = 3, 16
        k, s, p = 4, 1, 0

        sample_tensor = torch.ones((c_i, 128, 128))
        c2d = nn.Conv2d(in_channels=c_i, out_channels=c_o, kernel_size=k,
                        stride=s, padding=p)

        output_size = conv2d_output_size(
            sample_tensor.shape, out_channels=c_o, kernel_size=k, stride=s, padding=p)

        self.assertEqual(
            c2d(sample_tensor).detach().numpy().shape, output_size)

    def test_conv3d_output_size(self):
        c_i, c_o = 3, 16
        k, s, p = 4, 1, 0

        sample_tensor = torch.ones((c_i, 64, 128, 128))
        c3d = nn.Conv3d(in_channels=c_i, out_channels=c_o, kernel_size=k,
                        stride=s, padding=p)

        output_size = conv3d_output_size(
            sample_tensor.shape, out_channels=c_o, kernel_size=k, stride=s, padding=p)

        self.assertEqual(
            c3d(sample_tensor).detach().numpy().shape, output_size)