"""
fully_connected.py

Module containing basic implementation of Dense blocks in PyTorch.
"""
from torch.nn import Module
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Sequential

from sigmoid.nn.base import Base


class FullyConnectedBlock(Base):
    """ Basic FC block.
    """
    def __init__(self, n_input: int, n_output: int, n_layers: int, n_neurons: int, activation: Module = Identity()):
        """ Builds a dead-simple fully-connected NN.
        """
        super(FullyConnectedBlock, self).__init__((n_input, ), (n_output, ),
                                                  'fully-connected')
        self.act_ = activation
        if n_layers == 0:
            # input layer
            self.layerinfo_['single'] = {'input_c': 1,
                                         'input_h': 1,
                                         'input_w': n_input,
                                         'output_c': 1,
                                         'output_h': 1,
                                         'output_w': n_output}
        else:
            # input layer
            self.layerinfo_['input'] = {'input_c': 1,
                                        'input_h': 1,
                                        'input_w': n_input,
                                        'output_c': 1,
                                        'output_h': 1,
                                        'output_w': n_neurons}
            # hidden layers
            for ilyr in range(n_layers):
                self.layerinfo_[f"layer_{ilyr}"] = {'input_c': 1,
                                                    'input_h': 1,
                                                    'input_w': n_neurons,
                                                    'output_c': 1,
                                                    'output_h': 1,
                                                    'output_w': n_neurons}
            # output layer
            self.layerinfo_['output'] = {'input_c': 1,
                                         'input_h': 1,
                                         'input_w': n_neurons,
                                         'output_c': 1,
                                         'output_h': 1,
                                         'output_w': n_output}
        self.layerinfo_['final_activation'] = Identity()

    def build(self):

        _, final_act = self.layerinfo_.popitem()

        for lname, linfo in self.layerinfo_.items():
            self.layers_[lname] = Linear(linfo['input_w'], linfo['output_w'])
            if lname in ['output', 'single']:
                continue
            self.layers_[lname + "_act"] = self.act_
        self.layers_['final_activation'] = final_act
        self.model_ = Sequential(self.layers_)
