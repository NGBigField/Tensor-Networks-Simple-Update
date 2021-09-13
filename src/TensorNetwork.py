import numpy as np
import pickle
from os.path import join


class TensorNetwork:
    """A Tensor-Network object. Used in the field of Quantum Information and Quantum Computation"""
    def __init__(self, structure_matrix: np.array = None, tensors: list = None, weights: list = None,
                 spin_dim: np.int = 2, virtual_dim: np.int = 3, dir_path='tmp/networks', load_network=False,
                 network_name=None):
        """
        :param structure_matrix: A 2D numpy array of integers > 0, corresponds to the interconnection between the tensor
         of the Tensor Network.
        :param tensors: A list of numpy arrays of dimension k + 1. The first k dimension (which potentially can be
        different for any tensor in the list) corresponds to the virtual dimension of the Tensor Network,  and the 1
         dimension correspond to the physical dimension of the Tensor Network (Spin dimension). Each array
        corresponds to a tensor in a the Tensor Network
        :param weights: A list of 1D numpy arrays corresponds to the simple update weights between the tensors of the
        Tensor Network.
        :param spin_dim: Relevant only in tensors==None. Then spin_dim is the size of the 0 dimension of all generated
        random tensors.
        :param virtual_dim: Relevant only if tensors==None and weights==None. Then, virtual_dim is the size of all
        the generated weights.
        :param dir_path: dictionary path for loading and saving networks
        :param load_network: if True, load a network from path/network_name
        :param network_name: name of network to load (only in load_network == True)
        """
        if load_network and network_name is not None:
            self.network_name = network_name
            self.dir_path = dir_path
            self.load_network()
        else:
            assert structure_matrix is not None, 'You should give a structure matrix as an argument input.'
            assert spin_dim > 0, f'Spin dimension should be an integer larger than 0. Instead got {spin_dim}.'

            # verify the structure matrix is legit
            assert len(structure_matrix.shape) == 2, f'structure_matrix have {len(structure_matrix.shape)} dimensions, ' \
                                                     'instead of 2.'
            n, m = structure_matrix.shape
            for i in range(n):
                row = structure_matrix[i, :]
                row = row[row > 0]
                assert len(set(row)) == len(row), f'There are two different weights ' \
                                                  f'connected to the same dimension in tensor [{i}].'
            for j in range(m):
                column = structure_matrix[:, j]
                assert np.sum(column > 0) == 2, f'Weight vector [{j}] is not connected to two tensors.'

            if tensors is not None:
                assert n == len(tensors), f'num of rows in structure_matrix is ' \
                                          f'{n}, while num of tensors is ' \
                                          f'{len(tensors)}. They should be equal.'
                # generate a weights list in case didn't get one
                if weights is None:
                    weights = [0] * m
                    for j in range(m):
                        i = 0
                        for i in range(n):
                            if structure_matrix[i, j] > 0:
                                break
                        k = tensors[i].shape[structure_matrix[i, j]]
                        weights[j] = np.ones(k) / k

            # generate a random (gaussian(0, 1)) tensors list in case didn't get one
            else:
                tensors = [0] * n
                for i in range(n):
                    tensor_shape = [spin_dim] + [0] * np.sum(structure_matrix[i, :] > 0)
                    for j in range(m):
                        if structure_matrix[i, j] > 0:
                            assert structure_matrix[i, j] <= len(tensor_shape) - 1, f'structure_matrix[{i}, {j}] = ' \
                                                                                    f'{structure_matrix[i, j]} while it ' \
                                                                                    f'should be <= {len(tensor_shape) - 1}.'
                            if weights is not None:
                                tensor_shape[structure_matrix[i, j]] = len(weights[j])
                            else:
                                tensor_shape[structure_matrix[i, j]] = virtual_dim
                    tensors[i] = np.random.normal(np.ones(tensor_shape))

                # generate a weights list in case didn't get one
                if weights is None:
                    weights = [0] * m
                    for j in range(m):
                        i = 0
                        for i in range(n):
                            if structure_matrix[i, j] > 0:
                                break
                        k = tensors[i].shape[structure_matrix[i, j]]
                        weights[j] = np.ones(k) / k

            assert m == len(weights), f'num of columns in structure_matrix is ' \
                                      f'{m}, while num of weights is ' \
                                      f'{len(weights)}. They should be equal.'

            # check connectivity
            for i in range(n):
                # all tensor virtual legs connected
                assert len(tensors[i].shape) - 1 == np.sum(structure_matrix[i, :] > 0), \
                    f'tensor [{i}] is connected to {len(tensors[i].shape) - 1}  ' \
                    f'weight vectors but have ' \
                    f'{np.sum(structure_matrix[i, :] > 0)} virtual dimensions.'

            # verify each neighboring tensors has identical interaction dimension to their shared weights
            for i in range(n):
                for j in range(m):
                    k = structure_matrix[i, j]
                    if k > 0:
                        assert tensors[i].shape[k] == len(weights[j]), f'Dimension {k} size of Tensor [{i}] is' \
                                                                       f' {tensors[i].shape[k]}, while size of weight ' \
                                                                       f'vector [{j}] is {len(weights[j])}. They should ' \
                                                                       f'be equal.'
            self.virtual_dim = virtual_dim
            self.spin_dim = spin_dim
            self.tensors = tensors
            self.weights = weights
            self.structure_matrix = structure_matrix
            self.dir_path = dir_path
            self.network_name = network_name
            self.su_logger = None
            self.state_dict = None

    def create_state_dict(self):
        self.state_dict = {
            'tensors': self.tensors,
            'weights': self.weights,
            'structure_matrix': self.structure_matrix,
            'path': self.dir_path,
            'spin_dim': self.spin_dim,
            'virtual_size': self.virtual_dim,
            'network_name': self.network_name,
            'su_logger': self.su_logger
        }

    def unpack_state_dict(self):
        self.tensors = self.state_dict['tensors']
        self.weights = self.state_dict['weights']
        self.structure_matrix = self.state_dict['structure_matrix']
        self.dir_path = self.state_dict['path']
        self.spin_dim = self.state_dict['spin_dim']
        self.virtual_dim = self.state_dict['virtual_size']
        self.network_name = self.state_dict['network_name']
        self.su_logger = self.state_dict['su_logger']

    def save_network(self, filename='tensor_network'):
        print('Saving network...')
        self.create_state_dict()
        if self.network_name is None:
            self.network_name = filename
        with open(join(self.dir_path, self.network_name + '.pkl'), 'wb') as outfile:
            pickle.dump(self.state_dict, outfile, pickle.DEFAULT_PROTOCOL)

    def load_network(self):
        print('loading network...')
        with open(join(self.dir_path, self.network_name + '.pkl'), 'rb') as infile:
            self.state_dict = pickle.load(infile)
            self.unpack_state_dict()
