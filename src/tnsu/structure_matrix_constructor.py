import numpy as np

"""
A File for Structure Matrices constructions. 
"""


def infinite_structure_matrix_dict(name: str):
    """
    A dictionary of iPEPS structure matrices as written in the paper "A universal tensor network algorithm for
    any infinite lattice".
    :param name: name of the infinite Tensor Network
    :return: a structure matrix
    """
    structure_matrix = {'chain': np.array([[1, 2],
                                           [1, 2]]),

                        'peps': np.array([[1, 2, 3, 4, 0, 0, 0, 0],
                                          [1, 2, 0, 0, 3, 4, 0, 0],
                                          [0, 0, 1, 2, 0, 0, 3, 4],
                                          [0, 0, 0, 0, 1, 2, 3, 4]]),

                        'star': np.array([[1, 2, 3, 0, 0, 0, 0, 0, 0],
                                          [1, 0, 0, 2, 3, 0, 0, 0, 0],
                                          [0, 1, 0, 2, 0, 3, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 1, 2, 3, 0],
                                          [0, 0, 0, 0, 1, 0, 2, 0, 3],
                                          [0, 0, 1, 0, 0, 0, 0, 2, 3]]),

                        'cube': np.array([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [1, 2, 0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 5, 6]]),

                        'triangle': np.array([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [1, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 0, 5, 6, 0, 0, 0],
                                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 5, 6, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 5, 0, 6],
                                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 4, 0, 5, 6]]),
                        }
    return structure_matrix[name]


def square_peps_pbc(side: np.int):
    """
    Creates a structure matrix of a square lattice Tensor network with periodic boundary conditions (pbc)
    of shape (side, side). The total number of tensors in the network would be side^2.
    :param side: side-length of tensor network
    :return: a structure matrix
    """
    n_tensors = np.int(np.square(side))
    structure_matrix = np.zeros((n_tensors, 2 * n_tensors), dtype=np.int)
    for i in range(n_tensors):
        structure_matrix[i, i] = 4
        structure_matrix[i, i + n_tensors] = 1
        if i % side == 0:
            structure_matrix[i, i + side - 1] = 2
        else:
            structure_matrix[i, i - 1] = 2
        if i >= side:
            structure_matrix[i, n_tensors - side + i] = 3
        else:
            structure_matrix[i, 2 * n_tensors - side + i] = 3
    return structure_matrix


def rectangular_peps_obc(height: np.int, width: np.int):
    """
    Creates a structure matrix of a rectangular lattice tensor network with open (non-periodic) boundary
    conditions (obc) of shape (height x width). The total number of tensors in the network would be height x width.
    :param height: The height of the tensor network.
    :param width: The width of the tensor network.
    :return: a structure matrix
    """
    # edge = (node_a i, node_a j, node_a l, node_b i, node_b j, node_b l)

    # create tuples of tensor indices
    edge_list = []
    for i in range(height):
        for j in range(width):
            if i < height - 1:
                edge_list.append((i, j, 4, i + 1, j, 2))
            if j < width - 1:
                edge_list.append((i, j, 3, i, j + 1, 1))
    structure_matrix = np.zeros(shape=[height * width, len(edge_list)], dtype=np.int)

    # fill in the structure matrix
    for edge_idx, edge in enumerate(edge_list):
        node_a_idx = np.ravel_multi_index([edge[0], edge[1]], (height, width))
        node_b_idx = np.ravel_multi_index([edge[3], edge[4]], (height, width))
        structure_matrix[node_a_idx, edge_idx] = edge[2]
        structure_matrix[node_b_idx, edge_idx] = edge[5]

    # reorder dimension according to a constant order
    for i in range(structure_matrix.shape[0]):
        row = structure_matrix[i, np.nonzero(structure_matrix[i, :])[0]]
        new_row = np.array(range(1, len(row) + 1))
        order = np.argsort(row)
        new_row = new_row[order]
        structure_matrix[i, np.nonzero(structure_matrix[i, :])[0]] = new_row
    return structure_matrix

