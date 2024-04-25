from kagome.src import kagome
import numpy as np


def main(n:int)->np.ndarray:
    kagome_lattice, triangular_lattice_of_upper_triangles = kagome.create_kagome_lattice(N=n)
    num_tensors, num_edges = 0, 0
    sm = np.zeros(shape=(num_tensors, num_edges))

    return sm

