"""
## Antiferromagnetic Heisenberg Model PEPS representation 
In This simulation we find the ground state energy and PEPS representation of a $n\times n$ 
Antiferromagnetic Heisenberg model with periodic boundary conditions.
"""

run_from_colab = False
run_jupyter = False

import sys
import os

if run_from_colab:
    # clone the git reposetory
    if run_jupyter:
        pass
        # !git clone https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update

    # add path to .py files for import
    sys.path.insert(1, "/content/Tensor-Networks-Simple-Update/src")

    # mount google drive
    from google.colab import drive
    drive.mount('/content/gdrive')
        
    # path for saving the networks
    save_path = '/content/gdrive/MyDrive/tmp'
else:
    # clone the git reposetory
    if run_jupyter:
        pass
        # !git clone https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update
    
    # add path to .py files for import
    sys.path.insert(1, "Tensor-Networks-Simple-Update/src")
    
    # path for saving the networks
    save_path = '../tmp/networks'
    
if not os.path.exists(save_path):
    os.makedirs(save_path)



"""
### Finding the ground state with Tensor Network Simple Update
"""


import numpy as np
import matplotlib.pyplot as plt
from tnsu.tensor_network import TensorNetwork
import tnsu.simple_update as su
import tnsu.structure_matrix_constructor as smc


np.random.seed(216)

# Pauli matrices
pauli_x = np.array([[0, 1],
                    [1, 0]])
pauli_y = np.array([[0, -1j],
                    [1j, 0]])
pauli_z = np.array([[1, 0],
                    [0, -1]])
s_i = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
s_j = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
s_k = [pauli_x / 2.]

# The Tensor Network structure matrix
n = 2
structure_matrix = smc.square_peps_pbc(n)
print(f'There are {structure_matrix.shape[1]} edges, and {structure_matrix.shape[0]} tensors')

# AFH Hamiltonian interaction parameters
j_ij = [1.] * structure_matrix.shape[1]

# maximal bond dimension
d_max_ = [4]

# convergence error between consecutive lambda weights vectors
error = 1e-5

# maximal number of SU iterations
max_iterations = 200

# time intervals for the ITE
dts = [0.1, 0.01, 0.001, 0.0001, 0.00001]

# magnetic field weight (if 0, there is no magnetic field)
h_k = 0.

energies = []

# Run Simple Update
for d_max in d_max_:
    # create Tensor Network name for saving
    network_name = 'AFH_' + str(n) + 'x' + str(n) + '_pbc_' + 'D_' + str(d_max)

    # create the Tensor Network object
    AFH_TN = TensorNetwork(structure_matrix=structure_matrix,
                           virtual_dim=2,
                           network_name=network_name,
                           dir_path=save_path)

    # create the Simple Update environment
    AFH_TN_su = su.SimpleUpdate(tensor_network=AFH_TN,
                                dts=dts,
                                j_ij=j_ij,
                                h_k=h_k,
                                s_i=s_i,
                                s_j=s_j,
                                s_k=s_k,
                                d_max=d_max,
                                max_iterations=max_iterations,
                                convergence_error=error,
                                log_energy=True,
                                print_process=True)

    # run Simple Update algorithm over the Tensor Network state
    AFH_TN_su.run()

    # compute the energy per-site observable
    energy = AFH_TN_su.energy_per_site()
    print(f'| D max: {d_max} | Energy: {energy}\n')
    energies.append(energy)

    # save the tensor network
    AFH_TN.save_network()


""" 
### Plot the Results
"""

from tnsu.utils import plot_convergence_curve

# plot su convergence / energy curve
plot_convergence_curve(AFH_TN_su)