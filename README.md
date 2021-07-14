# Tensor Networks Simple-Update
This repo contains an implementation of the Simple-Update Tensor Network algorithm as described in the paper - A universal tensor network algorithm for any infinite lattice by  Saeed S. Jahromi and Roman Orus.

DOI:	10.1103/PhysRevB.99.195105

arXiv link to paper - [A universal tensor network algorithm for any infinite lattice](https://arxiv.org/abs/1808.00680)

## The Code



The folder [`src`](/src) is he heart of this project, here is a description of each file

| #   | file                                         | Subject             | 
|:----:|------------------------------------------------|:-----------------:|
| 1   | `TensorNetwork.py`                   | This is a Tensor Network class object which tracks the tensors, weights and their connectivity| 
| 2   | `SimpleUpdate.py`         | This is a Tensor Network Simple Update algorithm class which get as an input a TensorNetwork object and perform a simple update run on it. | 

### List of Notebooks

| #   | Subject                                         | Colab             | Nbviewer               |
|:----:|------------------------------------------------|:-----------------:|:---------------------:|
| 1   | Paper results reconstruction                   | [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/univeral_tensor_network_paper__reconstruction.ipynb#scrollTo=x9gTThCjbrzm)        | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/RoyElkabetz/Tensor-Networks-Simple-Update/blob/main/univeral_tensor_network_paper__reconstruction.ipynb)|

### Simulations
#### Spin-1/2 Antiferromagnetic Heisenberg (AFH) model

Below are some result of ground-state energy per-site simulated with the Simple Update algorithm over AFH Chain, Star, PEPS and Cube Tensor Networks. The AFH Hamiltonian can be written as

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/a6fe555d0e211c63ed33d4bff13ceff5fe57bbe9/assets/hamiltonian_eq.png" width="" height="60">

In the case of the Star tensor network lattice the AFH Hamiltonian is composite of two part corresponds to different type of edges (see paper in the link above).
The Chain, Star, PEPS and Cube infinite Tensor Networks are illustrated in the next figure.

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/88b9640ad702a74be625b3a0ca0069d6876fc137/assets/Tensor_Networks_diagrams.png" width="" height=""> 


Here are the ground-state energy per-site simulations for the Tensor Networks diagrams above

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/5978bb7b014c41a23fed8996999f07ae1ec58edb/assets/chain_star_peps_cube_plots.png" width="" height="">

#### Quantum Ising Model on a 2D Spin-1/2 Lattice
Next is the quantum Ising model simulated on a 2D lattice 

<img src="https://github.com/RoyElkabetz/Tensor-Networks-Simple-Update/blob/8e87ec1796e62a13c00c77261f04d68d2350443f/assets/Ising_model.png" width="" height="350">


## In Progress....
