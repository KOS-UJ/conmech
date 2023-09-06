
[/conmech/conmech/dynamics/factory/_dynamics_factory_2d.py](_dynamics_factory_2d.py)

#### `get_edges_features_matrix_numba`

returns:
`edges_features_matrix`, 
`element_initial_volume`, 
`local_stifness_matrices`

The `local_stifness_matrices` is the array of local stifness matrices (w[0, 0], w[0, 1], w[1, 0], w[1, 1]) per mesh element, so the matrices of $e$-th element are

`local_stifness_matrices[0, e]` = 
$
    \begin{bmatrix} 
    \int_{T_e} \frac{\partial \phi_i}{\partial x_1}\frac{\partial \phi_j}{\partial x_1}
    \end{bmatrix}_{i,j=1, \dots, N_e}
$

`local_stifness_matrices[1, e]` = 
$
    \begin{bmatrix} 
    \int_{T_e} \frac{\partial \phi_i}{\partial x_1}\frac{\partial \phi_j}{\partial x_2}
    \end{bmatrix}_{i,j=1, \dots, N_e}
$

`local_stifness_matrices[2, e]` = 
$
    \begin{bmatrix} 
    \int_{T_e} \frac{\partial \phi_i}{\partial x_2}\frac{\partial \phi_j}{\partial x_1}
    \end{bmatrix}_{i,j=1, \dots, N_e}
$

`local_stifness_matrices[3, e]` = 
$
    \begin{bmatrix} 
    \int_{T_e} \frac{\partial \phi_i}{\partial x_2}\frac{\partial \phi_j}{\partial x_2}
    \end{bmatrix}_{i,j=1, \dots, N_e}
$

Where<br>
$e$ is the index of the element<br>
$N_e$ is number of verticecs of element $e$<br>
$T_e$ is the surface of the mesh element $e$<br>
$\phi_j$ is $j$-th base function of the element $e$