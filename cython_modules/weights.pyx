# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from cython.parallel import prange, parallel
import cython
# import numpy as np
cimport numpy as cnp

ctypedef cnp.npy_bool Bool
ctypedef long Int
ctypedef double Float
# ctypedef int Int
# ctypedef float Float

from libc.math cimport sqrt

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)

cdef inline Int get_abs(Int x) nogil:
    if x > 0:
        return x
    return -x
 
cdef inline Int custom_hash(Int* cell, Int table_size) nogil:
    return get_abs((cell[0] * 92837111) ^ (cell[1] * 689287499) ^ (cell[2] * 283923481)) % table_size


cdef inline void cell_coord(Int* cell, Float* node, Float spacing_inv, Float move = 0) nogil:
    for i in range(3):
        cell[i] = <Int>((node[i] + move) * spacing_inv)

cdef inline Int cell_hash(Float* node, Float spacing_inv, Int table_size) nogil:
    cdef Int[3] cell
    cell_coord(cell, node, spacing_inv)
    return custom_hash(cell, table_size)



cdef inline Float get_min(Float* vector, int size) nogil:
    cdef Float result = vector[0]
    for i in range(1, size):
        if result > vector[i]:
            result = vector[i]
    return result


cdef inline void set_dot(Float* result, Float* matrix, Float* vector, int size) nogil:
    cdef Int i = 0, j = 0, k = 0
    for i in range(size):
         result[i] = 0.
         for j in range(size):
            k = i * size + j
            result[i] += (matrix[k] * vector[j])


cdef inline void set_diff(Float* result, Float* v1, Float* v2, int size) nogil:
    for i in range(size):
        result[i] = v1[i] - v2[i]


cdef Float get_det(Float* matrix) nogil:
    # cdef Float a11 = matrix[0], a12 = matrix[3], a13 = matrix[6]
    # cdef Float a21 = matrix[1], a22 = matrix[4], a23 = matrix[7]
    # cdef Float a31 = matrix[2], a32 = matrix[5], a33 = matrix[8]

    return matrix[0]* matrix[4]*matrix[8] + matrix[3]*matrix[7]*matrix[2] \
        + matrix[6]*matrix[1]*matrix[5] - matrix[6]*matrix[4]*matrix[2] \
        - matrix[3]*matrix[1]*matrix[8] - matrix[0]*matrix[7]*matrix[5]


# cdef Float get_det(Float[:,::1] matrix) nogil:
#     return matrix[0][0]*matrix[1][1]*matrix[2][2] + matrix[0][1]*matrix[1][2]*matrix[2][0] \
#         + matrix[0][2]*matrix[1][0]*matrix[2][1] - matrix[0][2]*matrix[1][1]*matrix[2][0] \
#         - matrix[0][1]*matrix[1][0]*matrix[2][2] - matrix[0][0]*matrix[1][2]*matrix[2][1]


cdef void set_matrix_inverse(Float* result, Float* matrix) nogil:
    cdef Float det = get_det(matrix)
    if (det == 0.0):
        # matrix[:] = 0.
        for i in range(9):
            matrix[i] = 0.
        return
    cdef Float det_inv = 1.0 / det
    
    result[0] =  (matrix[4] * matrix[8] - matrix[7] * matrix[5]) * det_inv
    result[3] = -(matrix[3] * matrix[8] - matrix[6] * matrix[5]) * det_inv
    result[6] =  (matrix[3] * matrix[7] - matrix[6] * matrix[4]) * det_inv
	
    result[1] = -(matrix[1] * matrix[8] - matrix[7] * matrix[2]) * det_inv
    result[4] =  (matrix[0] * matrix[8] - matrix[6] * matrix[2]) * det_inv
    result[7] = -(matrix[0] * matrix[7] - matrix[6] * matrix[1]) * det_inv
	
    result[2] =  (matrix[1] * matrix[5] - matrix[4] * matrix[2]) * det_inv
    result[5] = -(matrix[0] * matrix[5] - matrix[3] * matrix[2]) * det_inv
    result[8] =  (matrix[0] * matrix[4] - matrix[3] * matrix[1]) * det_inv


# cdef void set_matrix_inverse(Float* result, Float[:,::1] matrix) nogil:
#     cdef Float det = get_det(matrix)
#     if (det == 0.0):
#         matrix[:] = 0.
#         return
#     cdef Float det_inv = 1.0 / det

#     result[0] =  (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * det_inv
#     result[1] = -(matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1]) * det_inv
#     result[2] =  (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * det_inv

#     result[3] = -(matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) * det_inv
#     result[4] =  (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * det_inv
#     result[5] = -(matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0]) * det_inv

#     result[6] =  (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * det_inv
#     result[7] = -(matrix[0][0] * matrix[2][1] - matrix[0][1] * matrix[2][0]) * det_inv
#     result[8] =  (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * det_inv



cdef void initialize_hasher(
    Float[:, ::1] nodes,
    Float spacing_inv,
    Int[::1] cell_starts,
    Int[::1] node_cell,
    Int table_size,
    Int nodes_count
) nogil:
    cell_starts[:] = 0
    cdef Float* node
    cdef int i = 0
    cdef int h
    for i in range(nodes_count):
        node = &nodes[i, 0]
        h = cell_hash(node=node, spacing_inv=spacing_inv, table_size=table_size)
        cell_starts[h] += 1

    cdef Int start = 0
    i = 0
    for i in range(table_size):
        start += cell_starts[i]
        cell_starts[i] = start
    cell_starts[table_size] = start  # guard

    i = 0
    for i in range(nodes_count):
        node = &nodes[i, 0]
        h = cell_hash(node=node, spacing_inv=spacing_inv, table_size=table_size)
        cell_starts[h] -= 1
        node_cell[cell_starts[h]] = i



cdef Int query_hasher(
    Int[::1] query_nodes,
    Bool[::1] ready_nodes_mask,
    Float* query_node,
    Float max_dist,
    Int[::1] cell_starts,
    Int[::1] node_cell,
    Int table_size,
    Float spacing_inv) nogil:

    cdef Int h, start, end, node_id
    cdef Int[3] cell_start, cell_stop, cell

    cell_coord(cell=&cell_start[0], node=&query_node[0], spacing_inv=spacing_inv, move = -max_dist)
    cell_coord(cell=&cell_stop[0], node=&query_node[0], spacing_inv=spacing_inv, move = +max_dist)

    cdef Int query_count = 0
    for c1 in range(cell_start[0], cell_stop[0]+1):
        for c2 in range(cell_start[1], cell_stop[1]+1):
            for c3 in range(cell_start[2], cell_stop[2]+1):
                cell[0] = c1
                cell[1] = c2
                cell[2] = c3

                h = custom_hash(cell=cell, table_size=table_size)
                start = cell_starts[h]
                end = cell_starts[h + 1]
                for i in range(start, end):
                    node_id = node_cell[i]
                    if not ready_nodes_mask[node_id]:
                        query_nodes[query_count] = node_id
                        query_count += 1
    return query_count


cdef Float get_norm(Float* vector, int size) nogil:
    cdef Float result = 0.
    for i in range(size):
        result += vector[i]**2
    return sqrt(result)

cdef void set_element_center(Float* element_center, Float* nodes, Int* element, int size) nogil:
    cdef Int node_start = 0, i = 0, j = 0
    for i in range(size):
        element_center[i] = 0.
    for j in range(4):
        node_start = size * element[j]
        for i in range(size):
            element_center[i] += nodes[node_start + i]
    for i in range(size):
        element_center[i] = element_center[i] / 4.


cdef void set_element_matrix_and_node(Float* element_matrix, Float* element_normalizing_node,
            Float* nodes, Int* element) nogil:
    cdef Int i = 0, n = 0
    for i in range(3):
        element_normalizing_node[i] = nodes[3 * element[3] + i]
    for n in range(3):
        for i in range(3):
            element_matrix[3 * i + n] = nodes[3 * element[n] + i] - element_normalizing_node[i]


cdef void complete_to_one(Float* weights, int size) nogil:
    weights[size] = 1.
    for i in range(size):
        weights[size] -= weights[i]

cdef Float get_element_radius(Float* element_center, Float* nodes, Int* element, int size) nogil:
    cdef Int j = 0
    cdef Float maximal_radius = 0, radius
    cdef Float[3] vector

    for j in range(4):
        node_start = size * element[j]
        set_diff(result=vector, v1=&element_center[0], v2=&nodes[node_start], size=3)
        radius = get_norm(vector=vector, size=3)
        if radius > maximal_radius:
            maximal_radius = radius

    return maximal_radius


cpdef void find_closest_nodes_cython(
    Int[:, ::1] closest_nodes,
    Float[:, ::1] closest_weights,
    Float[:, ::1] interpolated_nodes,
    Int[:, ::1] base_elements,
    Float[:, ::1] base_nodes,
    Int[::1] query_nodes,
    Bool[::1] ready_nodes_mask,
    Int[::1] cell_starts,
    Int[::1] node_cell,
    Float spacing,
    Float element_radius_padding
) nogil:
    cdef Int nodes_count = len(interpolated_nodes)
    cdef Int elements_count = len(base_elements)
    cdef Int table_size = len(cell_starts) - 1
    cdef Float spacing_inv = 1 / spacing

    initialize_hasher(nodes=interpolated_nodes, spacing_inv=spacing_inv, cell_starts=cell_starts, node_cell=node_cell, table_size=table_size, nodes_count=nodes_count)
    
    cdef Float[4] weights
    cdef Float[3] vector, element_center, element_normalizing_node
    cdef Float[9] element_matrix, element_matrix_inv
    for k in range(9):
        element_matrix[k] = 0.
        element_matrix_inv[k] = 0.
    cdef Float smallest_weight, smallest_current_weight, element_radius
    cdef Int node_id, i, query_id, query_count

    cdef Int element_id
    # with nogil, parallel():
    # for element_id in prange(elements_count, nogil=True): #num_threads, schedule
    for element_id in range(elements_count):
        # with gil:
        #     print(threadid())

        # make a function
        
        set_element_center(element_center=&element_center[0], nodes=&base_nodes[0][0], element=&base_elements[element_id][0], size=3)
        element_radius = get_element_radius(element_center=&element_center[0], nodes=&base_nodes[0][0], element=&base_elements[element_id][0], size=3)

        set_element_matrix_and_node(element_matrix, element_normalizing_node, nodes=&base_nodes[0][0], element=&base_elements[element_id][0])
        set_matrix_inverse(element_matrix_inv, element_matrix)
        # set_matrix_inverse(element_matrix_inv, &element_nodes_matrices_T[element_id][0][0])
        # set_matrix_inverse(matrix, element_nodes_matrices_T[element_id])

        query_count = query_hasher(
            query_nodes=query_nodes,
            ready_nodes_mask=ready_nodes_mask,
            query_node=element_center,
            max_dist=element_radius + element_radius_padding,
            cell_starts=cell_starts,
            node_cell=node_cell,
            table_size=table_size,
            spacing_inv=spacing_inv
        )

        # with parallel():
        # for query_id in prange(query_count):
        for query_id in range(query_count):
            node_id = query_nodes[query_id]

            # set_diff(result=vector, v1=&interpolated_nodes[node_id][0], v2=&normalizing_element_nodes_T[element_id][0], size=3)
            set_diff(result=vector, v1=&interpolated_nodes[node_id][0], v2=element_normalizing_node, size=3)
            set_dot(result=weights, matrix=element_matrix_inv, vector=vector, size=3)
                    
            # weights sum to one
            complete_to_one(weights=&weights[0], size=3)

            # looking for weights that are closest to positive
            smallest_weight = get_min(vector=&weights[0], size=4)
            smallest_current_weight = get_min(vector=&closest_weights[node_id][0], size=4)

            # better weight found
            if smallest_weight > smallest_current_weight:
                for i in range(4):
                    closest_weights[node_id, i] = weights[i]
                    closest_nodes[node_id, i] = base_elements[element_id][i]
            if smallest_weight >= 0:
                # positive weights found, only one element can contain node
               ready_nodes_mask[node_id] = True

