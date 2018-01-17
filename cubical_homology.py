import numpy as np
from numba import cuda
from vector_field import VectorField
from math import ceil


@cuda.jit('int32(int32, int32, int32[:], int32[:])', device=True)
def incidence(face_index, facet_index, shape, strides):
    emb = shape.shape[0]
    delta_sum = 0
    acum_dim = 0
    for p in range(emb):
        face_p = face_index // strides[p] % shape[p]
        facet_p = facet_index // strides[p] % shape[p]
        delta_p = facet_p - face_p
        delta_sum += abs(delta_p)
        if abs(delta_p) == 1:
            inc = -(-1) ** acum_dim * delta_p
        acum_dim += facet_p % 2
    if delta_sum == 1:
        return inc
    else:
        return 0


@cuda.jit
def create_CubVF_kernel(index_array, dim_array, direction, dim, shape, strides, matrix):
    i = cuda.grid(1)
    size = index_array.size
    emb = shape.shape[0]
    if i < size and index_array[i] and dim_array[i] == dim:
        j = i + strides[direction]
        if j < size and index_array[j] and dim_array[j] == dim + 1:
            inc = incidence(i, j, shape, strides)
            if abs(inc) == 1:
                matrix[index_array[j] - 1, index_array[i] - 1] = inc
                index_array[i] = 0
                index_array[j] = 0
        cuda.syncthreads()


def create_cubical_vector_field(K):
    C = K.chain_complex()
    V = VectorField(C)

    I = np.zeros_like(K.cells_array)
    D = np.zeros_like(K.cells_array)
    for p in range(K.dim + 1):
        mask = np.where(K.cells_array * K._dim_masks[p])
        size = C.n[p]
        I[mask] = np.arange(1, size + 1)
        D[mask] = p

    I = I.reshape((I.size,)).astype(np.int32)
    D = D.reshape((D.size,)).astype(np.uint8)

    shape = np.array(K.shape, dtype=np.int32)
    strides = np.array(K.strides, dtype=np.int32)

    threadsperblock = 32
    blockspergrid = ceil(I.size / threadsperblock)
    emb = K.emb
    for p in K.dimensions:
        M = np.zeros_like(V[p])
        for direction in range(K.emb - 1, -1, -1):
            create_CubVF_kernel[blockspergrid, threadsperblock](I, D, direction, p, shape, strides, M)
        V[p] = M
    return V


if __name__ == '__main__':
    from cell_complexes import CubicalComplex
    from cells import CubicalCell
    from chain_map import ChainMap
    from chain_complexes import ChainComplex
    from time import time
    from example_complexes import borromean_rings_complement


    # K = CubicalComplex((5, 5))
    # K.add_cell((CubicalCell([(0, 1), (0, 1)])), add_faces=True)
    # K.add_cell((CubicalCell([(0, 1), (1, 2)])), add_faces=True)
    # K.add_cell((CubicalCell([(1, 2), (0, 1)])), add_faces=True)
    K = borromean_rings_complement()

    V = create_cubical_vector_field(K)
    C = V.src
    st = time()
    m_s = {p: np.argwhere(m.sum(axis=0))[:, 0] for p, m in V.matrices.items()}
    n_s = {p: max(1, m.shape[0]) for p, m in m_s.items()}
    pi_s = {p: np.zeros((n, C.n[p]), np.int32) for p, n in n_s.items()}

    m_t = {p + 1: np.argwhere(m.sum(axis=1))[:, 0] for p, m in V.matrices.items()}
    n_t = {p: max(1, m.shape[0]) for p, m in m_t.items()}
    pi_t = {p: np.zeros((n, C.n[p]), np.int32) for p, n in n_t.items()}

    m_c = {p: np.argwhere((abs(V[p].sum(axis=0)) + abs(V[p - 1].sum(axis=1))) == 0)[:, 0] for p in V.matrices}
    n_c = {p: max(1, m.shape[0]) for p, m in m_c.items()}
    pi_c = {p: np.zeros((n, C.n[p]), np.int32) for p, n in n_c.items()}

    for p, m in m_s.items():
        for i, j in enumerate(m):
            pi_s[p][i, j] = 1

    for p, m in m_t.items():
        for i, j in enumerate(m):
            pi_t[p][i, j] = 1

    for p, m in m_c.items():
        for i, j in enumerate(m):
            pi_c[p][i, j] = 1

    cplx_s = ChainComplex(C.cell_class)
    for p, m in m_s.items():
        cplx_s[p] = tuple(C[p][i] for i in m)

    cplx_t = ChainComplex(C.cell_class)
    for p, m in m_t.items():
        cplx_t[p] = tuple(C[p][i] for i in m)

    cplx_c = ChainComplex(C.cell_class)
    for p, m in m_c.items():
        cplx_c[p] = tuple(C[p][i] for i in m)

    pi_s_cm = ChainMap(C, cplx_s, matrices=pi_s)
    pi_t_cm = ChainMap(C, cplx_t, matrices=pi_t)
    pi_c_cm = ChainMap(C, cplx_c, matrices=pi_c)
    print('V Decomposition calculated in {:.3f}s'.format(time() - st))
