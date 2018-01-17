from cells import AbstractCell, CubicalCell, Simplex
from chain_complexes import ChainComplex
from chain_map import ChainMap
import numpy as np
from itertools import product, chain
import cubparser
import os.path
from numba import cuda
import math
import random
# from plyfile import PlyData


class CellComplex(object):
    _cells = {}
    _cell_class = None

    def __init__(self, cell_class=None):
        # Cells grouped by dimension.
        # Every element in self._cells[d] must be an AbstractCell instance (or derived class)
        self._cells = {}
        self._cell_class = cell_class

    def __contains__(self, cell):
        return (isinstance(cell, self._cell_class) and
                (cell.dim in self.dimensions) and (cell in self._cells[cell.dim]))

    def star(self, cell):
        for dim in (d for d in self.dimensions if d > cell.dim):
            for facet in self(dim):
                if cell <= facet and cell != facet:
                    yield facet
                else:
                    continue

    @property
    def cell_class(self):
        return self._cell_class

    def facets(self, cell):
        if (cell.dim + 1) in self.dimensions:
            return (c for c in self(cell.dim + 1) if cell < c)
        else:
            return ()

    def add_cell(self, cell, add_faces=False):
        if isinstance(cell, AbstractCell):
            d = cell.dim
            assert self._cell_class == cell.__class__
            if d in self._cells:
                if cell not in self._cells[d]:
                    self._cells[d].append(cell)
            else:
                self._cells[d] = [cell]
            if add_faces:
                for face in cell.boundary():
                    self.add_cell(face, add_faces)
        else:
            self.add_cell(self._cell_class(cell))

    def __call__(self, dimension):
        if dimension in self.dimensions:
            return (cell for cell in self._cells[dimension])
        else:
            return ()

    def __iter__(self):
        for d in self.dimensions:
            for cell in self._cells[d]:
                yield cell

    @property
    def dim(self):
        return max(self.dimensions)

    def chain_complex(self):
        result = ChainComplex(cell_class=self.cell_class)
        for q in self.dimensions:
            result[q] = tuple(cell for cell in self(q))
        diff = ChainMap(result, result, degree=-1)
        for cell in self:
            if cell.dim > 0:
                diff.set_image(cell, cell.differential())
        result.d = diff
        return result

    @property
    def dimensions(self):
        return tuple(d for d in self._cells if self._cells[d])

    def index(self, cell):
        cell_dim = cell.dim
        return sum(len(self._cells[d]) for d in self._cells if d < cell_dim) + self._cells[cell_dim].index(cell)


class SimplicialComplex(CellComplex):
    _vertices = None
    _vertex_count = 0
    # Expected to be a np.array of shape = (nr_vertices, 3)
    _faces = {}
    # Expected to be a dict of np.array of shape = (nr_vertices, dim + 1) where dim is the key
    _face_index = {}
    # Expected to be a dict of np.array of shape = (nr_vertices,) * (dim + 1)
    # _face_index[2][i, j] = k iff _faces[2][k] = [i, j]
    _dim = -1
    _face_count = {}

    def __init__(self):
        super(SimplicialComplex, self).__init__(Simplex)
        self._vertices = None
        self._vertex_count = 0
        self._faces = {}
        self._face_index = {}
        self._dim = -1
        self._face_count = {}

    @classmethod
    def from_ply_mesh(cls, file_name):
        simplicial_complex = SimplicialComplex()
        with open(file_name, 'rb') as f:
            plydata = PlyData.read(f)
            nr_points = plydata.elements[0].count
            simplicial_complex._vertex_count = nr_points
            simplicial_complex._vertices = np.vstack(
                (plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).T
            simplicial_complex._faces[2] = np.vstack(plydata['face'].data['vertex_indices'])
            simplicial_complex._face_count[0] = nr_points
            simplicial_complex._face_count[2] = simplicial_complex._faces[2].shape[0]
            edges = set()
            for i in range(simplicial_complex._face_count[2]):
                face = tuple(simplicial_complex._faces[2][i])
                for j in range(3):
                    edge = face[:j] + face[j+1:]
                    edges |= {edge}
            simplicial_complex._faces[1] = np.vstack(tuple(edges))
            simplicial_complex._face_count[1] = simplicial_complex._faces[1].shape[0]
            return simplicial_complex

    @classmethod
    def from_maximal_faces(cls, face_iterable):
        pass

    def __iter__(self):
        if self._vertex_count:
            for i in range(self._vertex_count):
                yield Simplex((i,))
            dim = max(self._face_count)
            for d in range(1, dim + 1):
                if d in self._face_count[d]:
                    for i in range(self._face_count[d]):
                        yield Simplex(tuple(self._faces[d][i]))

def cell_models(dimension):
    models = {d: [] for d in range(dimension + 1)}
    for cell in product(range(2), repeat=dimension):
        models[sum(cell)].append(cell)
    return models


def incidence(face_cmap, cell_cmap):
    face = np.array(face_cmap)
    cell = np.array(cell_cmap)
    delta = cell - face
    if np.sum(np.abs(delta)) == 1:
        p = np.where(delta)[0][0]
        i = delta[p]
        d = np.sum((cell % 2)[:p])
        return -i * (-1) ** d
    else:
        return 0


@cuda.jit
def create_cubical_differential_array(src, dst, diff):
    emb = src.shape[1]
    i, j = cuda.grid(2)
    if i >= src.shape[0] or j >= dst.shape[0]:
        return
    s = 0
    for x in range(emb):
        s += abs(src[i, x] - dst[j, x])
    if s == 1:
        for x in range(emb):
            if src[i, x] != dst[j, x]:
                p = x
                break
        v = src[i, p] - dst[j, p]
        d = 0
        for x in range(p):
            d += src[i, x] % 2
        inc = -v * (-1) ** d
    else:
        inc = 0
    diff[j, i] = inc


@cuda.jit
def close_boundary(flat_cells_array, strides, shape):
    i, j = cuda.grid(2)
    if i >= flat_cells_array.size or j >= flat_cells_array.size:
        return

    if not flat_cells_array[i]:
        return

    delta = 0
    for p in range(strides.shape[0]):
        i_p = (i // strides[p]) % shape[p]
        j_p = (j // strides[p]) % shape[p]
        if i_p != j_p:
            delta += 1
    if delta == 1:
        flat_cells_array[j] = 1


class CubicalComplex(CellComplex):
    _cells_array = None
    _strides = None
    _dim = None

    def __init__(self, shape=None):
        super(CubicalComplex, self).__init__(CubicalCell)
        self._cells_array = None
        self._dim_masks = {}
        if shape is not None:
            self.shape = shape

    def _create_dim_masks(self):
        models = cell_models(self.emb)
        indices = {dimension: [
            tuple(
                slice(0, s, 2) if m == 0 else slice(1, s, 2)
                for (m, s) in zip(model, self.shape))
            for model in models[dimension]] for dimension in range(self.emb + 1)}
        result = {dim: np.zeros_like(self._cells_array) for dim in range(self.emb + 1)}
        for dim in result:
            for slice_index in indices[dim]:
                result[dim][slice_index] = 1
        self._dim_masks = result

    @property
    def shape(self):
        if self._cells_array is None:
            return ()
        else:
            return self._cells_array.shape

    @shape.setter
    def shape(self, shape):
        # assert all(i % 2 == 1 for i in shape)
        if self._cells_array is not None:
            if shape is None:
                self._cells_array = None
            else:
                self._cells_array.shape = shape
        elif shape is not None:
            self._cells_array = np.zeros(shape, np.int32)

        if self._cells_array is not None:
            self._strides = (np.array(self._cells_array.strides) //
                             self._cells_array.itemsize)

        if shape is not None:
            self._create_dim_masks()

    @property
    def cells_array(self):
        return self._cells_array

    def chain_complex(self):
        result = ChainComplex(cell_class=CubicalCell)
        for q in self.dimensions:
            result[q] = tuple(cell for cell in self(q))

        matrices = {}
        for q in range(1, self.dim + 1):
            src = np.array([cell.cube_map for cell in result[q]]).astype(np.int32)
            dst = np.array([cell.cube_map for cell in result[q - 1]]).astype(np.int32)
            threadsperblock = (16, 16)
            blockspergrid_src = math.ceil(src.shape[0] / threadsperblock[0])
            blockspergrid_dst = math.ceil(dst.shape[0] / threadsperblock[1])
            blockspergrid = (blockspergrid_src, blockspergrid_dst)
            diff = np.zeros((dst.shape[0], src.shape[0]), dtype=np.int32)
            create_cubical_differential_array[blockspergrid, threadsperblock](src, dst, diff)
            matrices[q] = diff
        result.d = ChainMap(result, result, degree=-1, matrices=matrices)
        return result

    def add_cell(self, cell, add_faces=False):
        if not isinstance(cell, CubicalCell):
            raise TypeError(
                'Only CubicalCell can be added to a CubicalComplex,'
                ' rather than {}'.format(type(cell)))
        else:
            if self.shape is None:
                raise ValueError('The CubicalComplex has no initial shape.')
            else:
                self._cells_array[cell.cube_map] = 1
                if self._dim is None:
                    self._dim = cell.dim
                else:
                    self._dim = max(self._dim, cell.dim)
                if add_faces:
                    for intervals in product(*[
                            ((I[0], I[0]), (I[1], I[1]), I) for I in
                            cell.intervals]):
                        face = CubicalCell(intervals)
                        if face != cell:
                            self.add_cell(face)

    def __contains__(self, cell):
        if not isinstance(cell, CubicalCell):
            return False
        try:
            return bool(self._cells_array[cell.cube_map])
        except IndexError:
            return False

    @property
    def emb(self):
        try:
            return len(self.shape)
        except TypeError:
            return 0

    @property
    def dimensions(self):
        return range(self.dim + 1)

    def __call__(self, dimension):
        if dimension in self.dimensions:
            for idx in zip(*np.where(
                    self._cells_array * self._dim_masks[dimension])):
                yield CubicalCell(idx)
        else:
            return ()

    @classmethod
    def from_file(cls, cub_file):
        _, ext = os.path.splitext(cub_file)
        if ext == '.cub':
            parser = cubparser.vertices_from_file
        elif ext == '.cel':
            parser = cubparser.intervals_from_file

        cube_maps, shape = parser(cub_file)

        K = CubicalComplex(np.array(shape) + 2)
        for cube_map in cube_maps:
            for f in product(*[[(n,)] if n % 2 == 0 else [(n - 1,), (n,), (n + 1,)] for n in cube_map]):
                # face = tuple(zip(*f))[0]
                K._cells_array[f] = 1
                # print(f)
                dim_f = sum(i[0] % 2 for i in f)
                if K._dim is None:
                    K._dim = dim_f
                else:
                    K._dim = max(K._dim, dim_f)
        return K

    @classmethod
    def random(cls, shape, cell_count=None):
        new_shape = tuple(s if s % 2 == 1 else s + 1 for s in shape)
        K = CubicalComplex(new_shape)
        if cell_count is None:
            cell_count = random.randint(0, K.cells_array.size)
        else:
            cell_count = min(cell_count, K.cells_array.size)
        cells = random.sample(list(product(*[range(s)for s in K.shape])), cell_count)
        for cell in cells:
            K.add_cell(CubicalCell(cell), add_faces=True)

        return K

    def to_file(self, file_name):
        with open(file_name, 'w') as f:
            for cell in self:
                f.write('{}\n'.format(cell))

    @property
    def max_index(self):
        return self._cells_array.size

    @property
    def strides(self):
        A = self._cells_array
        return np.array(A.strides) // A.itemsize

    def _recalc_dim(self):
        dims = ([(d, np.any(self._cells_array * self._dim_masks[d])) for d in range(self.emb + 1)])
        self._dim = max(d for d, b in dims if b)

    @property
    def dim(self):
        if self._dim is None:
            self._recalc_dim()
            return self._dim
        else:
            return self._dim

    def __getitem__(self, index):
        cell = tuple(index // s % self.shape[i] for (i, s) in enumerate(self.strides))
        if self._cells_array[cell]:
            return CubicalCell(cell)
        else:
            return None

    def index(self, cell):
        return np.sum(self.strides * np.array(cell.cube_map))

    def __len__(self):
        return np.sum(self._cells_array)

    def __mul__(self, other):
        if not isinstance(other, CubicalComplex):
            raise ValueError('Cartesian product can only be calculated within cubical complexes')
        else:
            M = np.multiply.outer(self._cells_array, other._cells_array)
            result = CubicalComplex(M.shape)
            result._cells_array = M
            return result

    def __iter__(self):
        yield from chain.from_iterable((self(p) for p in range(self.dim + 1)))

    def complement(self):
        new_shape = tuple(np.array(self.shape) + 4)
        A = self._cells_array
        B = self.boundary()._cells_array
        K = CubicalComplex(new_shape)
        K._cells_array = np.ones_like(K._cells_array)
        K._cells_array[tuple(slice(2, s + 2) for s in self.shape)] = np.ones_like(self._cells_array) - (1 - B) * A
        return K

    def boundary(self):
        K = CubicalComplex(self.shape)
        M = self._cells_array
        A = {p: M * self._dim_masks[p] for p in self.dimensions}
        S = np.zeros_like(M)
        D = self._dim_masks
        Z = np.zeros_like(M)

        for p in range(S.ndim):
            for q in range(S.ndim):
                S += shift(shift(A[p], q) * A[p + 1], q, False) * D[p]
                S += shift(shift(A[p], q, False) * A[p + 1], q) * D[p]
        S = (S == 1).astype(np.int32)  # S represents free faces

        K._cells_array = S
        for cell in K:
            K.add_cell(cell, add_faces=True)


        return K


def shift(array, axe, positive=True):
    shape = array.shape
    result = np.zeros_like(array)
    slices = tuple(slice(0, s, 1) for s in shape)

    if positive:
        src_slices = slices[: axe] + (slice(0, shape[axe] - 1, 1),) + slices[axe + 1:]
        dst_slices = slices[: axe] + (slice(1, shape[axe], 1),) + slices[axe + 1:]
        result[dst_slices] = array[src_slices]
    else:
        src_slices = slices[: axe] + (slice(1, shape[axe], 1),) + slices[axe + 1:]
        dst_slices = slices[: axe] + (slice(0, shape[axe] - 1, 1),) + slices[axe + 1:]
        result[dst_slices] = array[src_slices]
    return result

if __name__ == '__main__':
    K = CubicalComplex((5, 5, 3))
    K.add_cell(CubicalCell([(0, 1), (0, 1), (0, 1)]), add_faces=True)
    K.add_cell(CubicalCell([(1, 2), (0, 1), (0, 1)]), add_faces=True)
    K.add_cell(CubicalCell([(0, 1), (1, 2), (0, 1)]), add_faces=True)

    M = K.cells_array
    # print(M)
    L = K.boundary()
