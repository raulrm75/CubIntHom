from cells import Chain, AbstractCell, TensorCell
import numpy as np
from itertools import product


class ChainMap(object):
    _src_complex = None
    _dst_complex = None
    _degree = 0
    _matrices = {}

    def __init__(self, src_complex, dst_complex=None, degree=0, matrices=None):
        self._src_complex = src_complex
        if dst_complex is None:
            self._dst_complex = self._src_complex
        else:
            self._dst_complex = dst_complex
        self._degree = degree
        self._matrices = {}
        if matrices is not None:
            self._matrices.update(matrices)

    def __call__(self, arg):
        if arg:
            if arg.dim not in self._matrices:
                return Chain()

            # if isinstance(arg, TensorCell):
            #     return self(arg[0]) @ self(arg[1])
            # elif isinstance(arg, Chain) and arg.cell_class is TensorCell:
            #     result = Chain()
            #     for cell in arg:
            #         result += arg[cell] * (self(cell[0]) @ self(cell[1]))
            #     return result

            zero_chain = Chain()
            if isinstance(arg, int) and arg == 0:
                return zero_chain

            elif isinstance(arg, AbstractCell):
                chain = Chain(arg)
            elif isinstance(arg, Chain):
                chain = arg
            else:
                raise NotImplemented('ChainMap can only evaluate cells or chains.')

            dim = chain.dim
            if dim not in self._matrices:
                return zero_chain
            else:
                M = self._matrices[dim]
                x = chain.to_array(self.src[dim])
                y = np.dot(M, x)
                result = Chain()
                for i in np.where(y)[0]:
                    result[self.dst[dim + self.degree][i]] = y[i, 0]
                return result
        else:
            return Chain()

    @property
    def matrices(self):
        return self._matrices

    def __add__(self, other):
        new_matrices = {}
        for q in set(self.matrices.keys()).union(other.matrices.keys()):
            M = self[q] + other[q]
            if np.any(M):
                new_matrices[q] = M
        return ChainMap(self.src, self.dst, self.degree, new_matrices)

    def __sub__(self, other):
        new_matrices = {}
        for q in set(self.matrices.keys()).union(other.matrices.keys()):
            M = self[q] - other[q]
            if np.any(M):
                new_matrices[q] = M
        return ChainMap(self.src, self.dst, self.degree, new_matrices)

    def __mul__(self, other):
        result = ChainMap(other.src, self.dst,
                          degree=self.degree + other.degree)
        for q in set(self.matrices.keys()).union(other.matrices.keys()):
                if np.any(self[q + other.degree]) and np.any(other[q]):
                    result[q] = np.dot(self[q + other.degree], other[q])
        return result

    def __pow__(self, other):
        if isinstance(other, int) and other >= 1:
            result = self
            for n in range(1, other):
                result = self * result
            return result
        else:
            raise NotImplemented(
                '__pow__ only accepts positive integer exponents.')

    @property
    def src(self):
        return self._src_complex

    @property
    def dst(self):
        return self._dst_complex

    @property
    def degree(self):
        return self._degree

    def __getitem__(self, idx):
        if idx in self._matrices:
            return self._matrices[idx]
        else:
            return np.zeros((self.dst.n[idx + self.degree], self.src.n[idx]), np.int32)

    def __setitem__(self, index, value):
        if index in self._matrices:
            self._matrices[index] = value
        else:
            self._matrices[index] = np.zeros((self.dst.n[index + self.degree], self.src.n[index]), np.int32)
            self._matrices[index] = value

    def __str__(self):
        return '<Chain map: \n{}>'.format('\n'.join(
            ('{} --> {}'.format(q, str(self[q])) for q in
             self.src.dimensions)))

    def __eq__(self, other):
        if isinstance(other, int):
            return all(np.all(self[q] == other) for q in self.matrices)
        elif isinstance(other, ChainMap):
            return all(np.all(self[q] == other[q]) for q in self.dimensions)
        else:
            return False

    def set_image(self, cell, value):
        if value:
            q = cell.dim
            if q not in self._matrices:
                self._matrices[q] = np.zeros((self.dst.n[q + self.degree], self.src.n[q]), np.int32)
            cell_idx = self.src.index(cell)
            if isinstance(value, int):
                self._matrices[q][:, cell_idx] = value
            elif isinstance(value, AbstractCell):
                value_idx = self.dst.index(value)
                self._matrices[q][value_idx, cell_idx] = 1
            elif isinstance(value, Chain):
                y = value.to_array(self.dst[q + self.degree])
                self._matrices[q][:, cell_idx] = y[:, 0]
            else:
                raise NotImplemented()

    def remove_image(self, cell):
        q = cell.dim
        if q in self._matrices:
            cell_idx = self.src.index(cell)
            self._matrices[q][:, cell_idx] = 0

    @classmethod
    def identity(cls, src_complex, dst_complex=None, degree=0):
        if dst_complex is None:
            dst_complex = src_complex
        result = ChainMap(src_complex, dst_complex, degree)
        for q in src_complex.dimensions:
            result[q] = np.eye(
                dst_complex.n[q + degree], src_complex.n[q], dtype=int)
        return result

    def __abs__(self):
        return ChainMap(self.src, self.dst, self.degree, {q: np.abs(M) for q, M in self._matrices.items()})

    def __neg__(self):
        return ChainMap(self.src, self.dst, self.degree, {q: -M for q, M in self._matrices.items()})

    @property
    def dimensions(self):
        return (k for k in self._matrices if np.any(self[k]))

    @property
    def dim(self):
        return max(self.dimensions)

    @property
    def T(self):
        return ChainMap(
            self.dst, self.src,
            matrices={(q + self.degree): M.T for (q, M) in self._matrices.items()},
            degree=-self.degree
        )

    def __matmul__(self, other):
        assert self.degree == other.degree and other.degree == 0
        src = self.src @ other.src
        dst = self.dst @ other.dst
        result = ChainMap(src, dst)
        for cell in src:
            result.set_image(cell, self(cell[0]) @ other(cell[1]))
        return result


def linear_extension(cell_function):
    def result(arg):
        if isinstance(arg, AbstractCell):
            return cell_function(arg)
        else:
            value = Chain()
            for cell in arg:
                value += cell_function(cell) * arg[cell]
            return value
    return result


if __name__ == '__main__':
    from cells import Simplex

    @linear_extension
    def lowest_face(simplex):
        if simplex.dim > 0:
            return Simplex(simplex.vertices[:-1])

    s1 = Simplex((1, 2, 3))
    s2 = Simplex((2, 3, 4))
    print(lowest_face(s1))
    print(lowest_face(s1 + s2))
