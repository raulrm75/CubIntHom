# -*- coding: utf-8 -*-
from cells import Chain, AbstractCell, CubicalCell, Simplex, TensorCell
from itemproperties import itemproperty
import CHomP
import numpy as np
from chain_map import ChainMap, linear_extension
from reduction import Reduction
from itertools import product, chain


def to_subscript(n):
    return ''.join('₀₁₂₃₄₅₆₇₈₉'[int(ch)] for ch in str(n))


def to_superscript(n):
    return ''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(ch)] for ch in str(n))


class ChainComplex(object):
    _modules = {}
    _differential = None
    _cell_class = None
    _D = {}  # self.D[q] is SNF of self.d[q]
    _A = {}  # self.A[q] == np.linalg.inv(self.G[q])
    _G = {}
    _degree = -1
    # self.G[q - 1].dot(self.D[q]).dot(self.A[q]) == self.d[q]
    # self.A[q - 1].dot(self.d[q]).dot(self.G[q]) == self.D[q]
    _computedSNF = False

    def __init__(self, cell_class):
        self._modules = {}
        self._differential = None
        self._cell_class = cell_class
        self._D = {}
        self._A = {}
        self._G = {}
        self._computedSNF = False
        self._degree = -1

    def __getitem__(self, index):
        '''
        :param index: the index of the Z-module to be got.
        :return: a tuple of cells that generate the index-th module of the chain complex.
        '''

        if index in self._modules:
            return self._modules[index]
        else:
            return ()

    def __setitem__(self, index, value):
        '''
        :param index: the index of the Z-module to be set.
        :param value: a tuple of cells or tuples (representing a cell).
        :return: None
        '''
        if value:
            self._modules[index] = value
        elif index in self._modules:
            del self._modules[index]

    def __eq__(self, other):
        result = self.dim == other.dim
        if not result:
            return False

        for q in range(self.dim + 1):
            result = result and self[q] == other[q]
            if not result:
                return False

        for q in range(self.dim + 1):
            result = result and np.all(self.d[q] == other.d[q])
            if not result:
                return False

        return result

    @itemproperty
    def n(self, q):
        '''

        :param q:the index of the correspondind Z-module.
        :return: the number of elements in the basis of corresponding module.
        '''
        return max(1, len(tuple(self[q])))

    @property
    def d(self):
        return self._differential

    @d.setter
    def d(self, chain_map):
        self._differential = chain_map

    @property
    def dim(self):
        if self._modules:
            return max(k for k in self._modules if self._modules[k])
        else:
            return -1

    @property
    def dimensions(self):
        return (k for k in self._modules if self._modules[k])

    @itemproperty
    def D(self, q):
        if q in self._D:
            return self._D[q]
        else:
            return np.zeros((self.n[q + self._degree], self.n[q]), np.int32)

    @itemproperty
    def A(self, q):
        if q in self._A:
            return self._A[q]
        else:
            return np.zeros((self.n[q], self.n[q]), np.int32)

    @itemproperty
    def G(self, q):
        if q in self._G:
            return self._G[q]
        else:
            return np.zeros((self.n[q], self.n[q]), np.int32)

    def computeSNF(self):
        if not self._computedSNF:
            MSNF = CHomP.MatrixSNF()
            for q in range(self.dim + 1):
                MSNF.matrices[q] = CHomP.MMatrix(self.d[q])
            MSNF.computeSNF()
            for q in range(self.dim + 1):
                self._D[q] = MSNF.matrices[q]._data
                self._A[q] = MSNF.chgBasisA[q]._data
                self._G[q] = MSNF.chgBasisG[q]._data
            self._computedSNF = True

    def is_minimal(self):
        return not any(np.any(np.abs(self.d[q]) == 1) for q in self.dimensions)

    def __iter__(self):
        for q in self._modules:
            for cell in self._modules[q]:
                yield cell

    def id(self):
        return ChainMap.identity(self)

    @property
    def cell_class(self):
        return self._cell_class

    def index(self, cell):
        return self._modules[cell.dim].index(cell)

    def SNF_reduction(self):
        while True:
            self._computedSNF = False
            self.computeSNF()
            all_diag = True
            for p in self.dimensions:
                T = np.zeros_like(self.D[p])
                T[:min(self.D[p].shape), :min(self.D[p].shape)] = np.diag(np.diag(self.D[p]))
                all_diag = all_diag and np.all(T == self.D[p])
                if not all_diag:
                    continue
            if all_diag:
                break

        f_A = ChainMap(self, matrices={q: self.A[q] for q in self.dimensions})
        f_G = ChainMap(self, matrices={q: self.G[q] for q in self.dimensions})
        f_D = ChainMap(self, matrices={q: self.D[q] for q in self.dimensions}, degree=self._degree)

        M = ChainComplex(cell_class=self.cell_class)
        for q in self.dimensions:
            M[q] = self[q]
        M.d = f_D

        return Reduction(
            src=self, dst=M,
            projection=f_A,
            inclusion=f_G,
            integral=ChainMap(self, degree=-self._degree)
        )

    @property
    def modules(self):
        return self._modules

    def dual(self):
        result = CochainComplex(self.cell_class)
        codifferential = self.d.T
        for q in self.dimensions:
            result[q] = self[q]
        result.d = codifferential
        return result

    def pprint(self, arg, cell_char='\u03C3'):
        if isinstance(arg, self._cell_class):
            full_index = sum(len(self._modules[d]) for d in self._modules if d < arg.dim)
            return '{}{}'.format(cell_char, to_subscript(full_index + self.index(arg)))
        elif isinstance(arg, Chain):
            return str(arg.change_support(lambda x: self.pprint(x, cell_char)))
        else:
            raise NotImplementedError()

    def __matmul__(self, other):
        assert isinstance(other,  self.__class__)
        result = ChainComplex(cell_class=TensorCell)

        dimensions = {}
        for i, j in product(self.dimensions, other.dimensions):
            if i + j in dimensions:
                dimensions[i + j].append((i, j))
            else:
                dimensions[i + j] = [(i, j)]

        for p in dimensions:
            result[p] = tuple(map(lambda l: l[0] @ l[1],
                                  chain(*[product(self[i], other[j]) for i, j in dimensions[p]])))
        d = ChainMap(result, degree=-1)
        # for cell1, cell2 in product(self, other):
        #     t = cell1 @ cell2
        #     if t.dim > 0:
        #         d_t = t.differential()
        #         d.set_image(t, d_t)
        result.d = d
        return result


class CochainComplex(ChainComplex):

    def __init__(self, cell_class):
        super().__init__(self, cell_class)
        self._degree = +1

    def SNF_reduction(self):
        while True:
            self._computedSNF = False
            self.computeSNF()
            all_diag = True
            for p in self.dimensions:
                T = np.zeros_like(self.D[p])
                T[:min(self.D[p].shape), :min(self.D[p].shape)] = np.diag(np.diag(self.D[p]))
                all_diag = all_diag and np.all(T == self.D[p])
                if not all_diag:
                    continue
            if all_diag:
                break

        f_A = ChainMap(self, matrices={q: self.A[q].T for q in self.dimensions})
        f_G = ChainMap(self, matrices={q: self.G[q].T for q in self.dimensions})
        f_D = ChainMap(self, matrices={q: self.D[q].T for q in self.dimensions}, degree=self._degree)

        M = ChainComplex(cell_class=self.cell_class)
        M._degree = +1
        for q in self.dimensions:
            M[q] = self[q]
        M.d = f_D

        return Reduction(
            src=self, dst=M,
            projection=f_G,
            inclusion=f_A,
            integral=ChainMap(self, degree=-self._degree)
        )

if __name__ == '__main__':
    from cell_complexes import CellComplex
    from vector_field import create_vector_field

    K = CellComplex(Simplex)
    cells = {
        0: [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)],
        1: [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
            (1, 2), (1, 5), (1, 7), (1, 8),
            (2, 5), (2, 6), (2, 8),
            (3, 4), (3, 5), (3, 8),
            (4, 5), (4, 6), (4, 7), (4, 8),
            (5, 6), (5, 7), (5, 9),
            (6, 8), (6, 9),
            (7, 8), (7, 9),
            (8, 9)],
        2: [(0, 1, 5), (1, 2, 5), (0, 2, 6), (0, 3, 5), (2, 5, 6), (0, 4, 6),
            (3, 4, 5), (5, 6, 9), (4, 6, 8), (4, 5, 7), (5, 7, 9), (7, 8, 9), (6, 8, 9), (3, 4, 8),
            (0, 4, 7), (1, 7, 8), (0, 3, 8), (0, 1, 7), (1, 2, 8), (0, 2, 8)]
    }
    for q in cells:
        for cell in cells[q]:
            K.add_cell(Simplex(cell))

    C = K.chain_complex()
    V = create_vector_field(C)
    am = V.am_model()
    f, g, h, M = am.projection, am.inclusion, am.homotopy, am.dst
    d = C.d
