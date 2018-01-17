from cells import Chain
from chain_map import ChainMap
import numpy as np
from alexander_whitney import AW


class Reduction(object):
    _src = None
    _dst = None
    _projection = None
    _inclusion = None
    _homotopy = None

    def __init__(self, src, dst, projection=None, inclusion=None,
                 integral=None):
        self._src = src
        self._dst = dst
        self._projection = None
        self._inclusion = None
        self._homotopy = None

        if projection is not None:
            self._projection = projection
        else:
            self._projection = ChainMap(self._src, self._dst)

        if inclusion is not None:
            self._inclusion = inclusion
        else:
            self._inclusion = ChainMap(self._dst, self._src)

        if integral is not None:
            self._homotopy = integral
        else:
            self._homotopy = ChainMap(self._src, self._dst, 1)

    def __mul__(self, other):
        return Reduction(
            other.src,
            self.dst,
            self.projection * other.projection,
            other.inclusion * self.inclusion,
            other.homotopy +
            other.inclusion * self.homotopy * other.projection)

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def projection(self):
        return self._projection

    @property
    def inclusion(self):
        return self._inclusion

    @property
    def homotopy(self):
        return self._homotopy

    def print_test(self):
        h = self.homotopy
        f = self.projection
        g = self.inclusion
        C = self.src
        M = self.dst

        print('f * g == M.id: {}'.format(f * g == M.id()))
        print('C.id - g * f == C.d * h + h * C.d: {}'.format(C.id() - g * f == C.d * h + h * C.d))
        print('h * h == 0: {}'.format(h * h == 0))
        print('f * h == 0: {}'.format(f * h == 0))
        print('h * g == 0: {}'.format(h * g == 0))

    def vector_field(self):
        from vector_field import VectorField
        V = VectorField(self.src)
        for q in self.src.dimensions:
            V[q] = self.homotopy[q] * (np.abs(self.src.d[q + 1].T) != 0).astype(np.int32)
        return V

    def diagonal(self):
        M, f, g = self.dst, self.projection, self.inclusion
        M2 = M @ M
        D = ChainMap(M, M2)
        for cell in M:
            aw = AW(g(cell))
            f_aw = sum((aw[cell] * (f(cell[0]) @ f(cell[1])) for cell in aw), Chain())
            D.set_image(cell, f_aw)
        return D

    def cup_product(self):
        D = self.diagonal()
        M2 = D.dst
        M = D.src
        cup = ChainMap(M2, M)
        for cell in M2:
            t_0, t_1 = cell[0], cell[1]
            c_0, c_1 = Chain(t_0), Chain(t_1)
            p = t_0.dim
            q = t_1.dim

            result = Chain()
            for s in M[p + q]:
                d = D(s)
                result += sum(d[c] * c_0(c[0]) * c_1(c[1]) for c in d) * s
            cup.set_image(cell, result)
        return cup

    def cap_product(self):
        D = self.diagonal()
        M2 = D.dst
        M = D.src
        def cap(cell):
        # cap = ChainMap(M2, M)
        # for cell in M2:
            t_0, t_1 = cell[0], cell[1]
            c_0, c_1 = Chain(t_0), Chain(t_1)
            p = t_0.dim
            q = t_1.dim
            d = D(c_0)
            result = sum((
                (d[c] * c_1(c[1])) * c[0] for c in d if c[1].dim == q
            ), Chain())
            # cap.set_image(cell, result)
            return result
        return cap

