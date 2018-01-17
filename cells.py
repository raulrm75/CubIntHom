from abc import abstractmethod
from functools import total_ordering
from collections import Iterable
import inspect
import numpy as np
from itertools import product, chain, combinations

class AbstractCell(object):
    '''
    Base class for all cells. Currently, Simplex and CubicalCell.
    '''
    _cell_complex = None
    _chain_complex = None

    @abstractmethod
    def __str__(self):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass

    @abstractmethod
    def differential(self):
        pass

    @abstractmethod
    def boundary(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __lt__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    def __add__(self, other):
        if isinstance(other, Chain):
                return other + Chain(self)
        elif isinstance(other, self.__class__):
            return Chain(self) + Chain(other)
        else:
            raise ValueError('Wrong class of arguments: {} and {}.'.format(self.__class__, type(other)))

    def __sub__(self, other):
        if isinstance(other, Chain):
            return - other + Chain(self)
        elif isinstance(other, self.__class__):
            return Chain(self) - Chain(other)
        else:
            raise ValueError('Wrong class of arguments: {} and {}.'.format(self.__class__, type(other)))

    def __neg__(self):
        return - Chain(self)

    def __mul__(self, other):
        if isinstance(other, int):
            return other * Chain(self)
        else:
            raise ValueError('Wrong class of arguments: {} and {}.'.format(self.__class__, type(other)))

    def __rmul__(self, other):
        return int(other) * Chain(self)

    def __matmul__(self, other):
        if isinstance(other, AbstractCell):
            return TensorCell(self, other)
        elif isinstance(other, Chain):
            return Chain({self @ tau: other[tau] for tau in other})


class Chain(object):
    _coeff = {}
    _chain_complex = None

    def __init__(self, arg=None):
        self._coeff = {}
        if arg is None:
            self._coeff = {}
        elif isinstance(arg, AbstractCell):
            self._coeff = {arg: 1}
        elif isinstance(arg, dict):
            self._coeff.update(arg)
        elif isinstance(arg, Chain):
            self._coeff.update(arg._coeff)
        else:
            raise SyntaxError(
                'A valid cellular object, dictionary or Chain' +
                'must be provided, instead of {}.'.format(type(arg)))

    def __hash__(self):
        return hash(tuple(hash(cell) for cell in self._coeff))

    def __len__(self):
        return len(self._coeff)

    def __iter__(self):
        for cell in self._coeff:
            yield cell

    def __getitem__(self, cell):
        assert isinstance(cell, AbstractCell)
        return int(self._coeff.get(cell, 0))

    def __setitem__(self, cell, value):
        assert isinstance(cell, AbstractCell), '{} is {} instead of AbstractCell'.format(cell, cell.__class__)
        self._coeff[cell] = int(value)

    def __add__(self, other):
        if isinstance(other, Chain):
            other_chain = other
        else:
            other_chain = Chain(other)

        if other_chain == 0:
            return self
        else:
            result = Chain(self)
            for cell in other_chain._coeff:
                if cell in result._coeff:
                    result._coeff[cell] += other_chain._coeff[cell]
                else:
                    result._coeff[cell] = other_chain._coeff[cell]

            null_cells = [cell for cell, coeff in result._coeff.items() if coeff == 0]

            for cell in null_cells:
                del result._coeff[cell]

            return result

    def __sub__(self, other):
        if isinstance(other, Chain):
            other_chain = other
        else:
            other_chain = Chain(other)

        if other_chain == 0:
            return self
        else:
            result = Chain(self)
            for cell in other_chain._coeff:
                if cell in result._coeff:
                    result._coeff[cell] -= other_chain._coeff[cell]
                else:
                    result._coeff[cell] = -other_chain._coeff[cell]

            null_cells = [cell for cell, coeff in result._coeff.items() if coeff == 0]

            for cell in null_cells:
                del result._coeff[cell]

            return result

    def __mul__(self, other):
        if isinstance(other, int):
            if self == 0:
                return self
            else:
                result = Chain()
                if other == 0:
                    return result
                else:
                    result._coeff.update(self._coeff)
                    for cell in result._coeff:
                        result._coeff[cell] *= other
                    return result
        else:
            raise NotImplemented('Only integers and chains can be multiplied.')

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        result = Chain(self)
        for cell in result._coeff:
            result._coeff[cell] *= (-1)
        return result

    def __repr__(self):
        return '<Chain: {}>'.format(self)

    def __str__(self):
        if self == 0:
            return '0'
        else:
            data = []
            for cell, coeff in self._coeff.items():
                sign = '+' if coeff >= 0 else '-'
                data.append((sign, coeff, cell))
            terms = ['{} {}{}'.format(
                sign,
                '' if abs(coeff) == 1 else '{} · '.format(abs(coeff)), cell)
                for sign, coeff, cell in data]
            result = ' '.join(terms)
            if result[0] == '+':
                return result[1:]
            else:
                return result

    def __len__(self):
        return len(self._coeff)

    def __delitem__(self, key):
        del self._coeff[key]

    @property
    def dim(self):
        if self._coeff:
            first_cell = next(iter(self._coeff.keys()))
            # print('DEBUG: first_cell dim is dim({}) = {}'.format(first_cell, first_cell.dim))
            return first_cell.dim
        else:
            return -1

    @property
    def cell_class(self):
        if self._coeff:
            return next(iter(self._coeff.keys())).__class__
        else:
            return AbstractCell

    def differential(self):
        result = Chain()
        for cell, coeff in self._coeff.items():
            diff = cell.differential()
            result += coeff * diff
        return result

    def __eq__(self, other):
        if isinstance(other, int):
            return other == 0 and not self._coeff
        elif isinstance(other, Chain):
            return self._coeff == other._coeff
        else:
            return False

    def __ne__(self, other):
        return not (self == other)

    def dot(self, other):
        if isinstance(other, Chain):
            dot_support = set(self._coeff.keys()).intersection(other._coeff.keys())
            return sum(self._coeff[cell] * other._coeff[cell] for cell in dot_support)
        elif isinstance(other, AbstractCell):
            return self[other]
        else:
            raise NotImplemented('<·, ·> is only implemented for chains.')

    def coeffs(self):
        return set(self._coeff.values())

    def keys(self):
        return (key for key in self._coeff)

    def change_support(self, func):
        return Chain({func(c): self[c] for c in self})

    def filtered_by(self, filter):
        return Chain({c: self[c] for c in self if filter(c)})

    def to_array(self, basis):
        result = np.zeros((len(basis), 1), np.int32)
        for cell in self:
            if not cell in basis:
                print('ERROR: {} not in {}'.format(cell, str(basis)))
            result[basis.index(cell), 0] = self[cell]
        return result

    def pop(self):
        cell, value = self._coeff.popitem()
        return cell, value

    def __matmul__(self, other):
        # Tensor product of chains
        result = Chain()
        if not isinstance(other, Chain):
            other = Chain(other)

        for sigma, tau in product(self, other):
            result += (self[sigma] * other[tau]) * (sigma @ tau)
        return result

    def __call__(self, other):
        if isinstance(other, AbstractCell):
            return self._coeff.get(other, 0)
        elif isinstance(other, Chain):
            return sum(other[cell] * self._coeff.get(cell, 0) for cell in other)
        else:
            return 0


@total_ordering
class Simplex(AbstractCell):
    _vertices = ()

    def __init__(self, vertices):
        self._vertices = tuple(sorted(vertices))

    def __str__(self):
        return '<{}>'.format(', '.join(str(v) for v in self._vertices))

    def __repr__(self):
        return '<Simplex: [{}]>'.format(', '.join(str(v) for v in self._vertices))

    @property
    def dim(self):
        return len(self._vertices) - 1

    @property
    def vertices(self):
        return self._vertices

    def differential(self):
        result = Chain()
        if self.dim > 0:
            sign = +1
            for i, s in enumerate(self._vertices):
                face = Simplex(self._vertices[:i] + self._vertices[i + 1:])
                result[face] = sign
                sign *= -1
        return result

    def boundary(self):
        for i, s in enumerate(self._vertices):
            yield Simplex(self._vertices[:i] + self._vertices[i + 1:])

    def __eq__(self, other):
        if isinstance(other, Simplex):
            return self._vertices == other._vertices
        else:
            return self == Simplex(other)

    def __lt__(self, other):
        return set(self._vertices) < set(other.vertices)

    def __hash__(self):
        return hash(self._vertices)

    def __getitem__(self, item):
        return self._vertices[item]

    def AW(self):
        n = self.dim
        # return sum((
        #     (-1) ** ((n * (n + 1)) // 2 - k) * Simplex(self.vertices[:k + 1]) @ Simplex(self.vertices[k:])
        #     for k in range(n + 1)), Chain())
        return sum((
            Simplex(self.vertices[:k + 1]) @ Simplex(self.vertices[k:])
            for k in range(n + 1)), Chain())

int_types = (int, np.int8, np.int16, np.int32, np.int64, np.uint8,
    np.uint16, np.uint32, np.uint64)


def is_cube_map(iterable):
    '''
    Test whether an iterable is a cube_map, i.e., it is a list of integers.
    '''
    return (isinstance(iterable, Iterable) and
        all(isinstance(i, int_types) for i in iterable))


def is_interval_list(iterable):
    '''
    Test whether an iterable is an interval list, i.e., it is a list
    of pairs of consecutive integers.
    '''
    return (isinstance(iterable, Iterable) and
        all(isinstance(I, Iterable) and 1 <= len(I) <= 2 and
                isinstance(max(I), int_types) and
                isinstance(min(I), int_types) and
                0 <= max(I) - min(I) <= 1 for I in iterable))


def interval_to_str(interval):
    if min(interval) == max(interval):
        return '({})'.format(min(interval))
    else:
        return '({},{})'.format(min(interval), max(interval))


class CubicalCell(AbstractCell):
    _intervals = None
    _cube_map = None
    _emb = None
    _dim = None
    _center = None
    _index = None

    def __init__(self, arg):
        if is_interval_list(arg):
            self._emb = len(arg)
            self._intervals = tuple((min(I), max(I)) for I in arg)
            self._cube_map = tuple(I[0] + I[1] for I in self._intervals)
            self._dim = sum(i % 2 for i in self.cube_map)
            self._center = tuple(
                (min(I) + max(I)) / 2 for I in self._intervals)
            self._index = None
        elif is_cube_map(arg):
            self._emb = len(arg)
            self._cube_map = arg
            self._intervals = tuple(
                (i // 2, i // 2) if i % 2 == 0 else
                ((i - 1) // 2, (i + 1) // 2)
                for i in self.cube_map)
            self._dim = sum(i % 2 for i in self.cube_map)
            self._center = tuple((min(I) + max(I)) / 2 for I in
                self._intervals)
            self._index = None
        else:
            raise TypeError('An appropiate argument must be provided instead '
                'of {} of {}.'.format(type(arg), type(arg[0])))

    def __str__(self):
        if self._intervals:
            return "x".join(interval_to_str(I) for I in self._intervals)
        else:
            return "()"

    def __repr__(self):
        return '<CubicalCell: {}>'.format(self)

    @property
    def dim(self):
        return self._dim

    @property
    def emb(self):
        return len(self._intervals)

    def differential(self):
        bdry = {}
        cmap = self.cube_map
        s = 1
        for p, c in enumerate(c for c in cmap):
            if c % 2 == 1:
                pos_bdry = cmap[:p] + (cmap[p] - 1,) + cmap[p+1:]
                neg_bdry = cmap[:p] + (cmap[p] + 1,) + cmap[p+1:]

                bdry[pos_bdry] = s
                bdry[neg_bdry] = -s
                s *= -1
        return Chain({CubicalCell(key): bdry[key] for key in bdry})

    def boundary(self):
        bdry = {}
        cmap = self.cube_map
        for p, c in enumerate(c for c in cmap):
            if c % 2 == 1:
                yield CubicalCell(cmap[:p] + (cmap[p] - 1,) + cmap[p+1:])
                yield CubicalCell(cmap[:p] + (cmap[p] + 1,) + cmap[p+1:])

    def __eq__(self, other):
        if is_cube_map(other):
            return self._cube_map == other
        elif is_interval_list(other):
            return self._intervals == other
        elif isinstance(other, CubicalCell):
            return np.all(self._cube_map == other.cube_map)
        else:
            return False

    def __ne__(self, other):
        return not (self == other)

    def __le__(self, other):
        if is_cube_map(other) or is_interval_list(other):
            if len(other) == self._emb:
                return self <= CubicalCell(other)
            else:
                raise TypeError('Uncomparable cubic cells because of its '
                                'different embedding numbers.')
        elif isinstance(other, CubicalCell):
            if self._emb == other._emb:
                return all(set(I1) <= set(I2) for (I1, I2) in
                    zip(self._intervals, other._intervals))
            else:
                raise TypeError('Uncomparable cubic cells because of its '
                                'different embedding numbers.')

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        if is_cube_map(other) or is_interval_list(other):
            if len(other) == self._emb:
                return self <= CubicalCell(other)
            else:
                raise TypeError('Uncomparable cubic cells because of its '
                                'different embedding numbers.')
        elif isinstance(other, CubicalCell):
            if self._emb == other._emb:
                print(list(map(set, self._intervals)))
                print(list(map(set, other._intervals)))
                return all(set(I1) >= set(I2) for (I1, I2) in
                    zip(self._intervals, other._intervals))

    def __gt__(self, other):
        return self >= other and self != other

    def __hash__(self):
        return hash(self._cube_map)

    @property
    def intervals(self):
        return self._intervals

    def __getitem__(self, i):
        return self._intervals[i]

    @property
    def cube_map(self):
        return self._cube_map

    def degenerate(self, faces, direction):
        non_degenerated_idx = [i for i, I in enumerate(self.intervals) if I[1] > I[0]]
        to_degenerate = [non_degenerated_idx[i] for i in faces]
        new_intervals = []
        for i, I in enumerate(self.intervals):
            if i in to_degenerate:
                new_intervals.append((I[direction],))
            else:
                new_intervals.append(I)
        return CubicalCell(new_intervals)

    def AW(self):
        n = self.dim
        s = list(range(n))
        result = Chain()
        for J in chain.from_iterable(combinations(s, r) for r in range(n + 1)):
            cJ = set(s) - set(J)
            sgn = (-1) ** len([(i, j) for (i, j) in product(J, cJ) if j < i])
            result += sgn * (self.degenerate(cJ, 0) @ self.degenerate(J, 1))
        return result


def dual(cls):
    result = None
    if inspect.isclass(cls):
        if issubclass(cls, AbstractCell):
            class DualCell(cls):
                def __call__(self, other):
                    return int(self == other)
            result = DualCell
        elif type(cls) is type(Chain):
            class Cochain(Chain):
                def __call__(self, other):
                    if self.cell_class == other.cell_class and self.dim == other.dim:
                        basis = tuple(set(self.support()).union(set(other.support())))
                        vec_self = np.array([[self[b] for b in basis]], dtype=np.int32).T
                        vec_other = np.array([[other[b] for b in basis]], dtype=np.int32).T
                        return int((vec_self.T @ vec_other)[0, 0])
                    else:
                        return 0
            result  = Cochain
    return result

CoCubicalCell = dual(CubicalCell)
CoSimplex = dual(Simplex)
CoChain = dual(Chain)


class TensorCell(AbstractCell):
    _factors = ()

    def __init__(self, cell1, cell2):
        self._factors = (cell1, cell2)

    @property
    def factors(self):
        return self._factors

    def __str__(self):
        return ' ⊗ '.join(map(str, self.factors))

    def __getitem__(self, item):
        return self._factors[item]

    @property
    def dim(self):
        return self[0].dim + self[1].dim

    def differential(self):
        sigma, tau = self.factors
        p, q = sigma.dim, tau.dim
        d_sigma = sigma.differential()
        d_tau = tau.differential()
        return (d_sigma @ tau) + (-1) ** p * (sigma @ d_tau)

    def boundary(self):
        for tensor in chain(
                product(self[0].boundary(), (self[1],)),
                product((self[0],), self[1].boundary())):
            yield TensorCell(*tensor)

    def __eq__(self, other):
        return self[0] == other[0] and self[1] == other[1]

    def __lt__(self, other):
        return self[0] < other[0] and self[1] < other[1]

    def __hash__(self):
        return hash((hash(self[0]), hash(self[1])))


if __name__ == '__main__':
    # a = 2 * CubicalCell([[1, 2], [3, 4]]) - 3 * CubicalCell([[3, 4], [1, 2]])
    # b = 2 * CubicalCell([[3, 4], [1, 2]])
    # print(a)
    # print(b)
    # A = CoChain(a)
    # print(A(b))

    # print(Simplex((1, 3, 2)).differential())
    # print(-3 * Simplex((1, 2, 3)) + (-Simplex((1, 5, 6))))
    # sigma, tau = Simplex((1, 2, 3)), Simplex((4, 5))
    # tensor = sigma @ tau
    # print(tensor)
    # print(tensor.differential())
    # cell = CubicalCell([(0, 1), (1,), (2, 3), (3,), (4, 5)])
    # print(cell.AW())

    print((Simplex((2, 5, 6))).AW())


