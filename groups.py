from abc import abstractmethod
import operator


class GroupElement(object):
    _value = None
    _group = None

    def __init__(self, value, group):
        self._value = value
        self._group = group

    @property
    def value(self):
        return self._value

    @property
    def group(self):
        return self._group

    def __add__(self, other):
        if isinstance(other, GroupElement) and other.group == self.group:
            return self.group.op_add(self.value, other.value)
        else:
            raise ValueError('Only instances of GroupElement with the same group can be operated.')

    def __neg__(self):
        return self.group.op_neg(self.value)

    def __sub__(self, other):
        return self + (-other)

    def __str__(self):
        return str(self._value)

    def __eq__(self, other):
        return isinstance(other, GroupElement) and self.value == other.value


class Group(object):
    _op_add = None
    _op_neg = None
    _neutral = None
    _contains = None
    _dimension = None

    def __init__(self, op_add, op_neg, neutral, contains):
        self._op_add = op_add
        self._op_neg = op_neg
        self._neutral = GroupElement(neutral, self)
        self._contains = contains

    def op_add(self, x, y):
        return self(self._op_add(x, y))

    def op_neg(self, x):
        return self(self._op_neg(x))

    @property
    def neutral(self):
        return self(self._neutral)

    def __contains__(self, x):
        if isinstance(x, GroupElement):
            return self._contains(x.value)
        else:
            return self._contains(x)

    @abstractmethod
    def __call__(self, x):
        pass

    @property
    def dimension(self):
        return self._dimension


class Integers(Group):

    def __init__(self):
        super().__init__(
            op_add=lambda x, y: x + y,
            op_neg=lambda x: -x,
            neutral=0,
            contains=lambda x: isinstance(x, int)
        )

    def __call__(self, x):
        if isinstance(x, GroupElement):
            a = x.value
        else:
            a = x

        if isinstance(a, int):
            return GroupElement(a, self)
        else:
            raise ValueError('Not int value cannot be assigned to Integers')

    def __str__(self):
        return 'Z'

    def __pow__(self, power):
        return FiniteGeneratedAbelianGroup(rank=power)

    def __eq__(self, other):
        return isinstance(other, Integers)


class FiniteIntegerGroup(Group):
    _modulo = None

    def __init__(self, modulo):
        self._modulo = modulo

        super().__init__(
            op_add=lambda x, y: (x + y) % modulo,
            op_neg=lambda x:(-x) % modulo,
            neutral=0,
            contains=lambda x: isinstance(x, int)
        )

    def __call__(self, x):
        if isinstance(x, GroupElement):
            a = x.value
        else:
            a = x

        if isinstance(a, int):
            return GroupElement(a % self._modulo, self)
        else:
            raise ValueError('Not int value cannot be assigned to Integers (mod {})'.format(self._modulo))

    def __str__(self):
        return 'Z/{}Z'.format(self._modulo)

    def __eq__(self, other):
        return isinstance(other, FiniteIntegerGroup) and self._modulo == other._modulo


class FiniteGeneratedAbelianGroup(Group):
    _torsion_list = ()
    _rank = 0
    _factors = ()

    def __init__(self, rank, torsion_list=()):
        self._rank = rank
        self._torsion_list = torsion_list
        self._dimension = self._rank + len(self._torsion_list)
        if self._rank > 0:
            self._factors = (Integers(),) * self._rank
        else:
            self._factors = ()

        if self._torsion_list:
            self._factors += tuple(FiniteIntegerGroup(m) for m in self._torsion_list)

        r = self._rank
        t = self._dimension - r

        def op_add(x, y):
            x_free = x[:r]
            y_free = y[:r]
            z_free = tuple(map(operator.add, x_free, y_free))
            x_torsion = x[r:]
            y_torsion = y[r:]
            if t > 0:
                z_torsion = tuple((x_torsion[i] + y_torsion[i]) % m for i, m in enumerate(self._torsion_list))
            else:
                z_torsion =()
            return z_free + z_torsion

        def op_neg(x):
            x_free = x[:r]
            z_free = tuple(map(operator.neg, x_free))
            x_torsion = x[r:]
            if t > 0:
                z_torsion = tuple((-x_torsion[i]) % m for i, m in enumerate(self._torsion_list))
            else:
                z_torsion = ()
            return z_free + z_torsion

        neutral = (0) * self._dimension

        def contains(x):
            return len(x) == self._dimension and all(isinstance(i, int) for i in x)

        super().__init__(
            op_add=op_add,
            op_neg=op_neg,
            neutral=neutral,
            contains=contains
        )

    def __call__(self, x):
        r = self._rank
        t = len(self._torsion_list)

        if isinstance(x, GroupElement):
            a = x.value
        else:
            a = x

        if len(x) == self._dimension and all(isinstance(i, int) for i in x):
            a_free = a[:r]
            if t > 0:
                a_torsion = tuple(a[r + i] % m for i, m in enumerate(self._torsion_list))
            else:
                a_torsion = ()
            return GroupElement(a_free + a_torsion, self)
        else:
            raise ValueError('{} does not belong to {}'.format(x, self))

    def __str__(self):
        if self._rank > 0:
            result_free = 'Z^{}'.format(self._rank)
        else:
            result_free = ''

        if self._torsion_list:
            result_torsion = ' + '.join('Z_{}'.format(m) for m in self._torsion_list)
        else:
            result_torsion = ''

        if result_free:
            if result_torsion:
                return '{} + {}'.format(result_free, result_torsion)
            else:
                return result_free
        else:
            return result_torsion

    def __eq__(self, other):
        return (isinstance(other, FiniteGeneratedAbelianGroup) and self._rank == other._rank and
            self._torsion_list == other._torsion_list)

    @property
    def is_free(self):
        return not self._torsion_list

    @property
    def rank(self):
        return self._rank


class TrivialGroup(Group):

    def __init__(self):
        super().__init__(
            op_add=lambda x, y: 0,
            op_neg=lambda x: 0,
            neutral=0,
            contains=lambda x: x == 0
        )

    def __str__(self):
        return '0'

    @property
    def rank(self):
        return 0

Z = Integers()

if __name__ == '__main__':
    G = FiniteGeneratedAbelianGroup(rank=3, torsion_list=(2, 3))
    print(G)
    x = G((1, 2, 3, 4, 5))
    print(x)
    y = G((5, 4, 3, 2, 1))
    print(x - y)

    Z = Integers()
    ZZ = Integers()
    ZZZ = Z**3
    print(ZZZ((1, 2, 3)) + ZZZ((-1, 2, 4)))

    print(Z == ZZ)