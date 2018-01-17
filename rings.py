from abc import abstractmethod
import groups
import operator


class RingElement(groups.GroupElement):

    def __mul__(self, other):
        return self._group.op_mul(self.value, other.value)

    @property
    def ring(self):
        return self._group


class Ring(groups.Group):
    _op_mul = None
    _unit = None

    def __init__(self, op_add, op_neg, op_mul, neutral, unit, contains):
        super().__init__(op_add, op_neg, neutral, contains)
        self._op_mul = op_mul
        self._unit = unit

    @property
    def unit(self):
        return self(self._unit)

    def op_mul(self, x, y):
        return self(self._op_mul(x, y))

    @abstractmethod
    def __call__(self, other):
        pass


class Integers(Ring):
    def __init__(self):
        super().__init__(
            op_add=lambda x, y: x + y,
            op_neg=lambda x: -x,
            op_mul=lambda x, y: x * y,
            neutral=0,
            unit=1,
            contains=lambda x: isinstance(x, int)
        )

    def __call__(self, x):
        if isinstance(x, RingElement):
            a = x.value
        else:
            a = x

        if isinstance(a, int):
            return RingElement(a, self)
        else:
            raise ValueError('Not int value cannot be assigned to Integers')

    def __str__(self):
        return 'Z'

    def __pow__(self, power):
        return FiniteGeneratedAbelianRing(rank=power)

    def __eq__(self, other):
        return isinstance(other, Integers)


class FiniteIntegerRing(Ring):
    _modulo = None

    def __init__(self, modulo):
        self._modulo = modulo

        super().__init__(
            op_add=lambda x, y: (x + y) % modulo,
            op_neg=lambda x:(-x) % modulo,
            op_mul=lambda x, y: (x * y) % modulo,
            neutral=0,
            unit=1,
            contains=lambda x: isinstance(x, int)
        )

    def __call__(self, x):
        if isinstance(x, RingElement):
            a = x.value
        else:
            a = x

        if isinstance(a, int):
            return RingElement(a % self._modulo, self)
        else:
            raise ValueError('Not int value cannot be assigned to Integers (mod {})'.format(self._modulo))

    def __str__(self):
        return 'Z/{}Z'.format(self._modulo)

    def __eq__(self, other):
        return isinstance(other, FiniteIntegerRing) and self._modulo == other._modulo


class FiniteGeneratedAbelianRing(Ring):
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

        def op_mul(x, y):
            x_free = x[:r]
            y_free = y[:r]
            z_free = tuple(map(operator.mul, x_free, y_free))
            x_torsion = x[r:]
            y_torsion = y[r:]
            if t > 0:
                z_torsion = tuple((x_torsion[i] * y_torsion[i]) % m for i, m in enumerate(self._torsion_list))
            else:
                z_torsion =()
            return z_free + z_torsion

        neutral = (0) * self._dimension

        unit = (1) * self._dimension

        def contains(x):
            return len(x) == self._dimension and all(isinstance(i, int) for i in x)

        super().__init__(
            op_add=op_add,
            op_neg=op_neg,
            op_mul=op_mul,
            neutral=neutral,
            unit=unit,
            contains=contains
        )

    def __call__(self, x):
        r = self._rank
        t = len(self._torsion_list)

        if isinstance(x, RingElement):
            a = x.value
        else:
            a = x

        if len(x) == self._dimension and all(isinstance(i, int) for i in x):
            a_free = a[:r]
            if t > 0:
                a_torsion = tuple(a[r + i] % m for i, m in enumerate(self._torsion_list))
            else:
                a_torsion = ()
            return RingElement(a_free + a_torsion, self)
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
        return (isinstance(other, FiniteGeneratedAbelianRing) and self._rank == other._rank and
            self._torsion_list == other._torsion_list)

    @property
    def is_free(self):
        return not self._torsion_list

    @property
    def rank(self):
        return self._rank


class TrivialRing(Ring):

    def __init__(self):
        super().__init__(
            op_add=lambda x, y: 0,
            op_neg=lambda x: 0,
            op_mul=lambda x, y: 0,
            neutral=0,
            unit=0,
            contains=lambda x: x == 0
        )

    def __str__(self):
        return '0'

    @property
    def rank(self):
        return 0

Z = Integers()
