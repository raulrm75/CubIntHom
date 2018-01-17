from groups import Group, TrivialGroup


class SequenceOf(object):
    _objects = {}
    _base_class = None
    _dimensions = ()
    _dimension = -1

    def __init__(self, base_class, objects={}):
        assert all(isinstance(O, base_class) for O in objects.values())
        assert all(i >= 0 for i in objects.keys())
        self._objects = objects
        self._base_class = base_class
        self._dimensions = tuple(sorted(self._objects.keys()))
        if self._dimensions:
            self._dimension = max(self._dimensions)
        else:
            self._dimension = -1

    @property
    def dimensions(self):
        return self._dimensions

    def __getitem__(self, item):
        if item in self._dimensions:
            return self._objects[item]
        else:
            return TrivialGroup()

    def __iter__(self):
        for i in self._dimensions:
            yield self._objects[i]


    def __str__(self):
        objects = (str(self[i]) for i in range(self._dimension + 1))
        return ', '.join(objects)

    @property
    def objects(self):
        return self._objects


if __name__ == '__main__':
    from groups import Integers, FiniteIntegerGroup
    Z = Integers()
    Z2 = FiniteIntegerGroup(2)
    S = SequenceOf(Group, {0: Z**3, 1: Z**2, 3:Z2})
    print(S[2])

