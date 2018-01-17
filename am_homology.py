from sequences import SequenceOf
from groups import FiniteGeneratedAbelianGroup
from chain_map import ChainMap
from rings import Ring
from collections import defaultdict


class HomologyGroups(SequenceOf):
    _generators = {}
    _ranks = {}

    def __init__(self, objects={}, generators={}):
        super().__init__(base_class=FiniteGeneratedAbelianGroup, objects=objects)
        self._generators = generators
        self._ranks = {q: O.rank for q, O in objects.items()}

    @classmethod
    def from_am_model(cls, am_model):
        H = am_model.dst
        d = H.d
        free = {q: [cell for cell in H[q] if d(cell) == 0] for q in H.dimensions}
        tor = {q: [] for q in H.dimensions}
        for q in H.dimensions:
            for cell in H[q + 1]:
                delta = d(cell)
                if delta != 0:
                    for face in delta:
                        l = delta[face]
                        tor[q].append((l, face))
                        free[q].remove(face)

        objects = {}
        generators = {}
        for q in free:
            rank = len(free[q])
            torsion_list = tuple(abs(t[0]) for t in tor[q])
            objects[q] = FiniteGeneratedAbelianGroup(rank, torsion_list)
            generators[q] = tuple(free[q]) + tuple(t[1] for t in tor[q])

        return HomologyGroups(objects, generators)

    @property
    def generators(self):
        return self._generators


class CohomologyGroups(SequenceOf):
    _generators = {}
    _ranks = {}

    def __init__(self, objects={}, generators={}):
        super().__init__(base_class=FiniteGeneratedAbelianGroup, objects=objects)
        self._generators = generators
        self._ranks = {q: O.rank for q, O in objects.items()}

    @classmethod
    def from_am_model(cls, am_model):
        H = am_model.dst
        d = ChainMap(H, degree=+1, matrices={(q - 1): mat.T for (q, mat) in H.d._matrices.items()})
        free = {q: [cell for cell in H[q] if d(cell) == 0] for q in H.dimensions}
        tor = {q: [] for q in H.dimensions}
        for q in H.dimensions:
            for cell in H[q - 1]:
                delta = d(cell)
                if delta != 0:
                    for face in delta:
                        l = delta[face]
                        tor[q].append((l, face))
                        free[q].remove(face)

        objects = {}
        generators = {}
        for q in free:
            rank = len(free[q])
            torsion_list = tuple(abs(t[0]) for t in tor[q])
            objects[q] = FiniteGeneratedAbelianGroup(rank, torsion_list)
            generators[q] = tuple(free[q]) + tuple(t[1] for t in tor[q])

        return CohomologyGroups(objects, generators)

    @property
    def generators(self):
        return self._generators


def cohomologyring_from_am_model(am_model):
    G = CohomologyGroups.from_am_model(am_model)

    def contains(x):
        return (isinstance(x, dict) and
                all(x[d] in A for d, A in G.objects.items()))

    def op_add(x, y):
        result = {}
        for d in x:
            if d in y:
                result[d] = G[d].op_add(x[d], y[d])
            else:
                result[d] = x[d]
        for d in y:
            if d not in x:
                result[d] = y[d]
        return result

    def op_neg(x):
        return {d: A.op_neg(x[d]) for d, A in G.objects.items()}

    neutral = {d: A.neutral.value for d, A in G.objects.items()}

    unit = {0: (1,) * G[0].dimension}

    cup = am_model.src.cup_product()

    def op_mul(x, y):
        h_x = G.to_chain(x)
        h_y = G.to_chain(y)

        g = am_model.inclusion

        c_x = {d: g(c) for d, c in h_x.items()}
        c_y = {d: g(c) for d, c in h_y.items()}

        cup_list = defaultdict(list)

        for d1, c1 in c_x.items():
            for d2, c2 in c_y.items():
                cup_list[d1 + d1].append(cup(c1, c2))

        result = {d: sum(l) for d, l in cup_list.items()}
        return G.to_dict(result)

    return Ring(
        op_add=op_add,
        op_neg=op_neg,
        op_mul=op_mul,
        neutral=neutral,
        unit=unit,
        contains=contains
    )



