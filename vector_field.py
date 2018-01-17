from chain_map import ChainMap
from cells import AbstractCell, Chain
from chain_complexes import ChainComplex
import numpy as np
from reduction import Reduction
import networkx as nx
from time import time


class VectorField(ChainMap):

    def __init__(self, chain_complex):
        super().__init__(chain_complex, chain_complex, degree=+1)

    def set_image(self, src, dst):
        assert isinstance(src, AbstractCell) and isinstance(dst, AbstractCell)
        super().set_image(
            src,
            self.src.d(dst)[src] * dst
        )

    def is_source(self, cell):
        q = cell.dim
        return np.any(self[q][:, self.src.index(cell)])

    def is_target(self, cell):
        q = cell.dim - 1
        return self.src[q] and np.any(self[q][self.src.index(cell), :])

    def is_critical(self, cell):
        return not (self.is_source(cell) or self.is_target(cell))

    # def available_facets(self, cell):
    #     if self.is_critical(cell):
    #         return (facet for facet in self.src[cell.dim + 1] if
    #                 self.is_critical(facet) and
    #                 abs(self.src.d(facet)[cell]) == 1)
    #     else:
    #         return ()

    def am_model(self, do_SNF=True, verbose=False):
        st = time()
        pt = time()
        reduction = self.reduction()
        if verbose:
            print('Vector field reduction calculated in {:.3f}s'.format(time() - pt))

        pt = time()
        V = create_vector_field(reduction.dst)
        if verbose:
            print('Reduced vector field built in {:.3f}s'.format(time() - pt))

        while V != 0:
            pt = time()
            reduction = V.reduction() * reduction
            if verbose:
                print('Vector field reduction calculated in {:.3f}s'.format(time() - pt))
            pt  =time()
            V = create_vector_field(reduction.dst)
            if verbose:
                print('Reduced vector field built in {:.3f}s'.format(time() - pt))

        if do_SNF:
            # Compute SNF if necessary
            if reduction.dst.d != 0:
                pt = time()
                snf_reduction = reduction.dst.SNF_reduction()
                if verbose:
                    print('SNF reduction calculated in {:.3f}s'.format(time() - pt))
                pt = time()
                reduction = snf_reduction * reduction
                if verbose:
                    print('Final reduction built in {:.3f}s'.format(time() - pt))

        # Make all differentials have positive coefficient
        pt = time()
        f_sign = ChainMap(reduction.dst)
        h_sign = ChainMap(reduction.dst, degree=1)
        for sigma in reduction.dst:
            co_d = reduction.dst.d.T
            c = co_d(sigma)
            if c:
                tau = next(iter(c))
                if c[tau] < 0:
                    f_sign.set_image(sigma, -sigma)
                else:
                    f_sign.set_image(sigma, sigma)
            else:
                f_sign.set_image(sigma, sigma)

        new_dst = ChainComplex(cell_class=reduction.dst.cell_class)
        for p in reduction.dst.dimensions:
            new_dst[p] = reduction.dst[p]
        new_d = ChainMap(new_dst, degree=-1)
        for p, M in reduction.dst.d.matrices.items():
            new_d[p] = np.abs(M)
        new_dst.d = new_d
        reduction = Reduction(reduction.dst, new_dst, f_sign, f_sign, h_sign) * reduction
        if verbose:
            print('Signs adjusted in {:.3f}s'.format(time() - pt))
            print('Full time {:.3f}s'.format(time() - pt))
        return reduction

    # def source_cells(self, p):
    #     return tuple(sigma for sigma in self.src[p] if self.is_source(sigma))

    # def target_cells(self, p):
    #     return tuple(sigma for sigma in self.src[p] if self.is_target(sigma))

    # def critical_cells(self, p):
    #     return tuple(sigma for sigma in self.src[p] if self.is_critical(sigma))

    def decomposition(self):
        source_complex = ChainComplex(cell_class=self.src.cell_class)
        target_complex = ChainComplex(cell_class=self.src.cell_class)
        critical_complex = ChainComplex(cell_class=self.src.cell_class)

        for p, M in self.src.modules.items():
            source_complex[p] = tuple(M[i] for i in np.argwhere(self[p].sum(0) != 0)[:, 0])
            N = self.src[p + 1]
            target_complex[p + 1] = tuple(N[i] for i in np.argwhere(self[p].sum(1) != 0)[:, 0])
            critical_complex[p] = tuple(
                M[i] for i in set(np.argwhere(self[p].sum(0) == 0)[:, 0]).intersection(
                set(np.argwhere(self[p - 1].sum(1) == 0)[:, 0]))
            )
        return {'t': target_complex, 's': source_complex, 'c': critical_complex}

    def _projections_inclusions(self, decomposition):
        cplx = decomposition
        target_projection = ChainMap(self.src, cplx['t'])
        source_projection = ChainMap(self.src, cplx['s'])
        critical_projection = ChainMap(self.src, cplx['c'])
        target_inclusion = ChainMap(cplx['t'], self.src)
        source_inclusion = ChainMap(cplx['s'], self.src)
        critical_inclusion = ChainMap(cplx['c'], self.src)
        for sigma in self.src:
            if self.is_source(sigma):
                source_projection.set_image(sigma, sigma)
                source_inclusion.set_image(sigma, sigma)
            elif self.is_target(sigma):
                target_projection.set_image(sigma, sigma)
                target_inclusion.set_image(sigma, sigma)
            else:
                critical_projection.set_image(sigma, sigma)
                critical_inclusion.set_image(sigma, sigma)
        return (target_projection, source_projection, critical_projection,
                target_inclusion, source_inclusion, critical_inclusion)

    def reduction(self):
        cplx = self.decomposition()

        d = self.src.d
        pi_t, pi_s, pi_c, iota_t, iota_s, iota_c = self._projections_inclusions(cplx)
        d_33 = pi_c * d * iota_c
        d_31 = pi_c * d * iota_t
        d_21 = pi_s * d * iota_t

        d_21_inv = ChainMap(d_21.dst, d_21.src, degree=+1)
        for p in self.dimensions:
            d_21_inv[p] = np.linalg.inv(d_21[p + 1]).astype(np.int32)
        d_23 = pi_s * d * iota_c

        cplx['c'].d = d_33 - d_31 * d_21_inv * d_23

        f = pi_c - d_31 * d_21_inv * pi_s
        g = iota_c - iota_t * d_21_inv * d_23

        h = iota_t * d_21_inv * pi_s

        return Reduction(self.src, cplx['c'], f, g, h)


def create_vector_field(C):
    V = VectorField(C)
    d = C.d
    D = d.T
    G = nx.DiGraph()

    for src in C:
        G.add_node(src)
        if V.is_critical(src):
            facets = D(src)
            for tgt in facets:
                coef = d(tgt)[src]
                if not V.is_source(tgt) and not V.is_target(tgt) and abs(coef) == 1:
                    G.add_node(tgt)
                    G.add_edge(src, tgt)
                    for dst in d(tgt):
                        if dst != src:
                            if not G.has_node(dst):
                                G.add_node(dst)
                            G.add_edge(tgt, dst)
                    if any(nx.simple_cycles(G)):
                        G.remove_node(tgt)
                    else:
                        V.set_image(src, tgt)
                        break
    return V


if __name__ == '__main__':
    from cell_complexes import CellComplex
    from cells import Simplex
    from time import time

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
    t0 = time()
    V = create_vector_field(C)
    print('Vector field created in {:.3f}s'.format(time() - t0))

    am = V.am_model()
    for cell in am.dst.cells:
        print('d_M({}) = {}'.format(cell, am.dst.d(cell)))