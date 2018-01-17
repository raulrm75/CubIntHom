from chain_map import linear_extension
from cells import Simplex, CubicalCell


@linear_extension
def AW(cell):
    if isinstance(cell, Simplex):
        return cell.AW()
    elif isinstance(cell, CubicalCell):
        return cell.AW()
    else:
        raise NotImplementedError('Alexander-Whitney diagonal approximation is only defined'
                                  ' for simplices or cubical cells.')


if __name__ == '__main__':
    print(Simplex((0, 1, 2)).AW())
