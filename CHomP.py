import numpy as np
from itemproperties import itemproperty
from random import randint
from copy import copy


class Chain(object):
    _len = 0
    _data = []

    def __init__(self, arg=None):
        if isinstance(arg, np.ndarray) and arg.ndim == 1:
            self._data = [(n, arg[n]) for n in np.where(arg)[0]]
            # self._data.sort(key=lambda x: x[0])
            self._len = len(self._data)
        elif isinstance(arg, Chain):
            self._len = arg._len
            self._data = arg._data.copy()
        else:
            self._len = 0
            self._data = []

    def copy(self):
        return copy(self)

    def __str__(self):
        result = ' + '.join('·'.join(
            (str(e) if e > 0 else '({})'.format(e),
            '<{}>'.format(n))) for (n, e) in self._data)
        if result:
            return result
        else:
            return '0'

    def __getitem__(self, n):
        i = 0
        while i < self._len and self.n[i] < n:
            i += 1
        if i < self._len and self.n[i] == n:
            return self.e[i]
        else:
            return 0

    def __setitem__(self, n, e):
        i = 0
        while i < self._len and self.n[i] < n:
            i += 1

        if i < self._len and self.n[i] == n:
            if e == 0:
                self.remove(n)
            else:
                self.e[i] = e
        else:
            self.insertpair(i, n, e)

    def __delitem__(self, n):
        self.remove(n)

    def keys(self):
        for i in range(self._len):
            yield self.n[i]

    def values(self):
        for i in range(self._len):
            yield self.e[i]

    def __contains__(self, n):
        i = 0
        while i < self._len and self.n[i] < n:
            i += 1

        if i < self._len and self.n[i] == n:
            return True
        else:
            return False

    def __len__(self):
        return self._len

    def __iter__(self):
        for i in range(self._len):
            yield self.n[i]

    def __bool__(self):
        return self._len > 0

    def __eq__(self, other):
        return self._data == other._data

    def toarray(self, size=None):
        if size is None:
            size = max(self.keys()) + 1
        result = np.zeros(size, dtype=int)
        for i in range(self._len):
            result[self.n[i]] = self.e[i]
        return result

    @itemproperty
    def n(self, index):
        return self._data[index][0]

    @n.setter
    def n(self, index, value):
        self._data[index] = (value, self._data[index][1])

    @itemproperty
    def e(self, index):
        return self._data[index][1]

    @e.setter
    def e(self, index, value):
        self._data[index] = (self._data[index][0], value)

    def size(self):
        return self._len

    def empty(self):
        return not self._len

    def getcoefficient(self, n=-1):
        if n < 0:
            if self:
                return self[next(self.keys())]
            else:
                return 0
        else:
            return self[n]

    def findnumber(self, n):
        if n in self:
            keys = tuple(self.keys())
            return keys.index(n)
        else:
            return -1

    def coef(self, i):
        return self.e[i]

    def num(self, i):
        return self.n[i]

    def contains_non_invertible(self):
        return any(abs(e) > 0 for (n, e) in self._data)

    def findbest(self, table=None):
        if self._len <= 1:
            return self._len - 1

        best_delta = -1
        best_i = 0
        for i in range(self._len):
            this_delta = abs(self.e[i])
            if not table is None and this_delta == 1:
                return i
            if not i or this_delta < best_delta:
                best_delta = this_delta
                best_i = i

        if table is None:
            return best_i

        best_length = table[self.n[best_i]].size()
        for i in range(best_i + 1, self._len):
            if abs(self.e[i]) == best_delta:
                this_length = table[self.n[i]].size()
                if best_length > this_length:
                    best_length = this_length
                    best_i = i
        return best_i

    def add_(self, n, e):
        if e != 0:
            if n in self:
                self[n] += e
                if self[n] == 0:
                    del self[n]
            else:
                self[n] = e

    def remove(self, n):
        i = 0
        while i < self._len and self.n[i] < n:
            i += 1

        if i < self._len and self.n[i] == n:
            self.removepair(i)

    def add(self, other, e, number=-1, table=None):
        if e != 0 and other._len:
            tablen = self._len + other._len
            bigntab = [0] * tablen
            bigetab = [0] * tablen

            i = 0
            j = 0
            k = 0
            while i < self._len or j < other._len:
                if i >= self._len:
                    bigntab[k] = other.n[j]
                    bigetab[k] = e * other.e[j]
                    j += 1
                    if not table is None:
                        table[bigntab[k]].add(number, bigetab[k])
                    k += 1
                elif j >= other._len or self.n[i] < other.n[j]:
                    bigntab[k] = other.n[i]
                    bigetab[k] = e * other.e[i]
                    i += 1
                elif self.n[i] > other.n[j]:
                    bigntab[k] = other.n[j]
                    bigetab[k] = e * other.e[j]
                    j += 1
                    if not table is None:
                        table[bigntab[k]].add(number, bigetab[k])
                    k += 1
                else:
                    bigntab[k] = self.n[i]
                    addelem = e * other.e[j]
                    j += 1
                    bigetab[k] = self.e[i] + addelem
                    i += 1
                    if bigetab[k] != 0:
                        if not table is None:
                            table[bigntab[k]].add(number, addelem)
                        k += 1
                    elif table:
                        table[bigntab[k]].remove(number)

        self._data = list(zip(bigntab, bigetab))
        self._len = len(self._data)

    def swap(self, other, number=-1, other_number=-1, table=None):
        self._data, other._data = other._data, self._data
        self._len = len(self._data)
        other._len = len(other._data)
        if not table is None:
            table[number], table[other_number] = (table[other_number],
                table[number])

    def take(self, c):
        self._data, self._len = c._data, c._len
        del c

    def multiply(self, e, number=-1):
        if number >= 0:
            for i in range(self._len):
                if self.n[i] == number:
                    if e == 0:
                        self.removepair(i)
                    else:
                        self.e[i] *= e
        elif e != 0:
            for i in range(self._len):
                self.e[i] *= e
                if self.e[i] == 0:
                    self.removepair(i)
        else:
            self._len = 0
            self._data = []

    def insertpair(self, i, n, e):
        new_data = []
        for j in range(self._len + 1):
            if j < i:
                new_data.append(self._data[j])
            elif j == i:
                new_data.append((n, e))
            else:
                new_data.append(self._data[j - 1])
        self._data = new_data
        self._len += 1

    def removepair(self, i):
        del self._data[i]
        self._len -= 1

    def swapnumbers(self, number1, number2):
        if number1 != number2:
            if number1 > number2:
                number1, number2 = number2, number1
            i1 = 0
            i2 = 0
            while i1 < self._len and self.n[i1] < number1:
                i1 += 1

            while i2 < self._len and self.n[i2] < number2:
                i2 += 1

            if i1 < self._len and self.n[i1] == number1:
                if i2 < self._len and self.n[i2] == number2:
                    self.e[i1], self.e[i2] = self.e[i2], self.e[i1]
                else:
                    temp = self.e[i1]
                    for i in range(i1 + 1, i2):
                        self.n[i - 1] = self.n[i]
                        self.e[i - 1] = self.e[i]
                    self.n[i2 - 1] = number2
                    self.e[i2 - 1] = temp
            elif i2 < self._len and self.n[i2] == number2:
                temp = self.e[i2]
                for i in range(i2, i1, -1):
                    self.n[i] = self.n[i - 1]
                    self.e[i] = self.e[i - 1]
                self.n[i1] = number1
                self.e[i1] = temp


class MMatrix(object):
    _data = np.zeros((0, 0), int)
    dom_dom = []
    dom_img = []
    img_dom = []
    img_img = []

    def __init__(self, source=None):
        self._data = np.zeros((0, 0), int)
        self.dom_dom = []
        self.dom_img = []
        self.img_dom = []
        self.img_img = []

        if isinstance(source, MMatrix):
            # Copy constructor
            self._data = source._data.copy()
        elif isinstance(source, np.ndarray):
            # Construtor from array
            self._data = source.copy()
        else:
            # Default constructor
            pass

    def __str__(self):
        return str(self._data)

    def define(self, numrows, numcols):
        if self.nrows > numrows or self.ncols > numcols:
            raise Exception(
                "Trying to define a matrix smaller than it really is")
        else:
            self.increase(numrows, numcols)
            self.nrows = numrows
            self.ncols = numcols

    def identity(self, size):
        if not self.nrows and not self.ncols:
            self._data = np.zeros((size, size), dtype=int)
        elif size > self.nrows or size > self.ncols:
            size = min(self.nrows, self.ncols)
        self._data[:size, :size] = np.eye(size, dtype=int)

    def add_(self, row, col, e):
        # No size checking. In Pawel's original code, this method allows
        # automatic growing.
        self._data[row, col] = e

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __eq__(self, other):
        if isinstance(other, MMatrix):
            return np.all(self._data == other._data)
        else:
            return np.all(self._data == other)

    def get_(self, row, col):
        return self._data[row, col]

    def getrow(self, n):
        return self.rows[n]

    def getcol(self, n):
        return self.cols[n]

    def getnrows(self):
        return self.nrows

    def getncols(self):
        return self.ncols

    def addrow(self, dest, source, e):
        self._data[dest] += self._data[source] * e
        for m in self.img_img:
            #m.addrow(dest, source, e)
            m._data[dest] += m._data[source] * e
        for m in self.img_dom:
            #m.addcol(source, dest, -e)
            m._data[:, source] += m._data[:, dest] * (-e)

    def addcol(self, dest, source, e):
        self._data[:, dest] += self._data[:, source] * e
        for m in self.dom_dom:
            #m.addcol(dest, source, e)
            m._data[:, dest] += m._data[:, source] * e
        for m in self.dom_img:
            #m.addrow(source, dest, -e)
            m._data[source] += m._data[dest] * (-e)

    def swaprows(self, i, j):
        if i != j:
            self._data[[i, j], :] = self._data[[j, i], :]

            for m in self.img_img:
                #m.swaprows(i, j)
                m._data[[i, j], :] = m._data[[j, i], :]
            for m in self.img_dom:
                #m.swapcols(i, j)
                m._data[:, [i, j]] = m._data[:, [j, i]]

    def swapcols(self, i, j):
        if i != j:
            self._data[:, [i, j]] = self._data[:, [j, i]]

            for m in self.dom_dom:
                #m.swapcols(i, j)
                m._data[:, [i, j]] = m._data[:, [j, i]]
            for m in self.dom_img:
                #m.swaprows(i, j)
                m._data[[i, j], :] = m._data[[j, i], :]

    def multiplyrow(self, n, e):
        self.rows[n] *= e

    def multiplycol(self, n, e):
        self.cols[n] *= e

    def findrow(self, req_elements=1, start=-1):
        return self.findrowcol(req_elements, start, 1)

    def findcol(self, req_elements=1, start=-1):
        return self.findrowcol(req_elements, start, 0)

    def reducerow(self, n, preferred):
        the_other = -1
        len_ = self.rows[n].size()
        while len_ > 1:
            local = self.rows[n]
            best_i = local.findbest([self.cols[r] for r in range(self.ncols)])
            preferred_i = -1 if preferred < 0 else local.findnumber(preferred)
            if (preferred_i >= 0 and
                abs(local.coef(preferred_i)) == abs(local.coef(best_i))):
                best_i = preferred_i
            the_other = local.num(best_i)
            for i in range(len_):
                if i == best_i:
                    continue
                quotient = local.coef(i) // local.coef(best_i)
                self.addcol(local.num(i), local.num(best_i), -quotient)
            len_ = self.rows[n].size()
        return the_other

    def reducecol(self, n, preferred):
        the_other = -1
        len_ = self.cols[n].size()
        while len_ > 1:
            local = self.cols[n].copy()
            best_i = local.findbest([self.rows[r] for r in range(self.nrows)])
            preferred_i = -1 if preferred < 0 else local.findnumber(preferred)
            if (preferred_i >= 0 and
                abs(local.coef(preferred_i)) == abs(local.coef(best_i))):
                best_i = preferred_i
            the_other = local.num(best_i)
            for i in range(len_):
                if i == best_i:
                    continue
                quotient = local.coef(i) // local.coef(best_i)
                self.addrow(local.num(i), local.num(best_i), -quotient)
            len_ = self.cols[n].size()
        return the_other

    def simple_reductions(self, quiet=True):
        if self.nrows and self.ncols:
            countreduced = 0
            candidates = [r for r in range(self.nrows) if
                self.rows[r].size() == 1 and
                abs(self.rows[r].e[0]) == 1 and
                self.cols[self.rows[r].n[0]].size() > 1]
            while candidates:
                countreduced += 1
                self.reducecol(self.rows[candidates[0]].num(0), -1)

                candidates = [r for r in range(self.nrows) if
                    self.rows[r].size() == 1 and
                    abs(self.rows[r].e[0]) == 1 and
                    self.cols[self.rows[r].n[0]].size() > 1]

            #countreduced = 0
            #count = 4 * min(self.ncols, self.nrows)
            #nr = randint(0, self.nrows - 1)
            #nr_count = 0
            #nr_add = 0

            #while count:
                #if (self.rows[nr].size() == 1 and
                    #abs(self.rows[nr].e[0]) == 1 and
                    #self.cols[self.rows[nr].n[0]].size() > 1):
                    ##print('Reduce col {}'.format(nr))
                    #countreduced += 1
                    #self.reducecol(self.rows[nr].num(0), -1)
                #if nr_count:
                    #nr_count -= 1
                #else:
                    #nr_add = ((randint(0, 32767) >> 2) + 171) % self.nrows
                    #if nr_add < 1:
                        #nr_add = 1
                    #nr_count = 100
                #nr += nr_add
                #if nr >= self.nrows:
                    #nr -= self.nrows
                #if not quiet and not (count % 373):
                    #print(count)

                #count -= 1
            #if not quiet:
                #print(countreduced)

    def simple_form(self, quiet=True):
        if self.nrows and self.ncols:
            self.simple_reductions(quiet)
            count = 0
            row = -1
            col = self.findcol(2)
            prev_row = -1
            prev_col = -1
            if col < 0:
                row = self.findrow(2)

            while row >= 0 or col >= 0:
                while row >= 0 or col >= 0:
                    if row >= 0:
                        col = self.reducerow(row, prev_col)
                        prev_row = row
                        row = -1
                    elif col >= 0:
                        row = self.reducecol(col, prev_row)
                        prev_col = col
                        col = -1
                count += 1
                if not quiet and not count % 373:
                    print('{:<12}{}'.format(count, '\b' * 12))
                col = self.findcol(2)
                if col < 0:
                    row = self.findrow(2)

            if not quiet:
                print('{} reductions made'.format(count))

    def arrange_towards_SNF(self, invertible_count=[]):
        cur = 0
        for n in range(self.ncols):
            if self.cols[n].empty():
                continue
            if abs(self.cols[n].coef(0)) != 1:
                continue
            r = self.cols[n].num(0)
            if n != cur:
                self.swapcols(n, cur)
            if r != cur:
                self.swaprows(r, cur)
            cur += 1
        if invertible_count:
            invertible_count[0] = cur

        for n in range(cur, self.ncols):
            if self.cols[n].empty():
                continue
            r = self.cols[n].num(0)
            if n != cur:
                self.swapcols(n, cur)
            if r != cur:
                self.swaprows(r, cur)
            cur += 1
        return cur

    def is_diagonal(self):
        D = np.zeros_like(self._data)
        d = np.diagonal(self._data)
        L = list(range(d.shape[0]))
        D[L, L] = d
        return np.all(self._data == D)

    def is_in_SNF(self):
        if self.is_diagonal():
            d = np.diagonal(self._data)
            result = True
            for i in range(1, d.shape[0]):
                if d[i - 1]:
                    result = result and d[i] % d[i - 1] == 0
                    if not result:
                        break
            return result
        else:
            return False

    def simple_form_to_SNF(self, quiet=True):
        if not quiet:
            print('Determining the diagonal...')
        L = [0]
        indexEnd = self.arrange_towards_SNF(L)
        indexBegin = L[0]
        if not quiet:
            print('{} invertible and {} non invertible coefficients.'.format(
                indexBegin, indexEnd))
        if not quiet:
            print('Correcting the division condition...')
        countCorrections = 0
        divisionOk = False
        while not divisionOk:
            divisionOk = True
            for index in range(indexBegin + 1, indexEnd):
                e1 = self.get(index - 1, index - 1)
                e2 = self.get(index, index)
                if e2 % e1 == 0:
                    continue
                divisionOk = False
                self.division_SNF_correction(e1, index - 1, e2, index)
                countCorrections += 1

    def invert(self):
        pass

    def multiply(self, m1, m2):
        pass

    def submatrix(self, matr, domain, range_):
        pass

    @property
    def nrows(self):
        return self._data.shape[0]

    @nrows.setter
    def nrows(self, numrows):
        self.increaserows(numrows)

    @property
    def ncols(self):
        return self._data.shape[1]

    @ncols.setter
    def ncols(self, numcols):
        self.increasecols(numcols)

    @itemproperty
    def rows(self, n):
        return Chain(self._data[n])

    @rows.setter
    def rows(self, n, value):
        if isinstance(value, np.ndarray) and value.ndim == 1:
            self._data[n] = value
        elif isinstance(value, Chain):
            self._data[n][list(map(lambda x: x[0], value._data))] = list(map(
                lambda x: x[1], value._data))

    @itemproperty
    def cols(self, n):
        return Chain(self._data[:, n])

    @cols.setter
    def cols(self, n, value):
        if isinstance(value, np.ndarray) and value.ndim == 1:
            self._data[:, n] = value
        elif isinstance(value, Chain):
            self._data[:, n][list(map(lambda x: x[0], value._data))] = list(map(
                lambda x: x[1], value._data))

    def findrowcol(self, req_elements, start, which):
        i = start
        random_i = -1
        loopcounter = 0
        size = self.nrows if which else self.ncols
        if start < 0:
            random_i = randint(0, size - 1)
            i = random_i
            loopcounter = 1
        candidate = -1
        candidates_left = 3
        if not (self.ncols if which else self.nrows):
            if req_elements > 0 or i >= size:
                return -1
            else:
                return i

        while i < size:
            l = self.rows[i].size() if which else self.cols[i].size()
            if req_elements >= 0 and l >= req_elements:
                return i
            elif req_elements < 0 and not l:
                if which:
                    chain = self.rows[i]
                else:
                    chain = self.cols[i]
                if not candidates_left or chain.contains_non_invertible():
                    return i
                else:
                    candidate = i
                    candidates_left -= 1
                    if start < 0:
                        random_i = randint(0, size - 1)
                        i = random_i - 1
                        loopcounter = 1
            i += 1
            if i >= size:
                loopcounter -= 1
                if loopcounter:
                    i = 0
            if random_i >= 0 and not loopcounter and i >= random_i:
                break

        return candidate

    def increase(self, numrows, numcols):
        self.increaserows(numrows)
        self.increasecols(numcols)

    def increaserows(self, numrows):
        if numrows > self.nrows:
            new_data = np.zeros((numrows, self.ncols), int)
            new_data[:self.nrows, :] = self._data
            self._data = new_data

    def increasecols(self, numcols):
        if numcols > self.ncols:
            new_data = np.zeros((self.nrows, numcols), int)
            new_data[:, :self.ncols] = self._data
            self._data = new_data

    def division_SNF_correction(self, a, pos1, b, pos2):
        sigma, tau = MMatrix.extendedGCD(a, b)
        beta = sigma * a + tau * b
        alpha = a // beta
        gamma = b // beta
        self.addcol(pos1, pos2, 1)
        setRowsCols = [pos1, pos2]
        M = MMatrix(np.array([
            [sigma, tau],
            [-gamma, alpha]]))
        invM = MMatrix(np.array([
            [alpha, -tau],
            [gamma, sigma]]))

        self.mult_left(setRowsCols, setRowsCols, M, invM, True)
        self.addcol(pos2, pos1, -tau * gamma)

    @staticmethod
    def extendedGCD(a, b):
        aa = a
        bb = b
        xx = 0
        yy = 1
        lastx = 1
        lasty = 0

        while bb != 0:
            quotient = aa // bb
            remainder = aa % bb
            aa = bb
            bb = remainder
            xxx = lastx - quotient * xx
            lastx = xx
            xx = xxx
            yyy = lasty - quotient * yy
            lasty = yy
            yy = yyy

        return lastx, lasty

    def mult_left(self, setRows, setCols, M, invM, update_linked):
        size = M.getnrows()
        newRows = {}
        for row in range(size):
            affected = Chain()
            for col in range(size):
                rowChain = self.getrow(setCols[col])
                rowSize = rowChain.size()
                for cur in range(rowSize):
                    num = rowChain.num(cur)
                    if affected.findnumber(num) < 0:
                        affected.add_(num, 1)

            col_count = affected.size()
            for col in range(col_count):
                col_nr = affected.num(col)
                e = 0
                for k in range(size):
                    row_k = setCols[k]
                    e += M.get(row, k) * self.get(row_k, col_nr)
                if e != 0:
                    if not row in newRows:
                        newRows[row] = Chain()
                    newRows[row].add_(col_nr, e)

        for row in range(size):
            row_nr = newRows[row]
            row_ch = Chain(row_nr)
            row_prev = self.rows[row_nr]
            len_prev = row_prev.size()
            for i in range(len_prev):
                col_nr = row_prev.num(i)
                e = row_ch.getcoefficient(col_nr)
                if e == 0:
                    coef = self.get(row_nr, col_nr)
                    self.add_(row_nr, col_nr, -coef)
            len_ = row_ch.size()
            for i in range(len_):
                col_nr = row_ch.num(i)
                e = row_ch.coef(i) - self.get(row_nr, col_nr)
                if e != 0:
                    self.add_(row_nr, col_nr, e)

        if update_linked:
            for m in self.img_img:
                if m.nrows:
                    m.mult_left(setRows, setCols, M, invM, False)
            for m in self.img_dom:
                if m.ncols:
                    m.mult_right(setCols, setRows, invM, M, False)

    def mult_right(self, setRows, setCols, M, invM, update_linked):
        size = M.getncols()
        newCols = {}

        for col in range(size):
            affected = Chain()
            for row in range(size):
                colChain = self.getcol(setRows[row])
                colSize = colChain.size()
                for cur in range(colSize):
                    num = colChain.num(cur)
                    if affected.findnumber(num) < 0:
                        affected.add_(num, 1)
            row_count = affected.size()
            for row in range(row_count):
                row_nr = affected.num(row)
                e = 0
                for k in range(size):
                    col_k = setRows[k]
                    e += self.get(row_nr, col_k) * M.get(k, col)
                if e != 0:
                    if not col in newCols:
                        newCols[col] = Chain()
                    newCols[col].add_(row_nr, e)

        for col in range(size):
            col_nr = setCols[col]
            col_ch = newCols[col]

            col_prev = self.getcol(col_nr)
            len_prev = col_prev.size()
            for i in range(len_prev):
                row_nr = col_prev.num(i)
                e = col_ch.getcoefficient(row_nr)
                if e == 0:
                    coef = self.get(row_nr, col_nr)
                    self.add_(row_nr, col_nr, -coef)
            len_ = col_ch.size()
            for i in range(len_):
                row_nr = col_ch.num(i)
                e = col_ch.coef(i) - self.get(row_nr, col_nr)
                if e != 0:
                    self.add_(row_nr, col_nr, e)

        if update_linked:
            for m in self.dom_dom:
                if m.ncols:
                    m.mult_left(setRows, setCols, M, invM, False)
            for m in self.dom_img:
                if m.nrows:
                    m.mult_right(setCols, setRows, invM, M, False)

    def dot(self, other):
        if isinstance(other, MMatrix):
            return MMatrix(self._data.dot(other._data))
        else:
            return MMatrix(self._data.dot(other))

    @property
    def shape(self):
        return self._data.shape


class MatrixSNF(object):

    matrices = {}
    chgBasisA = {}
    chgBasisG = {}
    computedSNF = False

    def __init__(self):
        self.matrices = {}
        self.chgBasisA = {}
        self.chgBasisG = {}
        self.computedSNF = False

    #def setSize(self, q, numRows, numColumns):
        #if not q in self.matrices:
            #self.matrices[q] = MMatrix()

        #self.matrices[q].define(numRows, numColumns)

    def add_(self, q, row, column, element):
        self.matrices[q].add_(row, column, element)

    def getNumRows(self, q):
        return self.matrices[q].getnrows()

    def getNumCols(self, q):
        return self.matrices[q].getncols()

    def computeSNF(self, numMatr=None):
        if numMatr is None:
            numMatr = len(self.matrices)
        if not self.computedSNF:
            for q in range(numMatr):
                nCols = self.matrices[q].ncols
                self.chgBasisA[q] = MMatrix()
                self.chgBasisA[q].identity(nCols)
                self.chgBasisG[q] = MMatrix()
                self.chgBasisG[q].identity(nCols)
                self.matrices[q].dom_img.append(self.chgBasisA[q])
                self.matrices[q].dom_dom.append(self.chgBasisG[q])
                if q > 0:
                    self.matrices[q].img_img.append(self.chgBasisA[q - 1])
                    self.matrices[q].img_dom.append(self.chgBasisG[q - 1])
                if q < numMatr - 1:
                    self.matrices[q].dom_img.append(self.matrices[q + 1])
                if q > 0:
                    self.matrices[q].img_dom.append(self.matrices[q - 1])
            for q in range(numMatr - 1, -1, -1):
                if self.matrices[q].getnrows() == 0:
                    continue
                if self.matrices[q].getncols() == 0:
                    continue
                self.matrices[q].simple_form()
                self.matrices[q].simple_form_to_SNF()

            self.computedSNF = True

    def getColCoef(self, q, n):
        pass

    def getColCell(self, q, n):
        pass

    def getRowCoef(self, q, n):
        pass

    def getRowCell(self, q, n):
        pass

    def getNewChain(self, q, n):
        pass

    def getNewChainRow(self, q, n):
        pass

    def getOldChain(self, q, n):
        pass

    def getOldChainRow(self, q, n):
        pass

    @staticmethod
    def abstractChain(ch):
        pass


if __name__ == '__main__':
    a = np.array([1, 0, 0, 0, 1, 0])
    test = Chain(a)
    print(test._data)
    print(list(zip(np.where(a)[0], a[np.where(a)])))