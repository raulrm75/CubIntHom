import numpy as np


""""cubparser

This module enable reading cubical complexes from text files using chomp format.
"""


def vertices_from_file(file_name):
    vertices = []
    min_vertex = []
    max_vertex = []
    
    with open(file_name, 'r') as f:
        for line in f:
            if line and line[0] == '(':
                vertex = tuple(map(int, line.strip()[1:-1].split(',')))
                vertices.append(vertex)
                if not min_vertex:
                    min_vertex = list(vertex)
                else:
                    for i, n in enumerate(vertex):
                        if min_vertex[i] > n:
                            min_vertex[i] = n
                if not max_vertex:
                    max_vertex = list(vertex)
                else:
                    for i, n in enumerate(vertex):
                        if max_vertex[i] < n:
                            max_vertex[i] = n

    shape = tuple(2 * (np.abs(min_vertex) + max_vertex) + 1)
    return tuple(tuple(2 * (np.abs(min_vertex) + vertex) + 1) for vertex in vertices), shape


def str_to_interval(s):
    return tuple(map(int, s[1:-1].split(',')))


def intervals_to_cubemap(intervals):
    return tuple(sum(I) if len(I) == 2 else 2 * sum(I) for I in intervals)


def intervals_from_file(file_name):
    intervals = []
    shape = []
    with open(file_name, 'r') as f:
        for line in f:
            if line and line[0] in '[(':
                interval_list = line.strip().split('x')
                interval_list = [s.strip() for s in interval_list]
                interval = list(map(str_to_interval, interval_list))
                intervals.append(interval)
                if not shape:
                    shape = [max(I) for I in interval]
                else:
                    for i, I in enumerate(interval):
                        if shape[i] < max(I):
                            shape[i] = max(I)

    cmaps, shape = tuple(intervals_to_cubemap(I) for I in intervals), tuple(2 * np.array(shape) + 1)
    return cmaps, shape

if __name__ == '__main__':
    # Test 1: OK
    # cmaps, shape = vertices_from_file('tests/kleinbot.cub')
    # for cube_map in cmaps:
    #     print(cube_map, shape)
    # print('=' * 80)
    # Test 2: OK
    # cmaps, shape = intervals_from_file('tests/mybing2.cel')
    # for cube_map in cmaps:
    #     print(cube_map, shape)

    # Test 3
    # cmaps, shape = intervals_from_file('tests/kleinbot2.cel')
    # t = tuple(-min(cube_map[i] for cube_map in cmaps) for i in range(len(shape)))
    # s = tuple(max(cube_map[i] + t[i] for cube_map in cmaps) + 1 for i in range(len(shape)))
    # for cube_map in cmaps:
    #     print(cube_map, t, tuple(map(add, t, cube_map)), tuple(map(add, t, shape)), s)
    # for cube_map in cmaps:
    #     print(cube_map, shape, all(0 <= cube_map[i] < shape[i] for i in range(len(shape))))

    # Test 4
    cube_maps, shape = intervals_from_file('tests/bing2.cel')
    for cube_map in cube_maps:
        print(cube_map)
        input('>>')
