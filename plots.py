import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from itertools import product


def homothetic(center, ratio, point):
    return tuple(np.array(point) * ratio + (1 - ratio) * np.array(center))


def edge_vertices(cell):
    vertices = list(zip(*cell.intervals))
    center = tuple((np.array(vertices[0]) + np.array(vertices[1]))/2)
    ratio = 0.8
    return [tuple(reversed(homothetic(center, ratio, point))) for point in vertices]


def face_vertices(cell):
    vertices = list(product(*cell.intervals))
    vertices[2], vertices[3] = vertices[3], vertices[2]
    center = tuple((np.array(vertices[0]) + np.array(vertices[2]))/2)
    ratio = 0.8
    return [tuple(reversed(homothetic(center, ratio, point))) for point in vertices]


def barycenter(cell):
    return sum(np.array(s) for s in set(product(*cell.intervals))) / (2 ** cell.dim)


def configure_2D_plot(K):
    fig, ax = plt.subplots()
    ax.set_autoscaley_on(False)
    ax.set_autoscalex_on(False)
    max_val = max(K.shape) // 2 + 1
    ax.set_ylim([-1, max_val])
    ax.set_xlim([-1, max_val])
    plt.xticks(np.arange(max_val))
    plt.yticks(np.arange(max_val))
    return fig, ax


def plot_cubical_2D_cell_complex(K):
    fig, axes = configure_2D_plot(K)
    verts = np.array([cell.cube_map for cell in K(0)]) / 2
    edges = np.array([edge_vertices(cell) for cell in K(1)])
    faces = np.array([face_vertices(cell) for cell in K(2)])
    axes.scatter(verts[:, 1], verts[:, 0], alpha=0.5)
    edges_coll = LineCollection(edges)
    edges_coll.set_color('green')
    edges_coll.set_alpha(0.5)
    faces_coll = PolyCollection(faces, )
    faces_coll.set_color('pink')
    axes.add_collection(edges_coll)
    axes.add_collection(faces_coll)
    return fig, axes


def plot_cubical_2D_vector_field(K, V):
    coker_V = [sigma for sigma in K if V(sigma)]
    p = np.array([barycenter(sigma) for sigma in coker_V])
    a = np.array([sigma.cube_map for sigma in coker_V])
    b = np.array([list(V(sigma))[0].cube_map for sigma in coker_V])
    u = (b - a) * 0.2
    # a *= 0.5
    plt.quiver(p[:,1], p[:,0], u[:,1], u[:,0], color='red', width=0.004, alpha=0.5)


def plot_2D_cells(cells, axes):
    verts = np.array([cell.cube_map for cell in cells if cell.dim == 0]) / 2
    edges = np.array([edge_vertices(cell) for cell in cells if cell.dim == 1])
    faces = np.array([face_vertices(cell) for cell in cells if cell.dim == 2])
    if verts.shape[0]:
        axes.scatter(verts[:, 1], verts[:, 0], marker='s')
    edges_coll = LineCollection(edges)
    edges_coll.set_color('green')
    edges_coll.set_linewidth(2)
    faces_coll = PolyCollection(faces, )
    faces_coll.set_color('pink')
    axes.add_collection(edges_coll)
    axes.add_collection(faces_coll)