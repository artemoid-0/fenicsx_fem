import dolfinx.mesh
from dolfinx import default_scalar_type
from dolfinx.fem import (Constant, dirichletbc, Function, functionspace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import create_unit_cube, locate_entities, locate_entities_boundary, meshtags, CellType

from dolfinx.plot import vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                 dx, grad, inner, dot, Measure, hexahedron)

from mpi4py import MPI

import numpy as np
import pyvista


# Создание трехмерной сетки (unit cube)
mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, cell_type=CellType.hexahedron)  # 10x10x10 элементов
V = functionspace(mesh, ("Lagrange", 1))
u = TrialFunction(V)
v = TestFunction(V)
# a = dot(grad(u), grad(v)) * dx


# Дирихле
dofs_x0 = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))         # CHANGED
dofs_yL = locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 1))         # CHANGED

u_bc = Function(V)
bc1 = dirichletbc(default_scalar_type(100), dofs_x0, V)
bc2 = dirichletbc(default_scalar_type(24), dofs_yL, V)

bcs = [bc1, bc2]




# Теплопроводность
def Omega_0(x):
    return x[0] <= 0.5  # Условие для первой области по координате x


def Omega_1(x):
    return x[0] >= 0.5  # Условие для второй области по координате x


def Omega_white(x):
    # Находим индекс ячейки, в которой находится точка x
    cell = mesh.find_cell(x)

    # Для 3D сетки определяем индексы ячеек
    i, j, k = cell[0], cell[1], cell[2] if len(cell) > 2 else (0, 0, 0)

    # Используем сумму индексов ячеек для шахматного распределения
    return (i + j + k) % 2 == 0



Q = functionspace(mesh, ("DG", 0))
kappa = Function(Q)
# cells_0 = locate_entities(mesh, mesh.topology.dim, Omega_0)
# cells_1 = locate_entities(mesh, mesh.topology.dim, Omega_1)
#
# kappa.x.array[cells_0] = np.full_like(cells_0, 1, dtype=default_scalar_type)
# kappa.x.array[cells_1] = np.full_like(cells_1, 0.1, dtype=default_scalar_type)

# Функция для определения материала в шахматном порядке (с учётом 3D)
def Omega_chessboard(cell_index):
    cell = topology.cell


# Получаем все клетки сетки (по всем клеткам в 3D)
topology = mesh.topology
geometry = mesh.geometry
c2v = mesh.topology.connectivity(mesh.topology.dim, 0)

np.random.seed(0)
cells_0, cells_1 = [], []
print(c2v.links(0))
for cell_index in range(c2v.num_nodes):
    vertices_of_cell = c2v.links(cell_index)
    print(vertices_of_cell)
    if np.random.rand() < 0.5:# cell_index % 2 == 0:
        cells_0.append(cell_index)
    else:
        cells_1.append(cell_index)

cells_0 = np.array(cells_0)
cells_1 = np.array(cells_1)

kappa.x.array[cells_0] = np.full_like(cells_0, 400, dtype=default_scalar_type)
kappa.x.array[cells_1] = np.full_like(cells_1, 0.033, dtype=default_scalar_type)

# exit()



# Неймана
# Создание меток границы для условий Неймана
boundaries = {
    1: lambda x: np.isclose(x[1], 0.0),  # y=0
    2: lambda x: np.isclose(x[2], 1.0)   # z=L
}

# Применение меток границ
facet_indices, facet_markers = [], []
for marker, locator in boundaries.items():
    facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, locator)
    # print(facets)
    # exit()
    facet_indices.append(facets)
    facet_markers.append(np.full(len(facets), marker, dtype=np.int32))
facet_indices = np.concatenate(facet_indices)
facet_markers = np.concatenate(facet_markers)
mt = meshtags(mesh, mesh.topology.dim - 1, facet_indices, facet_markers)

# Определение поверхностных мер для каждой границы
ds = Measure("ds", domain=mesh, subdomain_data=mt)

# Граничные условия Неймана
L = Constant(mesh, default_scalar_type(0)) * v * dx  # Начальная правая часть
# L += Constant(mesh, default_scalar_type(-1500)) * v * ds(1)  # Поток на границе y=0
# L += Constant(mesh, default_scalar_type(1000)) * v * ds(2)   # Поток на границе z=L






# f = Constant(mesh, default_scalar_type(-6))
a = inner(kappa * grad(u), grad(v)) * dx
L = Constant(mesh, default_scalar_type(1)) * v * dx

# Задание граничных условий для одной из граней
dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 1))  # Граница по y=0

problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()










# Фильтрация ячеек и определение маркеров
tdim = mesh.topology.dim
num_cells_local = mesh.topology.index_map(tdim).size_local
marker = np.zeros(num_cells_local, dtype=np.int32)
cells_0 = cells_0[cells_0 < num_cells_local]
cells_1 = cells_1[cells_1 < num_cells_local]
marker[cells_0] = 1
marker[cells_1] = 2
mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, x = vtk_mesh(mesh, tdim, np.arange(num_cells_local, dtype=np.int32))

# Визуализация маркеров
p = pyvista.Plotter(window_size=[800, 800])
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Marker"] = marker
grid.set_active_scalars("Marker")
p.add_mesh(grid, show_edges=True)
p.show()

# Визуализация решения
p2 = pyvista.Plotter(window_size=[800, 800])
grid_uh = pyvista.UnstructuredGrid(*vtk_mesh(V))
grid_uh.point_data["u"] = uh.x.array.real
grid_uh.set_active_scalars("u")
p2.add_mesh(grid_uh, show_edges=True)
p2.show()

print(uh.x.array.real)
