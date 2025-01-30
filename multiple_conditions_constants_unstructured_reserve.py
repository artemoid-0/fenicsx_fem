from types import CellType

from dolfinx import default_scalar_type
from dolfinx.fem import (Constant, Function, functionspace, dirichletbc, locate_dofs_topological,
                         locate_dofs_geometrical)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_cube, locate_entities, meshtags, CellType
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
from ufl import (Measure, TestFunction, TrialFunction, dx, grad, inner, rhs, lhs, SpatialCoordinate, div, FacetNormal,
                 dot)

import numpy as np
import pyvista


def assign_material_regions(mesh, kappa1, kappa2):
    Q = functionspace(mesh, ("DG", 0))
    kappa = Function(Q)

    c2v = mesh.topology.connectivity(mesh.topology.dim, 0)
    np.random.seed(0)

    cells_1, cells_2 = [], []
    for cell_index in range(c2v.num_nodes):
        if np.random.rand() < 0.5:  # cell_index % 2 == 0:
            cells_1.append(cell_index)
        else:
            cells_2.append(cell_index)

    cells_1 = np.array(cells_1)
    cells_2 = np.array(cells_2)

    kappa.x.array[cells_1] = kappa1
    kappa.x.array[cells_2] = kappa2
    return cells_1, cells_2, kappa



# Создаем сетку куба
mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10, CellType.hexahedron)
V = functionspace(mesh, ("CG", 1))
u, v = TrialFunction(V), TestFunction(V)


# Определение теплопроводности
cells_1, cells_2, kappa = assign_material_regions(mesh, 400, 2.4)

# Convection Heat Transfer Coefficient
r = Constant(mesh, default_scalar_type(10))

s = 24      # Ambient Temperature
g1 = -15    # Heat flux 1
g2 = 10     # Heat flux 2
f = 10      # Heat generation


# Базовая форма вариационной задачи
F = kappa * inner(grad(u), grad(v)) * dx - inner(f, v) * dx

# Определение граней для граничных условий
boundaries = [(1, lambda x: np.isclose(x[0], 0)),  # Грань x=0
              (2, lambda x: np.isclose(x[0], 1)),  # Грань x=L
              (3, lambda x: np.isclose(x[1], 0)),  # Грань y=0
              (4, lambda x: np.isclose(x[2], 1))]  # Грань z=L


facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

# Определение меры для граничных условий
mesh.topology.create_connectivity(fdim, mesh.topology.dim)
ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)

# Класс для граничных условий
class BoundaryCondition:
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            # facets = facet_tag.find(marker)
            dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[values[0]], values[1]))
            self._bc = dirichletbc(default_scalar_type(values[2]), dofs, V)
            print("HERE AFTER")
        elif type == "Neumann":
            # n = FacetNormal(mesh)
            self._bc = inner(values, v) * ds(marker)
        elif type == "RobinBC":
            self._bc = values[0] * inner(u - values[1], v) * ds(marker)
        else:
            raise TypeError("Неизвестное граничное условие: {0:s}".format(type))

    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type



# Применяем граничные условия через класс
boundary_conditions = [
    BoundaryCondition("Dirichlet", 1, (0, 0, 100)),
    BoundaryCondition("Dirichlet", 2, (1, 1, 24)),
    # BoundaryCondition("Neumann", 3, g1),
    # BoundaryCondition("Neumann", 4, g2),
    # BoundaryCondition("RobinBC", 4, (r, s)),
]

# Список условий Дирихле и добавление остальных условий в вариационную форму
bcs = [bc.bc for bc in boundary_conditions if bc.type == "Dirichlet"]
for bc in boundary_conditions:
    if bc.type != "Dirichlet":
        F += bc.bc

# Решение задачи
a = lhs(F)
L = rhs(F)  # Правая часть равна нулю, так как источник тепла отсутствует
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "cg", "pc_type": "lu"})
uh = problem.solve()

print(uh.x.array)
print("HERE")


# Фильтрация ячеек и определение маркеров
tdim = mesh.topology.dim
num_cells_local = mesh.topology.index_map(tdim).size_local
marker = np.zeros(num_cells_local, dtype=np.int32)
# cells_1 = cells_1[cells_1 < num_cells_local]
# cells_2 = cells_2[cells_2 < num_cells_local]
marker[cells_1] = 1
marker[cells_2] = 2
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
pyvista_cells, cell_types, geometry = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
grid.point_data["u"] = uh.x.array
grid.set_active_scalars("u")
plotter = pyvista.Plotter()
plotter.add_text("uh", position="upper_edge", font_size=14, color="black")
plotter.add_mesh(grid, show_edges=True)
plotter.show()