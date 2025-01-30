import numpy as np
import dolfinx
import ufl
from dolfinx import default_scalar_type
from dolfinx.io.gmshio import read_from_msh
from mpi4py import MPI
import petsc4py
from numpy import ndarray
from petsc4py import PETSc
from dolfinx.fem import Function, FunctionSpace, DirichletBC, functionspace
from dolfinx.mesh import create_unit_cube
from dolfinx.fem.petsc import LinearProblem
import pyvista

# Создание сетки (unit cube, делён на 8x8x8 элементов)
domain = create_unit_cube(MPI.COMM_WORLD, 30, 30, 30)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
# boundary_facets = np.arange(1, 1000)

# Может быть полезно
# for facet in boundary_facets:
#     facet_nodes = domain.topology.connectivity(fdim, facet)  # Узлы, принадлежащие данной грани

# Параметры теплопроводности и источника
kappa = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.033))
f = dolfinx.fem.Constant(domain, default_scalar_type(0))  # Источник тепла

# Функциональное пространство
V = functionspace(domain, ("CG", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Определяем нагрузку (например, равномерное распределение тепла)
L = f * v * ufl.dx # if f != 0 else ufl.zero()

# Условия на границе
# boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)

# Определяем узлы на верхней и нижней грани
top_face_nodes = []
bot_face_nodes = []
# Перебираем грани, полученные с помощью dolfinx.mesh.exterior_facet_indices
for i, node in enumerate(domain.geometry.x):
    if np.isclose(node[2], 1.0):  # Проверка, что узел на верхней грани
        top_face_nodes.append(i)
    elif np.isclose(node[2], 0.0):
        bot_face_nodes.append(i)

# Конвертируем в массивы для использования в DirichletBC
top_face_nodes = np.array(top_face_nodes, dtype=np.int32)
bot_face_nodes = np.array(bot_face_nodes, dtype=np.int32)

# Задаём температуру 100 на верхней грани и 10 на других гранях
bc_top = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(100), top_face_nodes, V)
bc_other = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(24), bot_face_nodes, V)

# Собираем граничные условия
bcs = [bc_top, bc_other]
# bcs = [bc_top]

# Слабая форма
a = kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

# print(kappa.x.array)
print("L vector:", L)
print("Matrix a", a)


# Создаём задачу
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "cg", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000})

# Решаем задачу
uh = problem.solve()

# Визуализация сетки
domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

print("Mesh displayed")

# Визуализация решения
u_topology, u_cell_types, u_geometry = dolfinx.plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.show()  # Display the solution plot

print("Solution displayed")
print(uh.x.array.real)

# # Искажение по температурному полю
# warped = u_grid.warp_by_scalar()
# plotter2 = pyvista.Plotter()
# plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
# plotter2.show()  # Display the warped plot
#
# print("Warped plot displayed")
