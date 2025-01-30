from dolfinx import default_scalar_type
from dolfinx.fem import (Constant, Function, functionspace, assemble_scalar,
                         dirichletbc, form, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_cube, locate_entities, meshtags
from dolfinx.plot import vtk_mesh

from mpi4py import MPI
from ufl import (FacetNormal, Measure, SpatialCoordinate, TestFunction, TrialFunction,
                 div, dot, dx, grad, inner, lhs, rhs)

import numpy as np
import pyvista

# Create a 3D unit cube mesh
mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)

# Exact solution for comparison (example function)
u_ex = lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2
x = SpatialCoordinate(mesh)

# Define physical parameters and boundary conditions
s = u_ex(x)
f = -div(grad(u_ex(x)))
n = FacetNormal(mesh)
g = -dot(n, grad(u_ex(x)))
kappa = Constant(mesh, default_scalar_type(1))
r = Constant(mesh, default_scalar_type(1000))

# Define function space and variational form
V = functionspace(mesh, ("Lagrange", 1))
u, v = TrialFunction(V), TestFunction(V)
F = kappa * inner(grad(u), grad(v)) * dx - inner(f, v) * dx
print(F)

# Define boundary markers and conditions for each face of the cube
boundaries = [
    (1, lambda x: np.isclose(x[0], 0)),  # x = 0 face
    (2, lambda x: np.isclose(x[0], 1)),  # x = 1 face
    (3, lambda x: np.isclose(x[1], 0)),  # y = 0 face
    (4, lambda x: np.isclose(x[1], 1)),  # y = 1 face
    (5, lambda x: np.isclose(x[2], 0)),  # z = 0 face
    (6, lambda x: np.isclose(x[2], 1))  # z = 1 face
]

# Create facet markers for the cube's boundaries
facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for marker, locator in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

mesh.topology.create_connectivity(fdim, mesh.topology.dim)
ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)


# Define the boundary condition classes
class BoundaryCondition():
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_D = Function(V)
            u_D.interpolate(values)
            facets = facet_tag.find(marker)
            dofs = locate_dofs_topological(V, fdim, facets)
            self._bc = dirichletbc(u_D, dofs)
        elif type == "Neumann":
            self._bc = inner(values, v) * ds(marker)
        elif type == "Robin":
            self._bc = values[0] * inner(u - values[1], v) * ds(marker)
        else:
            raise TypeError(f"Unknown boundary condition: {type}")

    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type


# Define Dirichlet, Neumann, and Robin boundary conditions
boundary_conditions = [
    # BoundaryCondition("Dirichlet", 1, u_ex),
    # BoundaryCondition("Dirichlet", 2, u_ex),
    # BoundaryCondition("Robin", 3, (r, s)),
    BoundaryCondition("Neumann", 4, g),
    BoundaryCondition("Neumann", 5, g),
    # BoundaryCondition("Robin", 6, (r, s))
]

# Apply boundary conditions
bcs = []
for condition in boundary_conditions:
    if condition.type == "Dirichlet":
        bcs.append(condition.bc)
    else:
        F += condition.bc

# Solve the problem
a = lhs(F)
L = rhs(F)
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Visualize the solution using PyVista
pyvista_cells, cell_types, geometry = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
grid.point_data["u"] = uh.x.array
grid.set_active_scalars("u")

plotter = pyvista.Plotter()
plotter.add_text("uh", position="upper_edge", font_size=14, color="black")
plotter.add_mesh(grid, show_edges=True)
plotter.show()
