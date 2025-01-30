import csv
import json
import os
import shutil
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pyvista
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_geometrical
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_cube, locate_entities, meshtags, CellType
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
from pympler import asizeof
from ufl import Measure, TestFunction, TrialFunction, dx, grad, inner, rhs, lhs, ds

np.random.seed(0)

def assign_material_regions(mesh, kappa1, kappa2, ratio=0.5):
    """
    Функция назначает материалы ячейкам сетки в заданном соотношении.

    :param mesh: Сетка куба
    :param kappa1: Теплопроводность материала 1
    :param kappa2: Теплопроводность материала 2
    :param ratio: Соотношение материалов (доля материала 1)
    :return: Индексы ячеек для каждого материала и функция теплопроводности
    """
    Q = functionspace(mesh, ("DG", 0))
    kappa = Function(Q)

    # Соединение ячеек с вершинами
    c2v = mesh.topology.connectivity(mesh.topology.dim, 0)
    np.random.seed(0)

    cells_1, cells_2 = [], []
    for cell_index in range(c2v.num_nodes):
        if np.random.rand() < ratio:  # Материал 1 с вероятностью "ratio"
            cells_1.append(cell_index)
        else:  # Материал 2 в остальных случаях
            cells_2.append(cell_index)

    # Конвертация в numpy массивы
    cells_1 = np.array(cells_1, dtype=np.int32)
    cells_2 = np.array(cells_2, dtype=np.int32)

    # Заполнение функции теплопроводности
    kappa.x.array[cells_1] = kappa1
    kappa.x.array[cells_2] = kappa2
    return cells_1, cells_2, kappa


# Генерация меток для граней
def define_boundaries(mesh):
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),  # Грань x=0 (левая грань)
        (2, lambda x: np.isclose(x[0], 1)),  # Грань x=L (правая грань)
        (3, lambda x: np.isclose(x[1], 0)),  # Грань y=0 (передняя грань)
        (4, lambda x: np.isclose(x[1], 1)),  # Грань y=L (задняя грань)
        (5, lambda x: np.isclose(x[2], 0)),  # Грань z=0 (нижняя грань)
        (6, lambda x: np.isclose(x[2], 1)),  # Грань z=L (верхняя грань)
    ]
    facet_indices, facet_markers = [], []
    fdim = mesh.topology.dim - 1
    for marker, locator in boundaries:
        facets = locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    return meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])


# Формирование задачи
class BoundaryCondition:
    # Соответствие маркеров осям и значениям
    boundary_map = {
        1: (0, 0),  # x = 0
        2: (0, 1),  # x = 1
        3: (1, 0),  # y = 0
        4: (1, 1),  # y = 1
        5: (2, 0),  # z = 0
        6: (2, 1),  # z = 1
    }

    marker_to_side = {
        1: "LEFT",
        2: "RIGHT",
        3: "FRONT",
        4: "BACK",
        5: "BOTTOM",
        6: "TOP",
    }

    def __init__(self, bc_type, *, marker, value, V, u, v):
        """
        Параметры:
        - bc_type: Тип граничного условия ("Dirichlet", "Neumann", "Robin").
        - marker: Маркер границы.
        - value: Значение для граничного условия.
        - mesh: Сетка.
        - V: Пространство функций.
        - u: Тестовая функция (trial function).
        - v: Пробная функция (test function).
        """
        self.type = bc_type
        self.marker = marker
        self.value = value

        if marker not in BoundaryCondition.boundary_map:
            raise ValueError(f"Неизвестный маркер границы: {marker}")

        axis, position = BoundaryCondition.boundary_map[marker]

        if bc_type == "Dirichlet":
            # Локализация узлов для наложения условий Дирихле
            dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[axis], position))
            self.bc = dirichletbc(default_scalar_type(value), dofs, V)
        elif bc_type == "Neumann":
            # Поток на границе
            self.bc = value * v * ds(marker)
        elif bc_type == "Robin":
            # Граничное условие Робина
            self.bc = value[0] * inner(u - value[1], v) * ds(marker)
        else:
            raise ValueError(f"Неизвестное граничное условие: {bc_type}")

    def apply(self, F):
        """Применить граничное условие."""
        if self.type != "Dirichlet":
            F += self.bc

    def __str__(self):
        """Описание граничного условия в виде строки."""
        side = BoundaryCondition.marker_to_side.get(self.marker, "UNKNOWN")
        description = f"BoundaryCondition(type={self.type}, side={side}"
        if self.type == "Dirichlet":
            description += f", value={self.value})"
        elif self.type == "Neumann":
            description += f", flux_value={self.value})"
        elif self.type == "Robin":
            description += f", coefficient={self.value[0]}, reference_value={self.value[1]})"
        return description

    def __repr__(self):
        return self.__str__()



def setup_problem(*, mesh, kappa, f, u, v, boundary_conditions: list[BoundaryCondition]):

    # Вариационная форма
    F = kappa * inner(grad(u), grad(v)) * dx - inner(f, v) * dx


    # Отдельно собираем условия Дирихле
    bcs = [bc.bc for bc in boundary_conditions if bc.type == "Dirichlet"]

    # Применяем остальные условия к вариационной форме
    for bc in boundary_conditions:
        if bc.type != "Dirichlet":
            bc.apply(F)

    # Матрица системы и правая часть
    a = lhs(F)
    L = rhs(F)
    return a, L, bcs



# Решение задачи
def solve_problem(a, L, bcs):
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "cg", "pc_type": "lu"})
    return problem.solve()


# Визуализация результатов
def visualize_results(*, mesh, uh, V, cells_1, cells_2, save_path=None, experiment_id=None):
    tdim = mesh.topology.dim
    num_cells_local = mesh.topology.index_map(tdim).size_local
    marker = np.zeros(num_cells_local, dtype=np.int32)
    cells_1 = cells_1[cells_1 < num_cells_local]
    cells_2 = cells_2[cells_2 < num_cells_local]
    marker[cells_1] = 1
    marker[cells_2] = 2

    # Генерация базового имени файла
    def generate_filename(base_name):
        if experiment_id is not None:
            return f"{base_name}_exp{experiment_id}.png"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.png"

    # Визуализация регионов материалов
    topology, cell_types, x = vtk_mesh(mesh, tdim, np.arange(num_cells_local, dtype=np.int32))
    p = pyvista.Plotter(window_size=[800, 800], off_screen=bool(save_path))
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    grid.cell_data["Marker"] = marker
    grid.set_active_scalars("Marker")
    p.add_mesh(grid, show_edges=True)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        material_image_path = os.path.join(save_path, generate_filename("material_regions"))
        p.screenshot(material_image_path)
        print(f"Сохранено изображение регионов материалов: {material_image_path}")
    else:
        p.show()

    # Визуализация распределения температуры
    pyvista_cells, cell_types, geometry = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
    grid.point_data["u"] = uh.x.array
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter(off_screen=bool(save_path))
    plotter.add_text("uh", position="upper_edge", font_size=14, color="black")
    plotter.add_mesh(grid, show_edges=True)

    if save_path:
        temperature_image_path = os.path.join(save_path, generate_filename("temperature_distribution"))
        plotter.screenshot(temperature_image_path)
        print(f"Сохранено изображение распределения температуры: {temperature_image_path}")
    else:
        plotter.show()

    del plotter.ren_win
    plotter.close()
    plotter.deep_clean()
    plotter._theme = None
    del plotter
    pyvista.close_all()



# Функция для выполнения одного эксперимента
def run_experiment(mesh, save_path, kappa1, kappa2, ratio, r, s, g1, g2):
    """Выполняет тепловой анализ одной модели."""
    start_experiment_time = time.time()

    # Назначение материалов
    cells_1, cells_2, kappa = assign_material_regions(mesh, kappa1, kappa2, ratio=ratio)
    MPI.COMM_WORLD.Barrier()

    # Определение границ
    ds = Measure("ds", domain=mesh, subdomain_data=define_boundaries(mesh))
    MPI.COMM_WORLD.Barrier()

    # Определение граничных условий
    V = functionspace(mesh, ("CG", 1))
    u, v = TrialFunction(V), TestFunction(V)

    boundary_conditions = [
        BoundaryCondition("Dirichlet", marker=1, value=100, V=V, u=u, v=v),
        BoundaryCondition("Dirichlet", marker=2, value=24, V=V, u=u, v=v),
        BoundaryCondition("Neumann", marker=3, value=g1, V=V, u=u, v=v),
        BoundaryCondition("Neumann", marker=6, value=g2, V=V, u=u, v=v),
        BoundaryCondition("Robin", marker=6, value=(r, s), V=V, u=u, v=v),
    ]

    # Постановка задачи
    f = 10# Constant(mesh, default_scalar_type(10))  # Источник тепла
    a, L, bcs = setup_problem(mesh=mesh, kappa=kappa, f=f, u=u, v=v, boundary_conditions=boundary_conditions)
    MPI.COMM_WORLD.Barrier()

    # Решение задачи
    uh = solve_problem(a, L, bcs)
    MPI.COMM_WORLD.Barrier()

    # Визуализация
    if MPI.COMM_WORLD.rank == 0:
        visualize_results(mesh=mesh, uh=uh, V=V, cells_1=cells_1, cells_2=cells_2,
                          save_path=save_path, experiment_id=1)

    total_time = time.time() - start_experiment_time
    print(f"Process {MPI.COMM_WORLD.rank}: experiment completed in {total_time:.4f} seconds.")

    # # Запись данных только процессом 0
    # if MPI.COMM_WORLD.rank == 0:
    #     with XDMFFile(MPI.COMM_WORLD, f"{save_path}/simulation_results.xmdf", "w") as file:
    #         file.write_mesh(mesh)
    #         file.write_function(uh)

    return {"Total Time": round(total_time, 4)}




# Главная функция
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Starting thermal analysis...")

    # Параметры сетки и материалов
    save_path = "results_single"
    kappa1, kappa2 = 400, 2.4
    nx, ny, nz = 40, 40, 40
    r = 10 # Constant(MPI.COMM_WORLD, default_scalar_type(10))
    s, g1, g2 = 24, -15, 10

    # Удаление предыдущих результатов
    if rank == 0:
        shutil.rmtree(save_path, ignore_errors=True)

    # Создание сетки
    mesh = create_unit_cube(MPI.COMM_WORLD, nx, ny, nz, cell_type=CellType.hexahedron)
    MPI.COMM_WORLD.Barrier()

    # Запуск эксперимента
    result = run_experiment(mesh, save_path, kappa1, kappa2, ratio=0.25, r=r, s=s, g1=g1, g2=g2)

    # Вывод времени выполнения
    if rank == 0:
        print(f"Experiment completed successfully. Total time: {result['Total Time']} seconds.")




if __name__ == "__main__":
    main()
