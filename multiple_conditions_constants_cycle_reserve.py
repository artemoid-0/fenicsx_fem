import csv
import os
import time
from datetime import datetime

import numpy as np
from mpi4py import MPI
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_geometrical
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_cube, locate_entities, meshtags, CellType
from dolfinx.plot import vtk_mesh
from ufl import Measure, TestFunction, TrialFunction, dx, grad, inner, rhs, lhs, ds

import pyvista

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
        (1, lambda x: np.isclose(x[0], 0)),  # Грань x=0
        (2, lambda x: np.isclose(x[0], 1)),  # Грань x=L
        (3, lambda x: np.isclose(x[1], 0)),  # Грань y=0
        (4, lambda x: np.isclose(x[2], 1)),  # Грань z=L
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
    def __init__(self, bc_type, marker, values, mesh, V, u, v):
        self.type = bc_type
        if bc_type == "Dirichlet":
            # Локализация узлов для наложения условий Дирихле
            dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[values[0]], values[1]))
            self.bc = dirichletbc(default_scalar_type(values[2]), dofs, V)
        elif bc_type == "Neumann":
            # Поток на границе
            self.bc = values * v * ds(marker)
        elif bc_type == "Robin":
            # Граничное условие Робина
            self.bc = values[0] * inner(u - values[1], v) * ds(marker)
        else:
            raise ValueError(f"Неизвестное граничное условие: {bc_type}")

    def apply(self, F):
        if self.type != "Dirichlet":
            F += self.bc


def setup_problem(mesh, V, kappa, f, s, g1, g2, r):
    u, v = TrialFunction(V), TestFunction(V)

    # Вариационная форма
    F = kappa * inner(grad(u), grad(v)) * dx - inner(f, v) * dx

    # Граничные условия
    boundary_conditions = [
        BoundaryCondition("Dirichlet", 1, (0, 0, 100), mesh, V, u, v),
        BoundaryCondition("Dirichlet", 2, (1, 1, 24), mesh, V, u, v),
        BoundaryCondition("Neumann", 3, g1, mesh, V, u, v),
        BoundaryCondition("Neumann", 4, g2, mesh, V, u, v),
        BoundaryCondition("Robin", 4, (r, s), mesh, V, u, v),
    ]

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
def visualize_results(mesh, uh, V, cells_1, cells_2, save_path=None, experiment_id=None):
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
def run_experiment(mesh, V, experiment_id, csv_file, save_path, ratio):
    print(f"ЭКСПЕРИМЕНТ №{experiment_id}")

    start_experiment_time = time.time()

    # Назначение материалов
    assign_start = time.time()
    cells_1, cells_2, kappa = assign_material_regions(mesh, 400, 2.4, ratio=ratio)
    assign_time = time.time() - assign_start
    print(f"({experiment_id}) Время назначения материалов: {assign_time:.4f} секунд")

    # Определение границ
    boundary_start = time.time()
    ds = Measure("ds", domain=mesh, subdomain_data=define_boundaries(mesh))
    boundary_time = time.time() - boundary_start
    print(f"({experiment_id}) Время определения границ: {boundary_time:.4f} секунд")

    # Постановка задачи
    setup_start = time.time()
    f = Constant(mesh, default_scalar_type(10))  # Тепловыделение
    a, L, bcs = setup_problem(mesh, V, kappa, f, r=Constant(mesh, default_scalar_type(10)), s=24, g1=-15, g2=10)
    setup_time = time.time() - setup_start
    print(f"({experiment_id}) Время постановки задачи: {setup_time:.4f} секунд")

    # Решение задачи
    solve_start = time.time()
    uh = solve_problem(a, L, bcs)
    solve_time = time.time() - solve_start
    print(f"({experiment_id}) Время решения задачи: {solve_time:.4f} секунд")

    # Визуализация
    visualize_start = time.time()
    visualize_results(mesh, uh, V, cells_1, cells_2, save_path=save_path, experiment_id=experiment_id)
    visualize_time = time.time() - visualize_start
    print(f"({experiment_id}) Время визуализации: {visualize_time:.4f} секунд")

    total_time = time.time() - start_experiment_time
    print(f"({experiment_id}) Общее время эксперимента: {total_time:.4f} секунд")

    # Запись времени в CSV
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            experiment_id,
            round(assign_time, 4),
            round(boundary_time, 4),
            round(setup_time, 4),
            round(solve_time, 4),
            round(visualize_time, 4),
            round(total_time, 4)
        ])

# Главная функция
def main():
    # Подготовка CSV файла для записи времени
    save_path = "results"
    csv_file = "timing_results.csv"
    num_experiments = 96

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Experiment", "Material Assignment Time",
                         "Boundary Setup Time", "Problem Setup Time",
                         "Solve Time", "Visualization Time", "Total Time"])

    start_program_time = time.time()

    # Создание сетки
    mesh_start = time.time()
    mesh = create_unit_cube(MPI.COMM_WORLD, 20, 20, 20, cell_type=CellType.hexahedron)
    mesh_time = time.time() - mesh_start
    print(f"Время создания сетки: {mesh_time:.4f} секунд")

    V = functionspace(mesh, ("CG", 1))

    # Запуск экспериментов
    for i in range(num_experiments):
        run_experiment(mesh, V, i + 1, csv_file, save_path, ratio=0.1 * (i + 1))

    print(f"Общее время выполнения программы: {time.time() - start_program_time:.4f} секунд")

if __name__ == "__main__":
    main()
