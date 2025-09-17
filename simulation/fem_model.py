import dolfinx
import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import Function, functionspace, Constant, dirichletbc, locate_dofs_geometrical
from dolfinx.fem.petsc import LinearProblem

comm = MPI.COMM_WORLD


def run_simulation(E=10.0, nu=0.3):
    mesh = dolfinx.mesh.create_unit_interval(comm, 10)
    V = functionspace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # parametri simplificați
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    f = Constant(mesh, PETSc.ScalarType(1.0))

    a = mu * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    def left(x):
        return np.isclose(x[0], 0.0)

    bc = dirichletbc(PETSc.ScalarType(0.0),
                     locate_dofs_geometrical(V, left),
                     V)

    uh = Function(V)
    problem = LinearProblem(a, L, bcs=[bc], u=uh)
    uh = problem.solve()

    # returnează valoarea la x=1
    point = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    value = uh.eval(point, np.array([0], dtype=np.int32))

    if comm.rank == 0:
        print(f"[run_simulation] E={E}, u(1)={value[0]}")

    return float(value[0])
