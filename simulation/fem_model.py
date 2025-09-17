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

    # Linear Lagrange function space
    V = functionspace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)


    # Lame parameters
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    # Zero body force
    f = Constant(mesh, PETSc.ScalarType(1.0))

    a = mu * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    # Dirichlet BC at left end
    def left(x):
        return np.isclose(x[0], 0.0)

    bc = dirichletbc(PETSc.ScalarType(0.0),
                     locate_dofs_geometrical(V, left),
                     V)

    #Solve linear system
    uh = Function(V)
    problem = LinearProblem(a, L, bcs=[bc], u=uh)
    uh = problem.solve()

    # Displacement at right end x=1
    point = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    value = uh.eval(point, np.array([0], dtype=np.int32))

    if comm.rank == 0:
        print(f"[run_simulation] E={E}, u(1)={value[0]}")

    return float(value[0])
