import numpy as np
from scipy.optimize import minimize
from simulation.fem_model import run_simulation
from utils.helpers import target_value
from mpi4py import MPI

comm = MPI.COMM_WORLD

def objective(x):
    E = float(x[0])
    val = run_simulation(E=E)
    if val is None:
        return 1e6
    error = (val - target_value())**2
    if comm.rank == 0:
        print(f"[objective] E={E}, val={val}, error={error}")
    return error

def run_optimization():
    x0 = np.array([10.0])
    res = minimize(objective, x0, bounds=[(1.0, 100.0)])
    if comm.rank == 0:
        print("Optimization result:", res)
    return res
