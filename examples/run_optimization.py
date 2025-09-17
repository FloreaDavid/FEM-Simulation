from optimization.optimizer import run_optimization
from mpi4py import MPI

comm = MPI.COMM_WORLD

if comm.rank == 0:
    print("Starting optimization...")

run_optimization()
