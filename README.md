# 1D FEM Simulation & Optimization

This project demonstrates a **1D finite element simulation** of a linear elastic bar using **DolfinX** and **UFL**. The goal is to simulate the displacement of the bar under a given load and perform **optimization** to determine the material property (Young's modulus) that achieves a desired displacement.

The project is structured to separate **simulation** and **optimization**:

- `simulation/fem_model.py` – defines the FEM model, including mesh creation, function spaces, trial/test functions, boundary conditions, and system assembly.
- `optimization/optimizer.py` – implements the objective function and uses `scipy.optimize.minimize` to find the optimal Young's modulus.
- `examples/run_optimization.py` – runs the complete optimization workflow.
- `utils/helpers.py` – utility functions such as the target displacement.
  
All dependencies and environment configuration are managed via **Docker**, ensuring reproducibility across systems.  

