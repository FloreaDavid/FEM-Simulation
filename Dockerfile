# Folosește imaginea oficială DolfinX
FROM dolfinx/dolfinx:stable

# Setează directorul de lucru
WORKDIR /home/shared

# Copiază codul în container
COPY . /home/shared

RUN pip install --no-cache-dir scipy numpy

# Optional: rulează direct scriptul la start
CMD ["python3", "examples/run_optimization.py"]
