FROM dolfinx/dolfinx:stable

WORKDIR /home/shared

COPY . /home/shared

RUN pip install --no-cache-dir scipy numpy

CMD ["python3", "examples/run_optimization.py"]
