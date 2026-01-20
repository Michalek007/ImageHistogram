THREADS = 4

# Default: build both
all: omp mpi cuda

# OpenMP Build and Run
omp:
	icc -qopenmp omp.c my_timers.c image_data.c -o omp

run_omp: omp
	./omp $(THREADS)

# MPI Build and Run
mpi:
	mpicc mpi.c my_timers.c image_data.c -o mpi

run_mpi: mpi
	mpiexec -np $(THREADS) ./mpi

cuda:
	nvcc cuda.cu -o cuda

run_cuda:
	./cuda

# Cleanup
clean:
	rm -f omp mpi cuda