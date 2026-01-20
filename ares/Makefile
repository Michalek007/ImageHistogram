THREADS = 4

# Default: build both
all: omp mpi cuda

# OpenMP Build and Run
omp: omp.c my_timers.c image_data.c
	icc -qopenmp omp.c my_timers.c image_data.c -o omp

run_omp: omp
	./omp $(THREADS)

# MPI Build and Run
mpi: mpi.c my_timers.c image_data.c
	mpicc mpi.c my_timers.c image_data.c -o mpi

run_mpi: mpi
	mpiexec -np $(THREADS) ./mpi

cuda: cuda.cu
	nvcc cuda.cu -o cuda

run_cuda:
	./cuda

# Cleanup
clean:
	rm -f omp mpi cuda