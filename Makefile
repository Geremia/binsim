# Makefile to compile one C program and one CUDA program

# Rule to compile the C program (Default target)
C: binsim.c
	cc -Ofast -pedantic -Wall -std=c18 -o binsim binsim.c

# Rule to compile the CUDA program
CUDA: binsim.cu
	nvcc -O3 -o binsimCUDA binsim.cu

all: C CUDA

# Clean up binaries
clean:
	rm -f binsim binsimCUDA


