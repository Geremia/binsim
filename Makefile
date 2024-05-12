# Makefile to compile one C program and one CUDA program

# Default target
all: C

# Rule to compile the C program
C: binsim.c
	cc -Ofast -pedantic -Wall -std=c18 -o binsim binsim.c

# Rule to compile the CUDA program
CUDA: binsim.cu
	nvcc -O3 -o binsimCUDA binsim.cu

# Clean up binaries
clean:
	rm -f binsim binsimCUDA


