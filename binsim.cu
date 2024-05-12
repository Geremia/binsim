#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

__global__ void xorKernel(uint64_t *input1, uint64_t *input2, uint64_t *output, size_t numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements) {
		output[i] = ~(input1[i] ^ input2[i]);
	}
}

__global__ void countBitsKernel(uint64_t *data, unsigned long long *count, size_t numElements)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	unsigned long long localCount = 0;

	for (int i = index; i < numElements; i += stride) {
		localCount += __popcll(data[i]);	// Counts the number of 1s in data[i]
	}

	atomicAdd(count, localCount);
}

int main(int argc, char *argv[])
{
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <inputfile1> <inputfile2>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	FILE *f1 = fopen(argv[1], "rb");
	FILE *f2 = fopen(argv[2], "rb");

	if (!f1 || !f2) {
		perror("File opening failed");
		return EXIT_FAILURE;
	}
	// Determine the size of the file
	fseek(f1, 0, SEEK_END);
	size_t fileSize = ftell(f1);
	fseek(f1, 0, SEEK_SET);

	size_t numElements = fileSize / sizeof(uint64_t);
	size_t size = numElements * sizeof(uint64_t);

	// Allocate host memory
	uint64_t *h_input1 = (uint64_t *) malloc(size);
	uint64_t *h_input2 = (uint64_t *) malloc(size);
	uint64_t *h_output = (uint64_t *) malloc(size);
	unsigned long long h_count = 0;

	// Read data from files
	fread(h_input1, sizeof(uint64_t), numElements, f1);
	fread(h_input2, sizeof(uint64_t), numElements, f2);

	// Allocate device memory
	uint64_t *d_input1, *d_input2, *d_output;
	unsigned long long *d_count;
	cudaMalloc((void **)&d_input1, size);
	cudaMalloc((void **)&d_input2, size);
	cudaMalloc((void **)&d_output, size);
	cudaMalloc((void **)&d_count, sizeof(unsigned long long));
	cudaMemset(d_count, 0, sizeof(unsigned long long));

	// Copy data from host to device
	cudaMemcpy(d_input1, h_input1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input2, h_input2, size, cudaMemcpyHostToDevice);

	// Launch the XOR kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	xorKernel <<< blocksPerGrid, threadsPerBlock >>> (d_input1, d_input2, d_output, numElements);

	// Launch the bit counting kernel
	countBitsKernel <<< blocksPerGrid, threadsPerBlock >>> (d_output, d_count, numElements);

	// Copy the result back to host
	cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	printf("Similarity: %f\n", (float)h_count / (fileSize * 8));

	// Free device memory
	cudaFree(d_input1);
	cudaFree(d_input2);
	cudaFree(d_output);
	cudaFree(d_count);

	// Free host memory
	free(h_input1);
	free(h_input2);
}
