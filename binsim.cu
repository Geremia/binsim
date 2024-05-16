#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 8		// 64 bits per block, matching uint64_t

__device__ uint64_t count_bits(uint64_t n)
{
	uint64_t count = 0;
	while (n) {
		count += n & 1;
		n >>= 1;
	}
	return count;
}

__global__ void xorKernel(const uint64_t *input1, const uint64_t *input2, uint64_t *output, size_t numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements) {
		output[i] = ~(input1[i] ^ input2[i]);
	}
}

__global__ void countBitsKernel(const uint64_t *data, unsigned long long *count, size_t numElements)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	unsigned long long localCount = 0;

	for (int i = index; i < numElements; i += stride) {
		localCount += count_bits(data[i]);
	}

	atomicAdd(count, localCount);
}

__global__ void processRemainderKernel(const uint8_t *input1, const uint8_t *input2, size_t start, size_t size, unsigned long long *count)
{
	uint64_t buffer1 = 0, buffer2 = 0;
	uint64_t mask = (1ULL << (size * 8)) - 1;

	for (int i = 0; i < size; i++) {
		buffer1 |= ((uint64_t) input1[start + i]) << (i * 8);
		buffer2 |= ((uint64_t) input2[start + i]) << (i * 8);
	}

	uint64_t result = ~(buffer1 ^ buffer2) & mask;
	unsigned long long localCount = count_bits(result);

	atomicAdd(count, localCount);
}

int main(int argc, char *argv[])
{
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <inputfile1> <inputfile2>\n", argv[0]);
		return 1;
	}

	FILE *file1 = fopen(argv[1], "rb");
	FILE *file2 = fopen(argv[2], "rb");
	if (!file1 || !file2) {
		perror("Failed to open files");
		return 1;
	}

	// Determine file size
	fseek(file1, 0, SEEK_END);
	size_t fileSize = ftell(file1);
	fseek(file1, 0, SEEK_SET);

	size_t numElements = fileSize / sizeof(uint64_t);
	size_t remainder = fileSize % sizeof(uint64_t);

	// Allocate host memory
	uint64_t *h_input1 = (uint64_t *) malloc(fileSize);
	uint64_t *h_input2 = (uint64_t *) malloc(fileSize);
	uint64_t *h_output = (uint64_t *) malloc(fileSize);
	unsigned long long h_count = 0;

	// Read data from files
	fread(h_input1, 1, fileSize, file1);
	fread(h_input2, 1, fileSize, file2);

	// Allocate device memory
	uint64_t *d_input1, *d_input2, *d_output;
	unsigned long long *d_count;
	cudaMalloc((void **)&d_input1, numElements * sizeof(uint64_t));
	cudaMalloc((void **)&d_input2, numElements * sizeof(uint64_t));
	cudaMalloc((void **)&d_output, numElements * sizeof(uint64_t));
	cudaMalloc((void **)&d_count, sizeof(unsigned long long));
	cudaMemset(d_count, 0, sizeof(unsigned long long));

	// Copy data to device
	cudaMemcpy(d_input1, h_input1, numElements * sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_input2, h_input2, numElements * sizeof(uint64_t), cudaMemcpyHostToDevice);

	// Launch the XOR kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	xorKernel <<< blocksPerGrid, threadsPerBlock >>> (d_input1, d_input2, d_output, numElements);

	// Launch the bit counting kernel
	countBitsKernel <<< blocksPerGrid, threadsPerBlock >>> (d_output, d_count, numElements);

	// Process any remaining bytes
	if (remainder > 0) {
		uint8_t *d_input1_bytes, *d_input2_bytes;
		cudaMalloc((void **)&d_input1_bytes, fileSize);
		cudaMalloc((void **)&d_input2_bytes, fileSize);

		cudaMemcpy(d_input1_bytes, h_input1, fileSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_input2_bytes, h_input2, fileSize, cudaMemcpyHostToDevice);

		processRemainderKernel <<< 1, 1 >>> (d_input1_bytes, d_input2_bytes, numElements * sizeof(uint64_t), remainder, d_count);

		cudaFree(d_input1_bytes);
		cudaFree(d_input2_bytes);
	}

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
	free(h_output);

	fclose(file1);
	fclose(file2);

	return 0;
}
