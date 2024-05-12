#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>		// For sysconf
#include <inttypes.h>

#define BLOCK_SIZE 8		// 64 bits per block

typedef struct {
	FILE *file1, *file2;
	uint64_t start;		// Starting offset for this thread
	uint64_t count;		// Store the count of 1s for each thread
} ThreadData;

// Function to count bits in a 64-bit integer
uint64_t count_bits(uint64_t n)
{
	uint64_t count = 0;
	while (n) {
		count += n & 1;
		n >>= 1;
	}
	return count;
}

// Thread function to perform XOR and count bits
void *xor_and_count_bits(void *arg)
{
	ThreadData *data = (ThreadData *) arg;
	fseek(data->file1, data->start, SEEK_SET);
	fseek(data->file2, data->start, SEEK_SET);

	uint64_t buffer1, buffer2;
	uint64_t local_count = 0;
	uint64_t end = data->start + BLOCK_SIZE * (data->count / BLOCK_SIZE);

	for (uint64_t i = data->start; i < end; i += sizeof(uint64_t)) {
		if (fread(&buffer1, sizeof(uint64_t), 1, data->file1) == 1 && fread(&buffer2, sizeof(uint64_t), 1, data->file2) == 1) {
			uint64_t result = buffer1 ^ buffer2;
			local_count += count_bits(result);
		}
	}

	data->count = local_count;
	return NULL;
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

	// Get file size
	fseek(file1, 0, SEEK_END);
	uint64_t fileSize = ftell(file1);
	fseek(file1, 0, SEEK_SET);

	int num_threads = sysconf(_SC_NPROCESSORS_ONLN);	// Get the number of cores
	pthread_t threads[num_threads];
	ThreadData thread_data[num_threads];
	uint64_t chunkSize = fileSize / num_threads;

	for (int i = 0; i < num_threads; i++) {
		thread_data[i].file1 = file1;
		thread_data[i].file2 = file2;
		thread_data[i].start = i * chunkSize;
		thread_data[i].count = (i == num_threads - 1) ? (fileSize - thread_data[i].start) : chunkSize;
		pthread_create(&threads[i], NULL, xor_and_count_bits, &thread_data[i]);
	}

	unsigned int total_count = 0;
	for (int i = 0; i < num_threads; i++) {
		pthread_join(threads[i], NULL);
		total_count += thread_data[i].count;
	}

	printf("Similarity: %f\n", (float)total_count / (fileSize * BLOCK_SIZE));

	fclose(file1);
	fclose(file2);

	return 0;
}
