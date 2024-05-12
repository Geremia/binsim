#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>
#include <inttypes.h>

#define BLOCK_SIZE 8		// 64 bits per block, matching uint64_t

typedef struct {
	char *file1_path;
	char *file2_path;
	uint64_t start;		// Start byte for this thread
	uint64_t end;		// End byte for this thread
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

	FILE *file1 = fopen(data->file1_path, "rb");
	FILE *file2 = fopen(data->file2_path, "rb");
	if (!file1 || !file2) {
		perror("Failed to open file in thread");
		return NULL;
	}

	fseek(file1, data->start, SEEK_SET);
	fseek(file2, data->start, SEEK_SET);

	uint64_t buffer1, buffer2;
	unsigned long local_count = 0;

	for (uint64_t pos = data->start; pos < data->end; pos += sizeof(uint64_t)) {
		if (fread(&buffer1, sizeof(uint64_t), 1, file1) == 1 && fread(&buffer2, sizeof(uint64_t), 1, file2) == 1) {
			uint64_t result = ~(buffer1 ^ buffer2);
			local_count += count_bits(result);
		}
	}

	fclose(file1);
	fclose(file2);

	return (void *)(uintptr_t) local_count;
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
		fclose(file1);
		fclose(file2);
		return 1;
	}

	// Get file size
	fseek(file1, 0, SEEK_END);
	uint64_t fileSize = ftell(file1);
	fseek(file1, 0, SEEK_SET);

	fclose(file1);
	fclose(file2);

	int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
	pthread_t threads[num_threads];
	ThreadData thread_data[num_threads];

	unsigned long total_count = 0;
	uint64_t chunkSize = fileSize / num_threads;

	for (int i = 0; i < num_threads; i++) {
		thread_data[i].file1_path = argv[1];
		thread_data[i].file2_path = argv[2];
		thread_data[i].start = i * chunkSize;
		thread_data[i].end = (i == num_threads - 1) ? fileSize : (i + 1) * chunkSize;
		thread_data[i].end -= (thread_data[i].end % BLOCK_SIZE);

		pthread_create(&threads[i], NULL, xor_and_count_bits, &thread_data[i]);
	}

	// Wait for all threads to finish and collect results
	void *thread_result;
	for (int i = 0; i < num_threads; i++) {
		pthread_join(threads[i], &thread_result);
		total_count += (uint64_t) (uintptr_t) thread_result;
	}

	printf("Similarity: %lf\n", (double)total_count / (double)(fileSize * BLOCK_SIZE));

	return 0;
}
