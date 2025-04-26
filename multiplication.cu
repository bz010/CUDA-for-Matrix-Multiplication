#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h>


// DO NOT change the kernel function
__global__ void vector_add(int *a, int *b, int *c)
{
// DO NOT change the kernel function
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	c[index] = a[index] + b[index];
}


#define N (2048*2048)
#define THREADS_PER_BLOCK 128

int main()
{
	int* a, * b, * c, * golden;
	int* d_a, * d_b, * d_c;
	int size = N * sizeof(int);

	// from class
	#define NSTREAMS 4
	int nsdata = N / NSTREAMS;
	int iBytes = nsdata * sizeof(float);
    // ngrid.x = (nsdata + nblock.x - 1) / nblock.x;
	cudaStream_t streams[NSTREAMS];
	for (int i = 0; i < NSTREAMS; i++)
	{
		cudaStreamCreate(&streams[i]);
	}
	cudaMalloc( (void **) &d_a, size );
	cudaMalloc( (void **) &d_b, size );
	cudaMalloc( (void **) &d_c, size );

	a = (int *)malloc( size );
	b = (int *)malloc( size );
	c = (int *)malloc( size );
	golden = (int *)malloc(size);

	for( int i = 0; i < N; i++ )
	{
		a[i] = b[i] = i;
		golden[i] = a[i] + b[i];
		c[i] = 0;
	}
    // same for loop from lecture with some sweaks in how data is read since pointer was given
	// Async dma no time event(waitevent)
	for (int i = 0; i < NSTREAMS; i++)
	{
		int offset = i * nsdata;
		cudaMemcpyAsync(d_a + offset, a + offset, iBytes, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_b + offset, b + offset, iBytes, cudaMemcpyHostToDevice, streams[i]);
		vector_add <<<(nsdata + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, streams[i] >>> (d_a + offset, d_b + offset, d_c + offset);
		cudaMemcpyAsync(c + offset, d_c + offset, iBytes, cudaMemcpyDeviceToHost, streams[i]);
	}

	cudaDeviceSynchronize();
	

	bool pass = true;
	for (int i = 0; i < N; i++) {
		if (golden[i] != c[i])
			pass = false;
	}
	
	if (pass)
		printf("PASS\n");
	else
		printf("FAIL\n");

	printf("Ben Zhang, A16268103\n");

	free(a);
	free(b);
	free(c);
	free(golden);
	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );
	
	return 0;
} 
