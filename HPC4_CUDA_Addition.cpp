// vector addition CUDA  1
#include <iostream>
#include <time.h>
#define SIZE 100000
using namespace std;

__global__ void addVect(int *vect1, int *vect2, int *resultVect)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    resultVect[i] = vect1[i] + vect2[i];
}

int main()
{
    int *d_inVect1, *d_inVect2, *d_outResultVector; // data storage for gpu
    int vect1[SIZE], vect2[SIZE], resultVect[SIZE]; // data storage for cpu
    cudaEvent_t gpu_start, gpu_stop;
    float gpu_elapsed_time;

    // Initializing both the vectors
    for (int i = 0; i < SIZE; i++)
    {
        vect1[i] = i;
        vect2[i] = i;
    }
    // Parallel code

    // Allocate memory on GPU for 3 vectors
    cudaMalloc((void *)&d_inVect1, SIZE(sizeof(int)));
    cudaMalloc((void *)&d_inVect2, SIZE(sizeof(int)));
    cudaMalloc((void *)&d_outResultVector, SIZE(sizeof(int)));

    // COPY the vector contents
    cudaMemcpy(d_inVect1, vect1, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inVect2, vect2, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Start record for gpu_start
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start, 0);

    // blk is number of blocks with each block of 1024 threads
    int blk = SIZE / 1024;
    // Call the kernel
    addVect<<<blk + 1, 1024>>>(d_inVect1, d_inVect2, d_outResultVector);
    cudaDeviceSynchronize();
    cudaEventRecord(gpu_stop, 0);
    // Copy gpu mem to cpu mem
    cudaMemcpy(resultVect, d_outResultVector, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    cout << "The time taken by GPU is :" << gpu_elapsed_time << endl;

    // verify that the GPU did the work we requested
    bool success = true;
    int total = 0;
    cout << "\n Checking " << SIZE << " values in the array.......\n";
    for (int i = 0; i < SIZE; i++)
    {
        if ((vect1[i] + vect2[i]) != resultVect[i])
        {
            printf("Error:  %d + %d != %d\n", vect1[i], vect2[i], resultVect[i]);
            success = false;
        }
        total += 1;
    }
    if (success)
        cout << "We did it " << total << "  values correct!\n";

    // Sequential code of vector addition with time measurement
    clock_t startTime = clock();
    int resultVect2[SIZE];
    for (int i = 0; i < SIZE; i++)
    {
        resultVect2[i] = vect1[i] + vect2[i];
    }
    clock_t endTime = clock();
    cout << "\nTime for sequential: " << ((float)(endTime - startTime) / CLOCKS_PER_SEC) * 1000;
    cout << "\nAll results are correct!!!, \n Speedup = " << ((float)(endTime - startTime) / CLOCKS_PER_SEC) * 1000 / gpu_elapsed_time << "\n";
    // free the memory we allocated on the GPU
    cudaFree(d_inVect1);
    cudaFree(d_inVect2);
    cudaFree(d_outResultVector);

    return 0;
}