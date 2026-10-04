#include "cuda_check.cuh"

__global__ void hello(int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        printf("block=%u local=%u global=%d\n",
               blockIdx.x, threadIdx.x, i);
    }
}

int main(int argc, char **argv)
{
    // Optional: ./hello_gpu [n] [threads]   (defaults: n=20, threads=8)
    const int n = (argc > 1) ? std::atoi(argv[1]) : 20;
    const int threads = (argc > 2) ? std::atoi(argv[2]) : 8;
    const int blocks = (n + threads - 1) / threads;
    hello<<<blocks, threads>>>(n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("Host: the kernel has finished.\n");
    return 0;
}
