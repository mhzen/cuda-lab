// Exercise 8: fixed version (guard restored).
// Run: compute-sanitizer --tool memcheck ./vector_add_fixed
#include "cuda_check.cuh"
#include <vector>

__global__ void vector_add(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main()
{
    const int n = 257, threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const std::size_t bytes = n * sizeof(float);
    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n);
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < n; ++i) if (c[i] != 3.0f) ++errors;
    std::printf("n=%d blocks=%d errors=%d\n", n, blocks, errors);
    return 0;
}
