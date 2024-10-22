#include<cstdio>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#define BLOCK_SIZE 1024
#define DIVUP(n) (int)ceil((float)n / BLOCK_SIZE)

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")

__global__ void add_kernel_impl(const float* a, const float* b, float* res, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n){
        res[idx] = a[idx] + b[idx];
    }
}

void add_launcher(const float* a, const float* b, float* res, int n){
    dim3 threadSize(BLOCK_SIZE);
    dim3 blockSize((int)ceil((float) n / BLOCK_SIZE));
    add_kernel_impl<<<blockSize, threadSize>>>(a, b, res, n);
}

at::Tensor add_gpu(const at::Tensor &a_tensor, const at::Tensor &b_tensor){
    
    at::cuda::OptionalCUDAGuard device_guard(a_tensor.device());
    CHECK_CUDA(a_tensor);
    CHECK_CUDA(b_tensor);

    at::Tensor a_contig = a_tensor.contiguous();
    at::Tensor b_contig = b_tensor.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());

    int n = 1;
    for (size_t i = 0; i< a_contig.sizes().size(); ++i){
      n *= a_tensor.size(i);
    }

    const float* a = a_tensor.data_ptr<float>();
    const float* b = b_tensor.data_ptr<float>();
    float* res = result.data_ptr<float>();

    add_launcher(a, b, res, n);

    return result;
}