#include <cstdio>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#define BLOCK_SIZE 1024

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")



__global__ void mapping_dist_324_kernel_impl(
    const float* src,
    const int* index,
    float* dist,
    const int* shape_src,
    const int* shape_index,
    const int* shape_dist,
    int size_src
    ){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // for target dim = A, K, B, other
    // for src dim = A, B, other
    // for index dim = A, B
    int size_other = shape_src[2];
    int size_src_B = shape_src[1] * size_other;
    int size_index_B = shape_index[1];
    int size_dist_B = size_src_B;
    int size_dist_K = shape_dist[1] * size_dist_B;

    int order_other = idx % size_other;
    int order_B = idx % size_src_B / size_other;
    int order_A = idx / size_src_B;

    if (idx < size_src){
        dist[order_A * size_dist_K + index[order_A * size_index_B + order_B] * size_dist_B + order_B * size_other + order_other] = src[idx];
    }
}

void mapping_dist_324_launcher(const float* src, const int* index, float* dist, const int* shape_src, const int* shape_index, const int* shape_dist, int size_src){
    dim3 threadSize(BLOCK_SIZE);
    dim3 blockSize((int)ceil((float) size_src / BLOCK_SIZE));
    mapping_dist_324_kernel_impl<<<blockSize, threadSize>>>(src, index, dist, shape_src, shape_index, shape_dist, size_src);
}

at::Tensor mapping_dist_324_gpu(
  const at::Tensor &src_tensor,
  const at::Tensor &index_tensor,
//  at::Tensor &dist_tensor,
  const at::Tensor shape_src_tensor,
  const at::Tensor shape_index_tensor,
  const at::Tensor shape_dist_tensor
  ){
    at::cuda::OptionalCUDAGuard device_guard(src_tensor.device());
    CHECK_CUDA(src_tensor);
    CHECK_CUDA(index_tensor);

    at::Tensor src_contig = src_tensor.contiguous();
    at::Tensor index_contig = index_tensor.contiguous();
    at::Tensor shape_dist_cpu_tensor = shape_dist_tensor.to(torch::kCPU); 

    const float* src_ptr = src_contig.data_ptr<float>();
    const int* index_ptr = index_contig.data_ptr<int>();
    
    const int* shape_src = shape_src_tensor.data_ptr<int>();
    const int* shape_index = shape_index_tensor.data_ptr<int>();
    const int* shape_dist = shape_dist_tensor.data_ptr<int>();
    const int* shape_dist_cpu = shape_dist_cpu_tensor.data_ptr<int>();

    at::Tensor dist_tensor = torch::zeros({shape_dist_cpu[0], shape_dist_cpu[1], shape_dist_cpu[2], shape_dist_cpu[3]}, src_tensor.options());
    float* dist_ptr = dist_tensor.data_ptr<float>();

    int size_src = 1;
    for(int i = 0; i < src_contig.sizes().size(); ++i){
    	size_src *= src_contig.size(i);
    }

    mapping_dist_324_launcher(src_ptr, index_ptr, dist_ptr, shape_src, shape_index, shape_dist, size_src);

    return dist_tensor;
}

__global__ void mapping_src_223_kernel_impl(
    const float* src,
    const int* index,
    float* dist,
    const int* shape_src,
    const int* shape_index,
    const int* shape_dist,
    int size_dist
    ){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // for target dim = A,  B, other
    // for src dim = A, K
    // for index dim = A, B
    int size_other = shape_dist[2];
    int size_dist_B = shape_dist[1] * size_other;
    int size_index_B = shape_index[1];
    int size_src_K = shape_src[1];

    int order_B = idx % size_dist_B / size_other;
    int order_A = idx / size_dist_B;

    if (idx < size_dist){
        dist[idx] = src[order_A * size_src_K + index[order_A * size_index_B + order_B]];
    }
}

void mapping_src_223_launcher(const float* src, const int* index, float* dist, const int* shape_src, const int* shape_index, const int* shape_dist, int size_dist){
    dim3 threadSize(BLOCK_SIZE);
    dim3 blockSize((int)ceil((float) size_dist / BLOCK_SIZE));
    mapping_src_223_kernel_impl<<<blockSize, threadSize>>>(src, index, dist, shape_src, shape_index, shape_dist, size_dist);
}

at::Tensor mapping_src_223_gpu(
  const at::Tensor &src_tensor,
  const at::Tensor &index_tensor,
//  at::Tensor &dist_tensor,
  const at::Tensor shape_src_tensor,
  const at::Tensor shape_index_tensor,
  const at::Tensor shape_dist_tensor
  ){
    at::cuda::OptionalCUDAGuard device_guard(src_tensor.device());
    CHECK_CUDA(src_tensor);
    CHECK_CUDA(index_tensor);

    at::Tensor src_contig = src_tensor.contiguous();
    at::Tensor index_contig = index_tensor.contiguous();
    at::Tensor shape_dist_cpu_tensor = shape_dist_tensor.to(torch::kCPU); 

    const float* src_ptr = src_contig.data_ptr<float>();
    const int* index_ptr = index_contig.data_ptr<int>();
    
    const int* shape_src = shape_src_tensor.data_ptr<int>();
    const int* shape_index = shape_index_tensor.data_ptr<int>();
    const int* shape_dist = shape_dist_tensor.data_ptr<int>();
    const int* shape_dist_cpu = shape_dist_cpu_tensor.data_ptr<int>();

    at::Tensor dist_tensor = torch::zeros({shape_dist_cpu[0], shape_dist_cpu[1], shape_dist_cpu[2]}, src_tensor.options());
    float* dist_ptr = dist_tensor.data_ptr<float>();

    int size_dist = 1;
    for(int i = 0; i < shape_dist_tensor.size(0); ++i){
    	size_dist *= shape_dist_cpu[i];
    }

    mapping_src_223_launcher(src_ptr, index_ptr, dist_ptr, shape_src, shape_index, shape_dist, size_dist);

    return dist_tensor;
}
