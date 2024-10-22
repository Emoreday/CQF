#include <cuda.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>
#define BLOCK_SIZE 256
#define GROUP_SIZE 16
#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")

__device__ int8_t get_sign(float i){
    if (i < 0) return -1;
    return 1;
}

__device__ float clamp(float num, float min, float max){
    if(num > max) num = max;
    if(num < min) num = min;
    return num;
}

__device__ float where(bool condition, float a, float b){
    if(condition) return a;
    return b;
}

__global__ void bfp_quant_kernel_impl(
    const float* src,
    const float* exponent,
    float* dist,
    float emin,
    float emax,
    float max_number,
    float mbit,
    bool stochastic,
    uint32_t blockNum,
    uint8_t blockSize
    ){
        
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < blockNum){
        float num = 0;
        int8_t sign = 0;
        float exp = 0;
        float mval = (float)pow(2, mbit);
        float esbn = (float)pow(2, emin + 1);
        float ie = 0;
        float me = 0;
        float f = 0;
        curandStateSobol64_t state;

        for(uint8_t i = 0; i < blockSize; ++i){
            num = src[idx * blockSize + i] * (float)pow(2, -exponent[idx * blockSize + i]);
            sign = get_sign(num);
            num = fabs(num);
            exp = floor(log2(num + 1.0E-10f));
            exp = clamp(exp, emin + 1, emax);
            ie = num * (float)pow(2, -exp);
            me = (float)pow(2, exp);
            f = where(num < esbn, ie, ie - 1);
            if(stochastic){
                f = clamp(floor(f * mval
                // random method, here cuda function which is 10x of normal distrbuti random number
                + fabs(curand_normal(&state) / 10)
                ), 0, mval);
                f = f / mval * me;
            }else{
                f = round(f * mval);
                f = clamp(f, 0, mval);
                f = f / mval * me;
            }
            num = where(num < esbn, f, me + f);
            num = clamp(num, -max_number, max_number);
            dist[idx * blockSize + i] = sign * num * (float)pow(2, exponent[idx * blockSize + i]);

            num = 0;
            sign = 0;
            exp = 0;
            ie = 0;
            me = 0;
            f = 0;
        }

    }
    
}

void bfp_quant_launcher(
    const float* src,
    const float* exponent,
    float* dist,
    float emin,
    float emax,
    float max_number,
    float mbit,
    bool stochastic,
    uint32_t blockNum,
    uint8_t blockSize
    ){
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((int)ceil((float) blockNum / BLOCK_SIZE));
    bfp_quant_kernel_impl<<<blocks, threads>>>(src, exponent, dist, emin, emax, max_number, mbit, stochastic, blockNum, blockSize);
}

at::Tensor bfp_quant_gpu(
    const at::Tensor &src,
    const at::Tensor &exponent,
    float emin,
    float emax,
    float max_number,
    float mbit,
    bool stochastic
    ){
        CHECK_CUDA(src);
        CHECK_CUDA(exponent);
        at::cuda::OptionalCUDAGuard device_guard(src.device());

        at::Tensor src_contig = src.contiguous();
        at::Tensor exp_contig = exponent.contiguous();
        at::Tensor dist = torch::zeros(src.sizes(), src.options());
        // at::Tensor rand = torch::rand(src.sizes(), src.options());

        const float* src_ptr = src_contig.data_ptr<float>();
        const float* exp_ptr = exp_contig.data_ptr<float>();
        // const float* rand_ptr = rand.data_ptr<float>();
        float* dist_ptr = dist.data_ptr<float>();

        uint32_t size = 1;

        for(auto s : src.sizes()){
            size *= s;
        }

        uint32_t blockNum = size / GROUP_SIZE;

        bfp_quant_launcher(src_ptr, exp_ptr, dist_ptr, emin, emax, max_number, mbit, stochastic, blockNum, GROUP_SIZE);

        return dist;
}