#include<iostream>
#include<math.h>

int8_t get_sign(float i){
    if (i < 0) return -1;
    return 1;
}

float clamp(float num, float min, float max){
    if(num > max) num = max;
    if(num < min) num = min;
    return num;
}

float where(bool condition, float a, float b){
    if(condition) return a;
    return b;
}

void bfp_quant_kernel_impl(
    const float* src,
    const float* exponent,
    const float* rand,
    float* dist,
    float emin,
    float emax,
    float max_number,
    float mbit,
    bool stochastic,
    uint32_t blockNum,
    uint8_t blockSize,
    uint32_t idx
    ){
    
    if(idx < blockNum){
        float num = 0;
        int8_t sign = 0;
        float exp = 0;
        float mval = (float)pow(2, mbit);
        float esbn = (float)pow(2, emin + 1);
        float ie = 0;
        float me = 0;
        float f = 0;

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
                f = clamp(floor(f * mval + rand[idx * blockSize + i]), 0, mval);
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

int main(){
    float src[64] = {0};
    float exp[64] = {1};
    float ran[64] = {0};
    float dist[64] = {0};
    float max = 0;
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 16; ++ j){
            src[i*4 + j] = (float)rand() / RAND_MAX;
            ran[i*4 + j] = (float)rand() / RAND_MAX;
        }
    }
    for(int i = 0; i < 4; ++i){
        bfp_quant_kernel_impl(src, exp, ran, dist,-4, 3, pow(2,3)*(2-pow(2,-3)), 5, true, 4, 16,i);
    }
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 16; ++ j){
            std::cout<< dist[i*4 + j]<< "  ";
            if(max < fabs(src[i*4+j]-dist[i*4+j])) max = fabs(src[i*4+j]-dist[i*4+j]);
        }
    }
    std::cout << max << std::endl;
    return 0;
}