#include "tests/xoroshiro128_test.hpp"
#include "api.h"
#include "vexmath/fast_prng/Xoroshiro128plus.hpp"
#include "vexmath/fast_prng/Xoroshiro128plus_vectorized.hpp"
#include <arm_neon.h>
#include <math.h>
#include <random>
#include <stdio.h>

const int xoroshiro_N = 50000;

float output[xoroshiro_N];
int32_t int_output[xoroshiro_N];

#define TEST_FLOAT_MIN -1000000.0
#define TEST_FLOAT_MAX 10000.0

#define TEST_INT_MIN -1000
#define TEST_INT_MAX 1000000

int bench_float() {
    // test non - vectorized floats
    Xoroshiro128plus rng(2000);
    std::uniform_real_distribution<float> dist(TEST_FLOAT_MIN,TEST_FLOAT_MAX);
    for (int i = 0; i < xoroshiro_N; i++) {
        output[i] = dist(rng);
    }
    return 1;
}

int bench_Vfloat() {
    // test vectorized floats
    Vuniform_float32_t Vrand_float_gen(TEST_FLOAT_MAX, TEST_FLOAT_MAX, 2000);
    for (int i = 0; i < xoroshiro_N; i += 4) {
        vst1q_f32(output + i, Vrand_float_gen.alternate_get_float());
    }
    return 1;
}

int bench_int() {
    // test non - vectorized floats
    Xoroshiro128plus rng(2000);
    std::uniform_int_distribution<int> dist(TEST_INT_MIN,TEST_INT_MAX);
    for (int i = 0; i < xoroshiro_N; i++) {
        int_output[i] = dist(rng);
    }
    return 1;
}

int bench_Vint() {
    // test vectorized floats
    Vuniform_int32_t Vrand_int_gen(TEST_INT_MIN, TEST_INT_MAX, 2000);
    for (int i = 0; i < xoroshiro_N; i += 4) {
        vst1q_s32(int_output + i, Vrand_int_gen());
    }
    return 1;
}

bool float_validator(){
    // validates that all generated floats are within the bounds
    for(int i = 0;i < xoroshiro_N;i++){
        if(output[i] > TEST_FLOAT_MAX || output[i] < TEST_FLOAT_MIN){
            return false;
        }
    }
    return true;
}

bool int_validator(){
    // validates that all generated ints are within the bounds
    for(int i = 0;i < xoroshiro_N;i++){
        if(int_output[i] > TEST_INT_MAX || int_output[i] < TEST_INT_MIN){
            return false;
        }
    }
    return true;

}

void run_xoshiro_bench(const char* s, int (*fn)(),bool (*validator)()) {
    printf("benching %20s ..", s);
    fflush(stdout);
    int32_t it0 = pros::micros(), it1;
    double iter = 0;
    // avoid variations due time of pros::micros
    for (long long i = 0; i < 200; i++) {
        iter += fn();
        i++;
    }
    // verify output is valid
    bool valid = validator();
    if(!valid){
        printf(" -> failed validity tests!");
        // return;
    }

    it1 = pros::micros();
    double micro_t0 = (double)it0, micro_t1 = (double)it1;
    // double t0 = micro_t0/1000000.0,t1 = micro_t1/1000000.0;

    double d_microsec = ((micro_t1 - micro_t0) / ((double)iter));
    double d_millisec = d_microsec / 1000.0;
    double numbers_microsec = xoroshiro_N / d_microsec;

#define REF_FREQ_MHZ 667.0
    printf(
      " -> %d elements in %3.2f milliseconds -> %3.2f numbers/microsecond\n",
      xoroshiro_N,
      d_millisec,
      numbers_microsec);
}

void xoroshiro128_test() {
    for(int i = 0;i < xoroshiro_N;i++){
        // clears both lists to prevent variance due to caching and whatnot
        output[i] = 0;
        int_output[i] = 0;
    }

    printf("---------------------\n");
    printf("running xoroshiro benchmarks\n");
    run_xoshiro_bench("uniform float", bench_float,float_validator);
    run_xoshiro_bench("uniform int", bench_int,int_validator);
    run_xoshiro_bench("vector uniform float", bench_Vfloat,float_validator);
    run_xoshiro_bench("vector uniform int", bench_Vint,int_validator);
}
