#include "tests/xoroshiro128_test.hpp"
#include "api.h"
#include "vexmath/fast_prng/Xoroshiro128plus.hpp"
#include "vexmath/fast_prng/Xoroshiro128plus_vectorized.hpp"
#include <arm_neon.h>
#include <math.h>
#include <random>
#include <stdio.h>

const int xoroshiro_N = 10000;

float output[xoroshiro_N];

int bench_float() {
    // test non - vectorized floats
    uniform_float32_t rand_float_gen(-10, 10, 2000);
    for (int i = 0; i < xoroshiro_N; i++) {
        output[i] = rand_float_gen();
    }
    return 1;
}

int bench_Vfloat() {
    // test vectorized floats
    Vuniform_float32_t Vrand_float_gen(-10, 10, 2000);
    for (int i = 0; i < xoroshiro_N; i += 4) {
        vst1q_f32(output + i, Vrand_float_gen());
    }
    return 1;
}

void run_taylor_bench(const char* s, int (*fn)()) {
    printf("benching %20s ..", s);
    fflush(stdout);
    int32_t it0 = pros::micros(), it1;
    double iter = 0;
    // avoid variations due time of pros::micros
    for (long long i = 0; i < 200; i++) {
        iter += fn();
        i++;
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
    printf("---------------------\n");
    printf("running xoroshiro benchmarks\n");
    run_taylor_bench("uniform float", bench_float);
    run_taylor_bench("vector uniform float", bench_Vfloat);
    printf("---------------------\n");
}
