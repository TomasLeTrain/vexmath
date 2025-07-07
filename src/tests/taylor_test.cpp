#include "tests/taylor_test.hpp"
#include "api.h"
#include "vexmath/functions/trig_taylor.hpp"
#include "vexmath/functions/vectorized_exp_log.hpp"
#include "vexmath/functions/vectorized_trig.hpp"
#include "vexmath/functions/vectorized_trig_taylor.hpp"
#include <arm_neon.h>
#include <math.h>
#include <random>
#include <stdio.h>

const int taylor_N = 100000;

float x[taylor_N], center[taylor_N], t[taylor_N], xsin[taylor_N],
  xcos[taylor_N], ysin[taylor_N], ycos[taylor_N];

int bench_Vtaylor() {
    for (int i = 0; i < taylor_N; i += 4) {
        float32x4_t Vx = vld1q_f32(x + i);
        float32x4_t Vcenter = vld1q_f32(center + i);
        float32x4_t Vxsin = vld1q_f32(xsin + i);
        float32x4_t Vxcos = vld1q_f32(xcos + i);
        float32x4_t Vysin;
        float32x4_t Vycos;
        Vsincos_taylor(Vx, Vcenter, Vxsin, Vxcos, &Vysin, &Vycos);
        vst1q_f32(xsin + i, Vysin);
        vst1q_f32(xcos + i, Vycos);
    }
    return 1;
}

int bench_Vtaylor_delta() {
    for (int i = 0; i < taylor_N; i += 4) {
        float32x4_t Vt = vld1q_f32(t + i);
        float32x4_t Vxsin = vld1q_f32(xsin + i);
        float32x4_t Vxcos = vld1q_f32(xcos + i);
        float32x4_t Vysin;
        float32x4_t Vycos;
        Vsincos_taylor_delta(Vt, Vxsin, Vxcos, &Vysin, &Vycos);
        vst1q_f32(xsin + i, Vysin);
        vst1q_f32(xcos + i, Vycos);
    }
    return 1;
}

int bench_taylor() {
    for (int i = 0; i < taylor_N; i++) {
        sincos_taylor(x[i], center[i], xsin[i], xcos[i], &ysin[i], &ycos[i]);
    }
    return 1;
}

int bench_taylor_delta() {
    for (int i = 0; i < taylor_N; i++) {
        sincos_taylor_delta(t[i], xsin[i], xcos[i], &ysin[i], &ycos[i]);
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
    double numbers_microsec = taylor_N / d_microsec;

#define REF_FREQ_MHZ 667.0
    printf(
      " -> %d elements in %3.2f milliseconds -> %3.2f numbers/microsecond\n",
      taylor_N,
      d_millisec,
      numbers_microsec);
}

void taylor_test() {
    for (int i = 0; i < taylor_N; i++) {
        center[i] = ((double)(i % 2001) - 1000.0) / 1000.0; // [-1,1]
        x[i] = ((double)(i % 501) - 250.0) / 250.0; // [-1,1]
    }

    std::shuffle(center, center + taylor_N, std::default_random_engine(0));
    std::shuffle(x, x + taylor_N, std::default_random_engine(0));

    for (int i = 0; i < taylor_N; i++) {
        center[i] *= 2 * M_PI; // [-2PI,2PI]
        xsin[i] = sin(x[i]);
        xcos[i] = cos(x[i]);
    }
    // calculate t
    for (int i = 0; i < taylor_N; i++) {
        t[i] = x[i] - center[i];
    }

    printf("---------------------\n");
    printf("running taylor benchmarks\n");

    run_taylor_bench("taylor", bench_taylor);
    run_taylor_bench("taylor_delta", bench_taylor_delta);

    run_taylor_bench("Vtaylor", bench_Vtaylor);
    run_taylor_bench("Vtaylor_delta", bench_Vtaylor_delta);
    printf("---------------------\n");
}
