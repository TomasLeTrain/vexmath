#include "tests/xoroshiro128_test.hpp"
#include "api.h"
#include "vexmath/fast_prng/Xoroshiro128plus.hpp"
#include "vexmath/fast_prng/Xoroshiro128plus_vectorized.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <math.h>
#include <random>
#include <stdio.h>

const int xoroshiro_N = 50000;

float output[xoroshiro_N];
int32_t int_output[xoroshiro_N];

#define TEST_FLOAT_MIN -10000.0
#define TEST_FLOAT_MAX  10000.0

#define TEST_INT_MIN -10000
#define TEST_INT_MAX  10000

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
    Vuniform_float32_t Vrand_float_gen(TEST_FLOAT_MIN, TEST_FLOAT_MAX, 2000);
    for (int i = 0; i < xoroshiro_N; i += 4) {
        vst1q_f32(output + i, Vrand_float_gen());
    }
    return 1;
}

int bench_doubleNext_Vfloat() {
    // test vectorized floats
    Vuniform_float32_t Vrand_float_gen(TEST_FLOAT_MIN, TEST_FLOAT_MAX, 2000);
    for (int i = 0; i < xoroshiro_N; i += 8) {
        float32x4_t res1, res2;
        Vrand_float_gen.double_get_float(&res1, &res2);
        vst1q_f32(output + i, res1);
        vst1q_f32(output + i + 4, res2);
    }
    return 1;
}

int bench_multiple_Vfloat() {
    // test vectorized floats
    Vuniform_float32_t Vrand_float_gen1(TEST_FLOAT_MIN, TEST_FLOAT_MAX, 2000);
    Vuniform_float32_t Vrand_float_gen2(TEST_FLOAT_MIN*5, TEST_FLOAT_MAX*5, 2000);
    Vuniform_float32_t Vrand_float_gen3(TEST_FLOAT_MIN/3, TEST_FLOAT_MAX/3, 2000);

    for (int i = 0; i < xoroshiro_N - 2; i += 4) {
        vst1q_f32(output + i,     Vrand_float_gen1());
        vst1q_f32(output + i + 1, Vrand_float_gen2());
        vst1q_f32(output + i + 2, Vrand_float_gen3());
    }
    return 1;
}

int bench_one_Vfloat() {
    float a = TEST_FLOAT_MIN*5;
    float b = TEST_FLOAT_MAX*5;
    float k = (b-a) / static_cast<float>(UINT32_MAX);

    float aa = TEST_FLOAT_MIN/3;
    float bb = TEST_FLOAT_MAX/3;
    float kk = (bb-aa) / static_cast<float>(UINT32_MAX);
    // test vectorized floats
    Vuniform_float32_t Vrand_float_gen1(TEST_FLOAT_MIN, TEST_FLOAT_MAX, 2000);

    for (int i = 0; i < xoroshiro_N - 2; i += 4) {
        vst1q_f32(output + i,     Vrand_float_gen1());
        vst1q_f32(output + i + 1, Vrand_float_gen1.get_float(a,k));
        vst1q_f32(output + i + 2, Vrand_float_gen1.get_float(aa,kk));
    }
    return 1;
}

int bench_multiple_double_Vfloat() {
    // test vectorized floats
    Vuniform_float32_t Vrand_float_gen1(TEST_FLOAT_MIN, TEST_FLOAT_MAX, 2000);
    Vuniform_float32_t Vrand_float_gen3(TEST_FLOAT_MIN/3, TEST_FLOAT_MAX/3, 2000);

    float a = TEST_FLOAT_MIN*5;
    float b = TEST_FLOAT_MAX*5;
    float k = (b-a) / static_cast<float>(UINT32_MAX);

    for (int i = 0; i < xoroshiro_N - 2; i += 4) {
        float32x4_t res1, res2;
        Vrand_float_gen1.double_get_float(&res1, &res2,a,k);
        vst1q_f32(output + i,     res1);
        vst1q_f32(output + i + 1, res2);
        vst1q_f32(output + i + 2, Vrand_float_gen3());
    }
    return 1;
}

int bench_one_double_Vfloat() {
    // test vectorized floats
    float a = TEST_FLOAT_MIN*5;
    float b = -TEST_FLOAT_MAX*5;
    float k = (b-a) / static_cast<float>(UINT32_MAX);

    float aa = TEST_FLOAT_MIN/3;
    float bb = -TEST_FLOAT_MAX/3;
    float kk = (bb-aa) / static_cast<float>(UINT32_MAX);

    Vuniform_float32_t Vrand_float_gen1(TEST_FLOAT_MIN, TEST_FLOAT_MAX, 2000);

    for (int i = 0; i < xoroshiro_N - 2; i += 4) {
        float32x4_t res1, res2;
        Vrand_float_gen1.double_get_float(&res1, &res2,a,k);
        vst1q_f32(output + i,     res1);
        vst1q_f32(output + i + 1, res2);
        vst1q_f32(output + i + 2, Vrand_float_gen1.get_float(aa,kk));
    }
    return 1;
}

int bench_int() {
    // test non - vectorized ints
    Xoroshiro128plus rng(2000);
    std::uniform_int_distribution<int> dist(TEST_INT_MIN,TEST_INT_MAX);
    for (int i = 0; i < xoroshiro_N; i++) {
        int_output[i] = dist(rng);
    }
    return 1;
}

int bench_Vint() {
    // test vectorized ints
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
            printf("the number %f was generated with bounds: %f, %f\n",output[i], TEST_FLOAT_MIN, TEST_FLOAT_MAX);
            return false;
        }
    }
    return true;
}

bool int_validator(){
    // validates that all generated ints are within the bounds
    for(int i = 0;i < xoroshiro_N;i++){
        if(int_output[i] > TEST_INT_MAX || int_output[i] < TEST_INT_MIN){
            printf("the number %d was generated with bounds: %d, %d\n",int_output[i], TEST_INT_MIN, TEST_INT_MAX);
            return false;
        }
    }
    return true;
}

bool multiple_validator(){
    return true;
}

// dislays a visual of the distribution of numbers, doubles as a test
void float_dist_display(){
    constexpr float number_of_bins = 50.0;
    int bins[static_cast<int>(number_of_bins)];
    for(int i = 0;i < number_of_bins;i++){
        bins[i] = 0;
    }

    int min_count = 10000000;
    int max_count = 0;

    int max_height = 10; // number of characters that the maximum count occupies

    for(int i = 0;i < xoroshiro_N;i++){
        int current_bin = static_cast<int>(
            ((output[i] + TEST_FLOAT_MIN) / TEST_FLOAT_MAX) // [0,1]
            * (number_of_bins-1));
        bins[current_bin]++;
    }

    for(int i = 0;i < number_of_bins;i++){
        min_count = std::min(min_count,bins[i]);
        max_count = std::max(max_count,bins[i]);
    }

    float f_max_count = static_cast<float>(max_count);
    float f_max_height = static_cast<float>(max_height);
    float mul = f_max_height / f_max_count;
    printf("distribution of numbers: low count of %d, high count of %d\n\n",min_count,max_count);

    for(int h = max_height; h >= 0;h--){
        for(int i = 0;i < number_of_bins;i++){
            // techincally same for every height, but the number of bins is small enough that it doesnt matter
            int height_level = static_cast<int>(static_cast<float>(bins[i]) * mul);
            if(height_level >= h){
                printf("█");
            }else{
                printf(" ");
            }
        }
        printf("\n");
    }

    printf("\n");
    // to print we first normalize the samples
    printf("^%f",TEST_FLOAT_MIN);
    for(int i = 0;i < number_of_bins - 10;i++){ printf(" "); }
    printf("%f\n",TEST_FLOAT_MAX);
}
//
// dislays a visual of the distribution of numbers, doubles as a test
void int_dist_display(){
    constexpr float number_of_bins = 50.0;
    int bins[static_cast<int>(number_of_bins)];
    for(int i = 0;i < number_of_bins;i++){
        bins[i] = 0;
    }

    int max_count = 0;
    int min_count = 1000000;
    int max_height = 10; // number of characters that the maximum count occupies

    for(int i = 0;i < xoroshiro_N;i++){
        int current_bin = static_cast<int>(
            ((static_cast<float>(int_output[i]) + static_cast<float>(TEST_INT_MIN)) / static_cast<float>(TEST_INT_MAX)) // [0,1]
            * (number_of_bins-1));
        bins[current_bin]++;
    }

    for(int i = 0;i < number_of_bins;i++){
        min_count = std::min(min_count,bins[i]);
        max_count = std::max(max_count,bins[i]);
    }

    float f_max_count = static_cast<float>(max_count);
    float f_max_height = static_cast<float>(max_height);
    float mul = f_max_height / f_max_count;
    printf("distribution of numbers: low count of %d, high count of %d\n\n",min_count,max_count);

    for(int h = max_height; h >= 0;h--){
        for(int i = 0;i < number_of_bins;i++){
            // techincally same for every height, but the number of bins is small enough that it doesnt matter
            int height_level = static_cast<int>(static_cast<float>(bins[i]) * mul);
            if(height_level >= h){
                printf("█");
            }else{
                printf(" ");
            }
        }
        printf("\n");
    }

    printf("\n");
    // to print we first normalize the samples
    printf("%.0f",TEST_FLOAT_MIN);
    for(int i = 0;i < number_of_bins - 10;i++){ printf(" "); }
    printf("^%.0f\n",TEST_FLOAT_MAX);
}

void run_xoshiro_bench(const char* s, int (*fn)(),bool (*validator)(),void (*displayer)()) {
    printf("benching %40s ..", s);
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
    
    // verify output is valid
    bool valid = validator();
    if(!valid){
        printf(" -> failed validity tests!");
        // return;
    }


#define REF_FREQ_MHZ 667.0
    printf(
      " -> %d elements in %3.2f milliseconds -> %3.2f numbers/microsecond\n",
      xoroshiro_N,
      d_millisec,
      numbers_microsec);

    // displayer();
}

void xoroshiro128_test() {
    for(int i = 0;i < xoroshiro_N;i++){
        // clears both lists to prevent variance due to caching and whatnot
        output[i] = 0;
        int_output[i] = 0;
    }

    printf("---------------------\n");
    printf("running xoroshiro benchmarks\n");
    run_xoshiro_bench("uniform float", bench_float,float_validator,float_dist_display);
    run_xoshiro_bench("uniform int", bench_int,int_validator,int_dist_display);
    run_xoshiro_bench("vector uniform float", bench_Vfloat,float_validator,float_dist_display);
    run_xoshiro_bench("vector uniform doubleNext float", bench_doubleNext_Vfloat,float_validator,float_dist_display);
    run_xoshiro_bench("vector uniform int", bench_Vint,int_validator,int_dist_display);

    run_xoshiro_bench("vector diff_float multiple", bench_multiple_Vfloat,multiple_validator,float_dist_display);
    run_xoshiro_bench("vector diff_float one", bench_one_Vfloat,multiple_validator,float_dist_display);

    run_xoshiro_bench("vector diff_float multiple double", bench_multiple_double_Vfloat,multiple_validator,float_dist_display);
    run_xoshiro_bench("vector diff_float one double", bench_one_double_Vfloat,multiple_validator,float_dist_display);
}
