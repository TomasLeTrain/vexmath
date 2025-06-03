#pragma once

#include <arm_neon.h>

// vectorized fast inverse square root
inline float32x4_t V_rsqrt(float32x4_t number) {
    uint32x4_t i;
    float32x4_t x2, y;
    const float32x4_t threehalfs = vmovq_n_f32(1.5);
    const float32x4_t half = vmovq_n_f32(0.5);
    const uint32x4_t magic_number = vmovq_n_u32(0x5f3759df);

    x2 = number * half;
    y = number;
    i = vreinterpretq_u32_f32(y); // evil floating point bit level hacking
    i = magic_number - vshrq_n_u32(i, 1); // what the fuck?
    y = vreinterpretq_f32_u32(y);
    y = y * (threehalfs - (x2 * y * y)); // 1st iteration
    //	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can
    // be removed

    return y;
}

/**
 * @brief Returns square root of vector of numbers. Only an approximation, don't
 * use if precision is needed
 *
 * @param number input vector
 * @return square root of vector
 */
inline float32x4_t Vsqrt(float32x4_t number) {
    return number * V_rsqrt(number);
}
