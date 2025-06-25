#pragma once

#include <arm_neon.h>

// vectorized fast inverse square root
inline float32x4_t V_rsqrt(float32x4_t x) {
    uint32x4_t i;
    float32x4_t y;
    const float32x4_t c1 = vmovq_n_f32(0.703952253f);
    const float32x4_t c2 = vmovq_n_f32(2.38924456f);
    const uint32x4_t magic_number = vmovq_n_u32(0x5F1FFFF9);

    y = x;
    i = vreinterpretq_u32_f32(y);
    i = magic_number - vshrq_n_u32(i, 1);
    y = vreinterpretq_f32_u32(i);
    y = y * c1 * (c2 - x * y * y);
    // y = y * (threehalfs - x * y * y); // 2st iteration, unused

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
