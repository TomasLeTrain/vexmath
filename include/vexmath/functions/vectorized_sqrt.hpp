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

    return y;
}

// vectorized fast inverse square root
inline float32x4_t V_native_rsqrt(float32x4_t x) {
    // get initial estimate
    float32x4_t initial_estimate = vrsqrteq_f32(x);

    // square estimate
    float32x4_t squared_initial_estimate =
      vmulq_f32(initial_estimate, initial_estimate);

    // apply step
    float32x4_t newton_step = vrsqrtsq_f32(squared_initial_estimate, x);
    return vmulq_f32(initial_estimate, newton_step);
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

/**
 * @brief Returns square root of vector of numbers. Only an approximation, don't
 * use if precision is needed
 *
 * @param number input vector
 * @return square root of vector
 */
inline float32x4_t V_native_sqrt(float32x4_t number) {
    return number * V_native_rsqrt(number);
}
