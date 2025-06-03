/**
 * @file
 * @brief provides vectorized sin and cos second-degree tayler approximations
 */

#pragma once
#include <arm_neon.h>

/**
 * @brief approximates sin and cos for x
 *
 * @param x values for which sin and cos are evaluated
 * @param c center of the approximation
 * @param xsin the value of sin at c
 * @param xcos the value of cos at c
 * @param ysin where the result of the sin approximation is stored
 * @param ycos where the result of the cos approximation is stored
 */

inline void Vsincos_taylor(float32x4_t x,
                           float32x4_t c,
                           float32x4_t xsin,
                           float32x4_t xcos,
                           float32x4_t* ysin,
                           float32x4_t* ycos) {
    // sin = sin(c) + cos(c) (x-c) - sin(c) (x-c)^2/2
    // cos = cos(c) - sin(c) (x-c) - cos(c) (x-c)^2/2
    //
    static const float32x4_t Vhalf = vmovq_n_f32(0.5);

    float32x4_t t = (x - c);
    *ysin = xsin + xcos * t - xsin * t * t * Vhalf;
    *ycos = xcos - xsin * t - xcos * t * t * Vhalf;
}

/**
 * @brief approximates sin and cos for x. Simplifies calcuations if (x-c) is
 * already available
 *
 * @param difference between x and c, in other words (x-c)
 * @param xsin the value of sin at c
 * @param xcos the value of cos at c
 * @param ysin where the result of the sin approximation is stored
 * @param ycos where the result of the cos approximation is stored
 */
inline void Vsincos_taylor_delta(float32x4_t t,
                                 float32x4_t xsin,
                                 float32x4_t xcos,
                                 float32x4_t* ysin,
                                 float32x4_t* ycos) {
    // sin = sin(c) + cos(c) (x-c) - sin(c) (x-c)^2/2
    // cos = cos(c) - sin(c) (x-c) - cos(c) (x-c)^2/2
    static const float32x4_t Vhalf = vmovq_n_f32(0.5);

    *ysin = xsin + xcos * t - xsin * t * t * Vhalf;
    *ycos = xcos - xsin * t - xcos * t * t * Vhalf;
}
