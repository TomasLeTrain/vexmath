/**
 * @file
 * @brief provides sin and cos second-degree tayler approximations
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

inline void sincos_taylor(float x,float c,float xsin, float xcos, float *ysin, float *ycos) {
    // sin = sin(c) + cos(c) (x-c) - sin(c) (x-c)^2/2
    // cos = cos(c) - sin(c) (x-c) - cos(c) (x-c)^2/2
    float t = (x - c);
    *ysin = xsin + xcos * t - xsin * t * t * 0.5;
    *ycos = xcos - xsin * t - xcos * t * t * 0.5;
}

/**
 * @brief approximates sin and cos for x. Simplifies calcuations if (x-c) is already available
 *
 * @param difference between x and c, in other words (x-c)
 * @param xsin the value of sin at c
 * @param xcos the value of cos at c
 * @param ysin where the result of the sin approximation is stored
 * @param ycos where the result of the cos approximation is stored
 */
inline void sincos_taylor_delta(float t,float xsin, float xcos, float *ysin, float *ycos) {
    // sin = sin(c) + cos(c) (x-c) - sin(c) (x-c)^2/2
    // cos = cos(c) - sin(c) (x-c) - cos(c) (x-c)^2/2
    *ysin = xsin + xcos * t - xsin * t * t * 0.5;
    *ycos = xcos - xsin * t - xcos * t * t * 0.5;
    
}
