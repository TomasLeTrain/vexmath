/**
 * @file
 * @brief provides vectorized sin and cos second-degree tayler approximations
 */

#pragma once
#include <arm_neon.h>

/* possible formulations
 * total instructions: 6
 
    *ysin = (xsin + t * xcos) - xsin * (t * t * Vhalf);
    *ycos = (xcos - t * xsin) - xcos * (t * t * Vhalf);

    ----------
    
    1 -> t2 = vmul t, t
    2 -> t2_half = vmul t2, vhalf
    
           4     3                               
    sin = vmls (vmla xsin, t, xcos), xsin, t2_half
           6     5                               
    cos = vmls (vmls xcos, t, xsin), xcos, t2_half
*/

/* different formulation - implemented
 * total instructions: 5

    *ysin = xsin + t * (xcos - xsin * t * Vhalf);
    *ycos = xcos - t * (xsin - xcos * t * Vhalf);
    
    ----------

    1 -> t_half = vmul t, half
    
            3              2
    sin = vmla xsin, t, (vmls xcos, xsin, t_half)
            5              4
    cos = vmls xcos, t, (vmls xsin, xcos, t_half)
*/


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

    float32x4_t t = vsubq_f32(x, c);
    // t * 0.5
    float32x4_t t_half = vmulq_n_f32(t, 0.5f);

    float32x4_t sin = xcos;
    float32x4_t cos = xsin;

    // sin = xcos - xsin * (t * 0.5)
    // cos = xsin - xcos * (t * 0.5)
    sin = vmlsq_f32(sin, xsin, t_half);
    cos = vmlsq_f32(cos, xcos, t_half);

    // theoretically slower due to pointer dereference + more mov instructions
    // in practice this should be inlined and optimized by the compiler avoiding mul and add instructions
    *ysin = xsin;
    *ycos = xcos;

    *ysin = vmlaq_f32(*ysin,t,sin);
    *ycos = vmlsq_f32(*ycos,t,cos);

    // sin = vmulq_f32(t,sin);
    // cos = vmulq_f32(t,cos);
    // *ysin = vaddq_f32(xsin,sin);
    // *ycos = vsubq_f32(xcos,cos);
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

    // t * 0.5
    float32x4_t t_half = vmulq_n_f32(t, 0.5f);

    float32x4_t sin = xcos;
    float32x4_t cos = xsin;

    // sin = xcos - xsin * (t * 0.5)
    // cos = xsin - xcos * (t * 0.5)
    sin = vmlsq_f32(sin, xsin, t_half);
    cos = vmlsq_f32(cos, xcos, t_half);

    // theoretically slower due to pointer dereference + more mov instructions
    // in practice this should be inlined and optimized by the compiler avoiding mul+add instructions
    *ysin = xsin;
    *ycos = xcos;

    *ysin = vmlaq_f32(*ysin,t,sin);
    *ycos = vmlsq_f32(*ycos,t,cos);

    // sin = vmulq_f32(t,sin);
    // cos = vmulq_f32(t,cos);
    // *ysin = vaddq_f32(xsin,sin);
    // *ycos = vsubq_f32(xcos,cos);
}
