/* NEON implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/

/* Copyright (C) 2011  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/


#pragma once
#include <arm_neon.h>

typedef float32x4_t v4sf;  // vector of 4 float
typedef uint32x4_t v4su;  // vector of 4 uint32
typedef int32x4_t v4si;  // vector of 4 uint32

#define c_inv_mant_mask ~0x7f800000u
#define c_cephes_SQRTHF 0.707106781186547524
#define c_cephes_log_p0 7.0376836292E-2
#define c_cephes_log_p1 - 1.1514610310E-1
#define c_cephes_log_p2 1.1676998740E-1
#define c_cephes_log_p3 - 1.2420140846E-1
#define c_cephes_log_p4 + 1.4249322787E-1
#define c_cephes_log_p5 - 1.6668057665E-1
#define c_cephes_log_p6 + 2.0000714765E-1
#define c_cephes_log_p7 - 2.4999993993E-1
#define c_cephes_log_p8 + 3.3333331174E-1
#define c_cephes_log_q1 -2.12194440e-4
#define c_cephes_log_q2 0.693359375

/* natural logarithm computed for 4 simultaneous float 
   return NaN for x <= 0
*/
inline v4sf log_ps(v4sf x) {
  v4sf one = vdupq_n_f32(1);

  x = vmaxq_f32(x, vdupq_n_f32(0)); /* force flush to zero on denormal values */
  v4su invalid_mask = vcleq_f32(x, vdupq_n_f32(0));

  v4si ux = vreinterpretq_s32_f32(x);
  
  v4si emm0 = vshrq_n_s32(ux, 23);

  /* keep only the fractional part */
  ux = vandq_s32(ux, vdupq_n_s32(c_inv_mant_mask));
  ux = vorrq_s32(ux, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
  x = vreinterpretq_f32_s32(ux);

  emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
  v4sf e = vcvtq_f32_s32(emm0);

  e = vaddq_f32(e, one);

  /* part2: 
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  v4su mask = vcltq_f32(x, vdupq_n_f32(c_cephes_SQRTHF));
  v4sf tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
  x = vsubq_f32(x, one);
  e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
  x = vaddq_f32(x, tmp);

  v4sf z = vmulq_f32(x,x);

  v4sf y = vdupq_n_f32(c_cephes_log_p0);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p1));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p2));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p3));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p4));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p5));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p6));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p7));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p8));
  y = vmulq_f32(y, x);

  y = vmulq_f32(y, z);
  

  tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q1));
  y = vaddq_f32(y, tmp);


  tmp = vmulq_f32(z, vdupq_n_f32(0.5f));
  y = vsubq_f32(y, tmp);

  tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q2));
  x = vaddq_f32(x, y);
  x = vaddq_f32(x, tmp);
  x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), invalid_mask)); // negative arg will be NAN
  return x;
}

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

/* exp() computed for 4 float at once */
inline v4sf exp_ps(v4sf x) {
  v4sf tmp, fx;

  v4sf one = vdupq_n_f32(1);
  x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
  x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

  /* perform a floorf */
  tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

  /* if greater, substract 1 */
  v4su mask = vcgtq_f32(tmp, fx);    
  mask = vandq_u32(mask, vreinterpretq_u32_f32(one));


  fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

  tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
  v4sf z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
  x = vsubq_f32(x, tmp);
  x = vsubq_f32(x, z);

  static const float cephes_exp_p[6] = { c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5 };
  v4sf y = vld1q_dup_f32(cephes_exp_p+0);
  v4sf c1 = vld1q_dup_f32(cephes_exp_p+1); 
  v4sf c2 = vld1q_dup_f32(cephes_exp_p+2); 
  v4sf c3 = vld1q_dup_f32(cephes_exp_p+3); 
  v4sf c4 = vld1q_dup_f32(cephes_exp_p+4); 
  v4sf c5 = vld1q_dup_f32(cephes_exp_p+5);

  y = vmulq_f32(y, x);
  z = vmulq_f32(x,x);
  y = vaddq_f32(y, c1);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c2);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c3);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c4);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c5);
  
  y = vmulq_f32(y, z);
  y = vaddq_f32(y, x);
  y = vaddq_f32(y, one);

  /* build 2^n */
  int32x4_t mm;
  mm = vcvtq_s32_f32(fx);
  mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
  mm = vshlq_n_s32(mm, 23);
  v4sf pow2n = vreinterpretq_f32_s32(mm);

  y = vmulq_f32(y, pow2n);
  return y;
}

