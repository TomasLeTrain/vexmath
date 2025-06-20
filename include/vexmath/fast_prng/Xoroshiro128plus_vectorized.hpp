/**
 * @brief Modified from code by David Blackman and Sebastiano Vigna
 * (vigna@acm.org) by Sam Thompson
 */

#pragma once

#include "vexmath/fast_prng/SplitMix32.hpp"
#include <arm_neon.h>
#include <cstdint>
#include <stdint.h>

/**
 * @class VXoroshiro128plus
 * @brief Vectorized version of the xoroshiro PRNG generator
 *
 */
class VXoroshiro128plus {
  protected:
    uint32x4x4_t s;
    uint32x4x4_t s2;

  public:
    /**
     * @brief Explicit constructor which sets the rng seed.
     * @param seed the random seed
     */
    explicit VXoroshiro128plus(uint64_t seed) {
        setSeed(seed);
    }

    void setSeed(uint64_t seed) {
        SplitMix32 seed_generator(seed);
        // Shuffle the seed generator 8 times
        seed_generator.shuffle();

        for (int i = 0; i < 4; i++) {
            uint32_t a[4] = { seed_generator.next(),
                              seed_generator.next(),
                              seed_generator.next(),
                              seed_generator.next() };

            uint32_t b[4] = { seed_generator.next(),
                              seed_generator.next(),
                              seed_generator.next(),
                              seed_generator.next() };

            s.val[i] = vld1q_u32(a);
            s.val[i] = vld1q_u32(b);
        }
    }

    inline uint32x4_t next(void) {
        uint32x4_t result = s.val[0] + s.val[3];

        uint32x4_t t = vshlq_n_u32(s.val[1], 9);

        s.val[2] ^= s.val[0];
        s.val[3] ^= s.val[1];
        s.val[1] ^= s.val[2];
        s.val[0] ^= s.val[3];

        s.val[2] ^= t;

        // rotl
        s.val[3] = vshlq_n_u32(s.val[3], 11) | vshrq_n_u32(s.val[3], 32 - 11);
        return result;
    }

    inline void double_next(uint32x4_t * res1,uint32x4_t * res2) {
        *res1 = s.val[0] + s.val[3];
        *res2 = s2.val[0] + s2.val[3];

        uint32x4_t t = vshlq_n_u32(s.val[1], 9);
        uint32x4_t t2 = vshlq_n_u32(s2.val[1], 9);

        s.val[2]  ^= s.val[0];
        s2.val[2] ^= s2.val[0];

        s.val[3]  ^= s.val[1];
        s2.val[3] ^= s.val[1];

        s.val[1]  ^= s.val[2];
        s2.val[1] ^= s2.val[2];

        s.val[0]  ^= s.val[3];
        s2.val[0] ^= s2.val[3];

        s.val[2] ^= t;
        s2.val[2] ^= t2;

        // rotl
        t = vshrq_n_u32(s.val[3], (32 - 11));
        t2 = vshrq_n_u32(s2.val[3], (32 - 11));

        s.val[3] = vshlq_n_u32(s.val[3], 11);
        s2.val[3] = vshlq_n_u32(s2.val[3], 11);

        s.val[3] = vorrq_u32(s.val[3], t);
        s2.val[3] = vorrq_u32(s2.val[3], t2);

        // s.val[3] = vshlq_n_u32(s.val[3], 11) | vshrq_n_u32(s.val[3], 32 - 11);
    }




    /* This is the jump function for the generator. It is equivalent
       to 2^64 calls to next(); it can be used to generate 2^64
       non-overlapping subsequences for parallel computations. */

    void jump(void) {
        static const uint32_t JUMP[] = { 0x8764000b,
                                         0xf542d2d3,
                                         0x6fa035c3,
                                         0x77f2db5b };

        uint32x4_t s0 = vdupq_n_u32(0);
        uint32x4_t s1 = vdupq_n_u32(0);
        uint32x4_t s2 = vdupq_n_u32(0);
        uint32x4_t s3 = vdupq_n_u32(0);
        for (int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
            for (int b = 0; b < 32; b++) {
                if (JUMP[i] & UINT32_C(1) << b) {
                    s0 ^= s.val[0];
                    s1 ^= s.val[1];
                    s2 ^= s.val[2];
                    s3 ^= s.val[3];
                }
                next();
            }

        s.val[0] = s0;
        s.val[1] = s1;
        s.val[2] = s2;
        s.val[3] = s3;
    }

    /* This is the long-jump function for the generator. It is equivalent to
       2^96 calls to next(); it can be used to generate 2^32 starting points,
       from each of which jump() will generate 2^32 non-overlapping
       subsequences for parallel distributed computations. */

    void long_jump(void) {
        static const uint32_t LONG_JUMP[] = { 0xb523952e,
                                              0x0b6f099f,
                                              0xccf5a0ef,
                                              0x1c580662 };

        uint32x4_t s0 = vmovq_n_u32(0);
        uint32x4_t s1 = vmovq_n_u32(0);
        uint32x4_t s2 = vmovq_n_u32(0);
        uint32x4_t s3 = vmovq_n_u32(0);
        for (int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
            for (int b = 0; b < 32; b++) {
                if (LONG_JUMP[i] & UINT32_C(1) << b) {
                    s0 ^= s.val[0];
                    s1 ^= s.val[1];
                    s2 ^= s.val[2];
                    s3 ^= s.val[3];
                }
                next();
            }

        s.val[0] = s0;
        s.val[1] = s1;
        s.val[2] = s2;
        s.val[3] = s3;
    }
};

class Vuniform_int32_t : public VXoroshiro128plus {
  private:
    int32_t a;
    int32_t b;
    int32_t d;

  public:
    explicit Vuniform_int32_t(uint64_t seed)
        : VXoroshiro128plus(seed) {}

    // TODO: do bound checks so that a < b
    explicit Vuniform_int32_t(int32_t a, int32_t b, uint64_t seed)
        : a(a),
          b(b),
          d(b - a + 1),
          VXoroshiro128plus(seed) {}

    void setBounds(int32_t a, int32_t b) {
        this->a = a;
        this->b = b;
        this->d = b - a + 1;
    }

    // TODO: technically biased, maybe a rejection sampling approach could be
    // tried(although its not ideal since we are working with vectors)
    int32x4_t get_int() {
        // the modulus is not implemented in arm neon, so it is likely not
        // vectorized.
        return vdupq_n_s32(a) + vreinterpretq_s32_u32(next() % d);
    }

    int32x4_t operator()() {
        return get_int();
    }
};

class Vuniform_float32_t : public VXoroshiro128plus {
  private:
    float a;
    float b;
    float d;
    float k;

  public:
    explicit Vuniform_float32_t(uint64_t seed)
        : VXoroshiro128plus(seed) {}

    explicit Vuniform_float32_t(float a, float b, uint64_t seed)
        : a(a),
          b(b),
          d(b - a),
          k(d / static_cast<float>(UINT32_MAX)),
          VXoroshiro128plus(seed) {}

    void set_bounds(float a, float b) {
        this->a = a;
        this->b = b;
        this->d = b - a;
        this->k = d / static_cast<float>(UINT32_MAX);
    }

    /**
     * @brief makes vector of floating point numbers in the range of [0,1)
     *
     * @return vector of random floats
     */
    inline float32x4_t get_reduced_float(void) {
        // techincally discards some of bits generated, however the discarded
        // (lowest) bits are of lower quality anyways.
        uint32x4_t Vexponent = vdupq_n_u32(127U << 23);
        return vreinterpretq_f32_u32(Vexponent | vshrq_n_u32(next(), 9)) -
               vdupq_n_f32(1);
    }

    /**
     * @brief makes vector of floating point numbers in the range of [a,b)
     *
     * @return vector of random floats
     */
    inline float32x4_t alternative_get_float(void) {
        // slower due to more instructions being used
        // return vdupq_n_f32(a) + vmulq_n_f32(get_reduced_float(), d);
        return vmlaq_n_f32(vdupq_n_f32(a), get_reduced_float(), d);
    }

    inline float32x4_t get_float(void) {
        // faster than bit hacks since the conversion is directly
        // supported in neon
        // could be improved by not using a linear transformation
        // for better methods see
        // (Drawing random floating-point numbers from an interval)
        // [https://hal.science/hal-03282794v4/file/rand-in-range.pdf]
        // a = a + (float)(next()) * k
        return vmlaq_n_f32(vdupq_n_f32(a), vcvtq_f32_u32(next()), k);
    }

    // allows using same generator for different bounds to avoid overhead of
    // loading and storing vectors
    inline float32x4_t get_float(float a, float k) {
        return vmlaq_n_f32(vdupq_n_f32(a), vcvtq_f32_u32(next()), k);
    }

    inline void double_get_float(float32x4_t * float1, float32x4_t * float2) {
        uint32x4_t res1, res2;
        double_next(&res1, &res2);

        *float1 = vdupq_n_f32(a);
        float32x4_t tmp1 = vcvtq_f32_u32(res1);
        float32x4_t tmp2 = vcvtq_f32_u32(res2);
        *float2 = *float1;

        *float1 = vmlaq_n_f32(*float1, tmp1, k);
        *float2 = vmlaq_n_f32(*float2, tmp2, k);
    }

    inline void double_get_float(float32x4_t * float1, float32x4_t * float2, float aa, float kk) {
        uint32x4_t res1, res2;
        double_next(&res1, &res2);

        *float1 = vdupq_n_f32(a);
        *float2 = vdupq_n_f32(aa);

        float32x4_t tmp1 = vcvtq_f32_u32(res1);
        float32x4_t tmp2 = vcvtq_f32_u32(res2);

        *float1 = vmlaq_n_f32(*float1, tmp1, k);
        *float2 = vmlaq_n_f32(*float2, tmp2, kk);
    }

    float32x4_t operator()() {
        return get_float();
    }
};
