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
            s.val[i] = vld1q_u32(a);
        }
    }

    uint32x4_t next(void) {
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

    explicit Vuniform_int32_t(int32_t a, int32_t b, uint64_t seed)
        : a(a),
          b(b),
          d(b - a + 1),
          VXoroshiro128plus(seed) {}

    void setBounds(int32_t a, int32_t b) {
        this->a = a;
        this->b = a;
        this->d = b - a + 1;
    }

    // TODO: technically biased, maybe a rejection sampling approach could be
    // tried(although its not ideal since we are working with vectors)
    // could also try to convert to float,manipulate it, then convert to int
    int32x4_t get_int() {
        // the modulus is not implemented in arm neon, so it is likely not
        // vectorized. would be a good idea to switch to one of the alternatives
        // outlined above
        return vdupq_n_f32(a) + (vreinterpretq_s32_u32(next()) % d);
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

    void set_bounds(float a, float b){
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
    float32x4_t get_reduced_float(void) {
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
    float32x4_t get_float(void) {
        // could be improved by not using a linear transformation
        // for better methods see
        // (Drawing random floating-point numbers from an interval)
        // [https://hal.science/hal-03282794v4/file/rand-in-range.pdf]
        return vdupq_n_f32(a) + vmulq_n_f32(get_reduced_float(), d);

    }

    float32x4_t alternate_get_float(void) {
        // this might be faster since the uint -> float conversion is directly
        // supported in neon
        return vdupq_n_f32(a) + vmulq_n_f32(vcvtq_f32_u32(next()), k);
    }

    float32x4_t operator()() {
        return get_float();
        // return alternate_get_float();
    }
};
