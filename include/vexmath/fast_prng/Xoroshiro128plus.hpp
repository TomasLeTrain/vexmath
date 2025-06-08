/**
 * @brief Modified from code by David Blackman and Sebastiano Vigna
 * (vigna@acm.org) by Sam Thompson
 */

#pragma once

#include "vexmath/fast_prng/SplitMix32.hpp"
#include <random>
#include <stdint.h>

class Xoroshiro128plus {
  private:
    inline uint32_t rotl(const uint32_t x, int k) {
        return (x << k) | (x >> (32 - k));
    }

  protected:
    uint32_t s[4];

  public:
    Xoroshiro128plus() {}

    /**
     * @brief Explicit constructor which sets the rng seed.
     * @param seed the random seed
     */
    explicit Xoroshiro128plus(uint64_t seed) {
        setSeed(seed);
    }

    virtual void setSeed(uint64_t seed) {
        SplitMix32 seed_generator(seed);
        // Shuffle the seed generator 8 times
        seed_generator.shuffle();
        for (int i = 0; i < 4; i++) {
            s[i] = seed_generator.next();
        }
    }

    uint32_t next(void) {
        const uint32_t result = s[0] + s[3];

        const uint32_t t = s[1] << 9;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 11);

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

        uint32_t s0 = 0;
        uint32_t s1 = 0;
        uint32_t s2 = 0;
        uint32_t s3 = 0;
        for (int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
            for (int b = 0; b < 32; b++) {
                if (JUMP[i] & UINT32_C(1) << b) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                next();
            }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
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

        uint32_t s0 = 0;
        uint32_t s1 = 0;
        uint32_t s2 = 0;
        uint32_t s3 = 0;
        for (int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
            for (int b = 0; b < 32; b++) {
                if (LONG_JUMP[i] & UINT32_C(1) << b) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                next();
            }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }
};

class uniform_int32_t : public Xoroshiro128plus {
  private:
    int32_t a;
    int32_t b;
    std::uniform_int_distribution<int> dist;

  public:
    uniform_int32_t() {}

    explicit uniform_int32_t(int32_t a, int32_t b, uint64_t seed)
        : a(a),
          b(b),
          dist(a,b)
           {
        setSeed(seed);
    }

    int32_t get_int() {
        return dist(*this);
    }
    int32_t operator()() {
        return get_int();
    }
};

class uniform_float32_t : public Xoroshiro128plus {
  private:
    float a;
    float b;
    float d;

  public:
    uniform_float32_t() {}

    explicit uniform_float32_t(float a, float b, uint64_t seed)
        : a(a),
          b(b) {
        d = b - a;
        setSeed(seed);
    }

    /**
     * @brief makes a random floating point number in the range of [0,1)
     *
     * @return returns a random float in the range [0,1)
     */
    float get_reduced_float(void) {
        const uint32_t exponent = 127U << 23;
        uint32_t manipulated_uint = exponent | (next() >> 9);
        return reinterpret_cast<float&>(manipulated_uint) - 1.0;
    }

    /**
     * @brief makes vector of floating point numbers in the range of [a,b)
     *
     * @return vector of random floats
     */
    float get_float(void) {
        return a + (d * get_reduced_float());
    }

    float operator()() {
        return get_float();
    }
};
