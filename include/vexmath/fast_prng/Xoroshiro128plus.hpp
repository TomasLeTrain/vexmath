/**
 * @brief Modified from code by David Blackman, Sebastiano Vigna, and Sam
 * Thompson (vigna@acm.org)
 */

#pragma once

#include "vexmath/fast_prng/SplitMix32.hpp"
#include <cstdint>
#include <random>
#include <stdint.h>

/**
 * @class Xoroshiro128plus
 * @brief Fast PRNG generator of uint32_t numbers. Satisfies
 * UniformRandomBitGenerator so it can be used with the distributions from the
 * C++ random library
 *
 */
class Xoroshiro128plus {
  private:
    inline uint32_t rotl(const uint32_t x, int k) {
        return (x << k) | (x >> (32 - k));
    }

  protected:
    uint32_t state[4];

  public:
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
            state[i] = seed_generator.next();
        }
    }

    uint32_t next(void) {
        const uint32_t result = state[0] + state[3];

        const uint32_t t = state[1] << 9;

        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];

        state[2] ^= t;

        state[3] = rotl(state[3], 11);

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
                    s0 ^= state[0];
                    s1 ^= state[1];
                    s2 ^= state[2];
                    s3 ^= state[3];
                }
                next();
            }

        state[0] = s0;
        state[1] = s1;
        state[2] = s2;
        state[3] = s3;
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
                    s0 ^= state[0];
                    s1 ^= state[1];
                    s2 ^= state[2];
                    s3 ^= state[3];
                }
                next();
            }

        state[0] = s0;
        state[1] = s1;
        state[2] = s2;
        state[3] = s3;
    }

    // needed for satisfying UniformRandomBitGenerator
    using result_type = uint32_t;

    static constexpr result_type value = 1;

    static constexpr result_type min() {
        return 0;
    }

    static constexpr result_type max() {
        return UINT32_MAX;
    }

    result_type operator()() {
        return next();
    }
};
