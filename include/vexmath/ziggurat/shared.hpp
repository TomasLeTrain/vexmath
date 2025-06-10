/* shared.h: a set of simple functions used by both exponential.h and normal.h
 */
#pragma once

#include "vexmath/fast_prng/Xoroshiro128plus.hpp"
#include <math.h>
#include <memory>
#include <stdlib.h>

/* Test to see if rejection sampling is required in the overhang. See Fig. 2
 * in main text. */

#define _FAST_PRNG_SAMPLE_X(X_j, U)                    \
    (*(X_j) * pow(2, 31) + ((X_j)[-1] - *(X_j)) * (U))
#define _FAST_PRNG_SAMPLE_Y(i, U)                           \
    (Y[(i) - 1] * pow(2, 31) + (Y[(i)] - Y[(i) - 1]) * (U))

namespace math {
namespace ziggurat {
struct ziggurat_prng {
  public:
    Xoroshiro128plus prng;

    union {
        uint32_t l;
        int32_t sl;
    };

    ziggurat_prng(uint32_t seed)
        : prng(seed) {
        init(seed);
    }

    void init(uint32_t seed) {
        prng.setSeed(seed);
        l = prng.next();
    }

    uint32_t generate() {
        l = prng.next();
        return l;
    }

    ziggurat_prng& operator++(int) {
        l = prng.next();
        return *this;
    }
};

class NormalPRNG {
  public:
    ziggurat_prng fast_prng;

    NormalPRNG(uint32_t seed)
        : fast_prng(seed) {}

    void set_seed(uint32_t seed) {
        fast_prng.init(seed);
    }

    /**
     * @brief contains the PRN function utilized by the ziggurat
     *
     * @return uniformly distributed PRN in the interval [0, 2^32-1]
     */
    inline int32_t random_int31() {
        return fast_prng++.l & 0x7fffffff;
    }

    // normal functions
    uint_fast8_t _norm_sample_A(void);
    inline float normal(void);

    inline float normal(float mean, float std_deviation) {
        return mean + normal() * std_deviation;
    }

    // exponential functions
    inline float _exp_overhang(uint_fast8_t j);
    uint_fast8_t _exp_sample_A(void);
    inline float exponential(void);
};
} // namespace ziggurat
} // namespace math
