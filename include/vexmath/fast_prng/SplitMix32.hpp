//
// Created by Sam Thompson on 14/03/2019.
//

#pragma once

#include <stdint.h>

/**
 * @brief A random number generator using the splitmix64 algorithm - this is provided for generating the shuffle table
 * within the main Xoroshiro256+ algorithm.
 */
class SplitMix32
{
private:
    uint32_t x{}; /* The state can be seeded with any value. */

public:

    /**
     * @brief Default constructor, taking the RNG seed.
     * @param seed the seed to use
     */
    explicit SplitMix32(uint32_t seed) : x(seed){}

    /**
     * @brief Generates the next random integer
     * @return a random integer in the range of 0 to 2^64
     */
    uint32_t next()
    {
        uint32_t z = (x += UINT32_C(0x9E3779B9));
        z = (z ^ (z >> 16)) * UINT32_C(0x21f0aaad);
        z = (z ^ (z >> 15)) * UINT32_C(0x735a2d97);
        return z ^ (z >> 15);
    }

    /**
     * @brief Shuffle the random number 8 times.
     */
    void shuffle()
    {
        for(unsigned int i = 0; i < 8; i++)
        {
            next();
        }
    }
};
