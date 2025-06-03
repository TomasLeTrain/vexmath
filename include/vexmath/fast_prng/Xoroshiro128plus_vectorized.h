/**
 * @brief Modified from code by David Blackman and Sebastiano Vigna (vigna@acm.org) by Sam Thompson
 */

#pragma once

#include <arm_neon.h>
#include <stdint.h>
#include "math/fast_prng/SplitMix32.h"

class VXoroshiro128plus
{
private:
    static inline uint32x4_t rotl(const uint32x4_t x, const int k) {
        return vshlq_n_u32(x,k) | vshrq_n_u32(x,32-k);
    }

protected:
    uint32x4x4_t s;

public:
    VXoroshiro128plus() { }

    /**
     * @brief Explicit constructor which sets the rng seed.
     * @param seed the random seed
     */
    explicit VXoroshiro128plus(uint64_t seed)
    {
        setSeed(seed);
    }

    void setSeed(uint64_t seed)
    {
        SplitMix32 seed_generator(seed);
        // Shuffle the seed generator 8 times
        seed_generator.shuffle();

        for(int i = 0;i < 4;i++){
            uint32_t a[4] = {
                seed_generator.next(),
                seed_generator.next(),
                seed_generator.next(),
                seed_generator.next()
            };
            s.val[i] = vld1q_u32(a);
        }
    }

    uint32x4_t next(void) {
        const uint32x4_t result = s.val[0] + s.val[3];

        const uint32x4_t t = vshlq_n_u32(s.val[1],9);

        s.val[2] ^= s.val[0];
        s.val[3] ^= s.val[1];
        s.val[1] ^= s.val[2];
        s.val[0] ^= s.val[3];

        s.val[2] ^= t;

        s.val[3] = rotl(s.val[3], 11);

        return result;
    }


    /* This is the jump function for the generator. It is equivalent
       to 2^64 calls to next(); it can be used to generate 2^64
       non-overlapping subsequences for parallel computations. */

    void jump(void) {
        static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

        uint32x4_t s0 = vmovq_n_u32(0);
        uint32x4_t s1 = vmovq_n_u32(0);
        uint32x4_t s2 = vmovq_n_u32(0);
        uint32x4_t s3 = vmovq_n_u32(0);
        for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
            for(int b = 0; b < 32; b++) {
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
        static const uint32_t LONG_JUMP[] = { 0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662 };

        uint32x4_t s0 = vmovq_n_u32(0);
        uint32x4_t s1 = vmovq_n_u32(0);
        uint32x4_t s2 = vmovq_n_u32(0);
        uint32x4_t s3 = vmovq_n_u32(0);
        for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
            for(int b = 0; b < 32; b++) {
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
    int32x4_t va;
    int32x4_t vb;
    int32x4_t vd;
public:

    Vuniform_int32_t(){}

    explicit Vuniform_int32_t(int32_t a,int32_t b, uint64_t seed)
    {
        const static int32x4_t vone = vmovq_n_s32(1); 
        va = vdupq_n_s32(a);
        vb = vdupq_n_s32(b);
        vd = vb - va + vone;
        setSeed(seed);
    }
    int32x4_t get_int(){
        return va + (vreinterpretq_s32_u32(next()) % vd); 
    }

    int32x4_t operator()(){
        return get_int();
    }
};

class Vuniform_float32_t : public VXoroshiro128plus {
private:
    float32x4_t va;
    float32x4_t vb;
    float32x4_t vd;
public:

    Vuniform_float32_t(){}

    explicit Vuniform_float32_t(float32_t a, float32_t b, uint64_t seed)
    {
        va = vdupq_n_f32(a);
        vb = vdupq_n_f32(b);
        vd = vb - va;
        setSeed(seed);
    }

    /**
     * @brief makes vector of floating point numbers in the range of [0,1)
     *
     * @return vector of random floats
     */
    float32x4_t get_reduced_float(void){
        const static uint32x4_t vexponent = vmovq_n_u32(127U << 23); 
        const static float32x4_t vone = vmovq_n_f32(1); 
        return vreinterpretq_f32_u32(vexponent | vshrq_n_u32(next(),9)) - vone;
    }

    /**
     * @brief makes vector of floating point numbers in the range of [a,b)
     *
     * @return vector of random floats
     */
    float32x4_t get_float(void){
        return va + (vd * get_reduced_float());
    }
    float32x4_t operator()(){
        return get_float();
    }
};

