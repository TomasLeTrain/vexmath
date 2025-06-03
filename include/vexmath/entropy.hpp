/**
 * @file
 * @brief Provides an entropy source based on various statistics of the robot. Inspired by [`veranda`](https://github.com/Gavin-Niederman/veranda)
 */

#pragma once

#include "pros/misc.hpp"
#include "pros/rtos.hpp"
#include <inttypes.h>
#include <cstdio>

class FnvHasher {
private:
    uint64_t hash = 14695981039346656037ull;
public:
    void reset(){
        hash = 14695981039346656037ull;
    }
    void write(const void* data, const size_t len) {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        for(size_t i = 0;i < len;i++){
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
    }
    void write_u32(const uint32_t data) {
        write(&data, 4);
    }
    void write_u64(const uint64_t data) {
        write(&data, 8);
    }

    uint64_t finish() {
        return hash;
    }
};


template<typename T>
class RobotEntropy {
private:
    FnvHasher hasher;
public:
    uint32_t operator()(){
        hasher.reset();
        hasher.write_u32(pros::battery::get_voltage());
        hasher.write_u32(pros::battery::get_current());
        hasher.write_u64(pros::micros());
        return static_cast<T>(hasher.finish());
    }
};
