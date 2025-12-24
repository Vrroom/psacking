#pragma once
#include <chrono>
#include <iostream>
#include <string>

// Enable/disable timing output (default: enabled)
#ifndef PROFILE_TIMING
#define PROFILE_TIMING 1
#endif

class ScopedTimer {
    std::chrono::high_resolution_clock::time_point start_;
    const char* name_;
public:
    ScopedTimer(const char* name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
#if PROFILE_TIMING
        std::cerr << "[TIMING] " << name_ << ": " << (us / 1000.0) << " ms\n";
#endif
    }
};

#if PROFILE_TIMING
#define SCOPED_TIMER(name) ScopedTimer _timer_##__LINE__(name)
#else
#define SCOPED_TIMER(name)
#endif
