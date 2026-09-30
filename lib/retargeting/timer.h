#ifndef RETARGETING_TIMER_H
#define RETARGETING_TIMER_H

#include <chrono>

namespace retargeting {

// Measures wall-clock time between consecutive lap() calls.
class Stopwatch {
public:
    Stopwatch() : last_(Clock::now()) {}

    // Seconds since construction or the previous lap().
    double lap()
    {
        Clock::time_point now = Clock::now();
        double seconds = std::chrono::duration<double>(now - last_).count();
        last_ = now;
        return seconds;
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point last_;
};

}  // namespace retargeting

#endif  // RETARGETING_TIMER_H
