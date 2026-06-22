/*
 * This source file is part of EMT, the ElectroMagneticTool.
 *
 * Copyright (C) 2013-2015, Matteo Cicuttin - matteo.cicuttin@uniud.it
 * Department of Electrical Engineering, University of Udine
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of Udine nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(s) ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE AUTHOR(s) BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <sys/resource.h>
#include <sys/time.h>
#include <chrono>
#include <iostream>
#include <string>
#include <map>

class timecounter
{
    std::chrono::time_point<std::chrono::steady_clock>    begin, end;
    bool running;

public:
    timecounter()
        : running(false)
    {}

    void tic()
    {
        begin = std::chrono::steady_clock::now();
        running = true;
    }

    double toc()
    {
        end = std::chrono::steady_clock::now();
        running = false;
        std::chrono::duration<double> diff = end-begin;
        return diff.count();
    }

    double elapsed() const
    {
        if (not running)
            return std::chrono::duration<double>(end-begin).count();
        
        std::chrono::duration<double> diff =
            std::chrono::steady_clock::now()-begin;
        return diff.count();
    }
};

inline std::ostream&
operator<<(std::ostream& os, const timecounter& tc)
{
    os << tc.elapsed();
    return os;
}

class resmon
{
    rusage  start{};
    timeval start_time{};
    std::string label;

    static double to_seconds(const timeval& tv) {
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }

public:
    explicit resmon(std::string name = "resource monitor")
        : label(name) {
        getrusage(RUSAGE_SELF, &start);
        gettimeofday(&start_time, nullptr);
    }

    ~resmon() {
        rusage end{};
        getrusage(RUSAGE_SELF, &end);
        timeval end_time{};
        gettimeofday(&end_time, nullptr);

        long delta_maxrss = end.ru_maxrss - start.ru_maxrss;

        double user_time =
            to_seconds(end.ru_utime) - to_seconds(start.ru_utime);

        double sys_time =
            to_seconds(end.ru_stime) - to_seconds(start.ru_stime);

        double wall_time =
            to_seconds(end_time) - to_seconds(start_time);

        std::cout << "[resmon] " << label << "\n"
                  << " peak RSS         " << end.ru_maxrss << " KB\n"
                  << " delta peak RSS   " << delta_maxrss << " KB\n"
                  << " user cpu time    " << user_time << " s\n"
                  << " sys cpu time     " << sys_time << " s\n"
                  << " wall time        " << wall_time << " s\n";
    }
};