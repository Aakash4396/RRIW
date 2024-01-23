#pragma once
#ifndef INTERVAL_H
#define INTERVAL_H

class interval {
public:
    double infinity = 10000000.0;
    double imin, imax;

    __host__ __device__ interval() : imin(+infinity), imax(-infinity) {} // Default interval is empty

    __host__ __device__ interval(double _min, double _max) : imin(_min), imax(_max) {}

    __host__ __device__ bool contains(double x) const {
        return imin <= x && x <= imax;
    }

    __host__ __device__ bool surrounds(double x) const {
        return imin < x && x < imax;
    }

    __host__ __device__ double clamp(double x) const {
        if (x < imin) return imin;
        if (x > imax) return imax;
        return x;
    }
};

#endif

