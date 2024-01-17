#pragma once
#ifndef INTERVAL_H
#define INTERVAL_H

class interval {
public:
    float imin, imax;

    interval() : imin(+infinity), imax(-infinity) {} // Default interval is empty

    interval(float _min, float _max) : imin(_min), imax(_max) {}

    bool contains(double x) const {
        return imin <= x && x <= imax;
    }

    bool surrounds(double x) const {
        return imin < x && x < imax;
    }

    static const interval empty, universe;
};

const static interval empty(+infinity, -infinity);
const static interval universe(-infinity, +infinity);

#endif

