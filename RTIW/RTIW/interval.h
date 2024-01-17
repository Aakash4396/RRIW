#pragma once
#ifndef INTERVAL_H
#define INTERVAL_H

class interval {
public:
    double imin, imax;

    interval() : imin(+infinity), imax(-infinity) {} // Default interval is empty

    interval(double _min, double _max) : imin(_min), imax(_max) {}

    bool contains(double x) const {
        return imin <= x && x <= imax;
    }

    bool surrounds(double x) const {
        return imin < x && x < imax;
    }

    double clamp(double x) const {
        if (x < imin) return imin;
        if (x > imax) return imax;
        return x;
    }

    static const interval empty, universe;
};

const static interval empty(+infinity, -infinity);
const static interval universe(-infinity, +infinity);

#endif

