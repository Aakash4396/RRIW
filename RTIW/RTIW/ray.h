#pragma once
#ifndef RAY_H
#define RAY_H

#include "gVec3.h"
#include "point.h"

class Ray {
private:
    point orig;
    vec3 dir;
public:
    Ray() {}
    Ray(const point& origin, vec3& direction) : orig(origin), dir(direction) {}
    point origin() const { return orig; }
    vec3 direction() const { return dir; }
    template<typename T>
    point at(T t) { return orig + t * dir; }
};

#endif // !RAY_H
