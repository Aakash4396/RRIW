#pragma once
#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>


class vec3 {
private:
    double e[3];
public:
    __host__ __device__ vec3();
    __host__ __device__ vec3(double e0, double e1, double e2);
    __host__ __device__ vec3(const vec3& v);

    __host__ __device__ double x() const;
    __host__ __device__ double y() const;
    __host__ __device__ double z() const;
    
    __host__ __device__ vec3 operator-() const;
    __host__ __device__ double operator[](int i) const;
    __host__ __device__ double& operator[](int i);

    
    __host__ __device__ vec3& operator+=(const vec3&);
    __host__ __device__ vec3& operator-=(const vec3&);
    __host__ __device__ vec3& operator*=(const vec3&);
    __host__ __device__ vec3& operator/=(const vec3&);
    __host__ __device__ vec3& operator*=(const double t);
    __host__ __device__ vec3& operator/=(const double t);
    template <class _CharT, class _Traits>
    __host__ __device__ friend std::basic_ostream<_CharT, _Traits>& operator<<(std::basic_ostream<_CharT, _Traits>&, const vec3&);
    __host__ __device__ double length() const;
    __host__ __device__ double length_squared() const;
    __host__ __device__ bool near_zero() const;
};

#endif // VEC3_H



__host__ __device__ vec3::vec3() {
    e[0] = 0;
    e[1] = 0;
    e[2] = 0;
}

__host__ __device__ vec3::vec3(double e0, double e1, double e2) {
    e[0] = e0;
    e[1] = e1;
    e[2] = e2;
}


__host__ __device__ vec3::vec3(const vec3& v) {
    e[0] = v.x();
    e[1] = v.y();
    e[2] = v.z();
}


__host__ __device__ double vec3::x() const {
    return e[0];
}


__host__ __device__ double vec3::y() const {
    return e[1];
}


__host__ __device__ double vec3::z() const {
    return e[2];
}


__host__ __device__ vec3 vec3::operator-() const {
    return vec3(-x(), -y(), -z());
}


__host__ __device__ double vec3::operator[](int i) const {
    return e[i];
}


__host__ __device__ double& vec3::operator[](int i) {
    return e[i];
}


__host__ __device__ vec3& vec3::operator+=(const vec3& v) {
    e[0] += v.x();
    e[1] += v.y();
    e[2] += v.z();
    return *this;
}


__host__ __device__ vec3& vec3::operator-=(const vec3& v) {
    e[0] -= v.x();
    e[1] -= v.y();
    e[2] -= v.z();
    return *this;
}


__host__ __device__ vec3& vec3::operator*=(const vec3& v) {
    e[0] *= v.x();
    e[1] *= v.y();
    e[2] *= v.z();
    return *this;
}


__host__ __device__ vec3& vec3::operator/=(const vec3& v) {
    e[0] /= v.x();
    e[1] /= v.y();
    e[2] /= v.z();
    return *this;
}


__host__ __device__ vec3& vec3::operator*=(const double t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}


__host__ __device__ vec3& vec3::operator/=(const double t) {
    e[0] /= t;
    e[1] /= t;
    e[2] /= t;
    return *this;
}


__host__ __device__ vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}


__host__ __device__ vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}


__host__ __device__ vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

__host__ __device__ vec3 operator/(const vec3& u, const vec3& v) {
    return vec3(u.x() / v.x(), u.y() / v.y(), u.z() / v.z());
}

template <typename U>
__host__ __device__ vec3 operator*(U t, const vec3& v) {
    return vec3(t * v.x(), t * v.y(), t * v.z());
}

template <typename U>
__host__ __device__ vec3 operator*(const vec3& v, const U t) {
    return t * v;
}

template <typename U>
__host__ __device__ vec3 operator/(const vec3& v, const U t) {
    return ((double)1 / t) * v;
}

__host__ __device__ double dot(const vec3& u, const vec3& v) {
    return u.x() * v.x()
        + u.y() * v.y()
        + u.z() * v.z();
}


__host__ __device__ vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.y() * v.z() - u.z() * v.y(),
        u.z() * v.x() - u.x() * v.z(),
        u.x() * v.y() - u.y() * v.x());
}


__host__ __device__ vec3 unit_vector(vec3 v) {
    return v / v.length();
}

__host__ __device__ double vec3::length() const {
    return sqrt(length_squared());
}


__host__ __device__ bool vec3::near_zero() const {
    // Return true if the vector is close to zero in all dimensions.
    auto s = 1e-8;
    return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
}


__host__ __device__ double vec3::length_squared() const {
    return x() * x() + y() * y() + z() * z();
}

template <class _CharT, class _Traits>
__host__ __device__ std::basic_ostream<_CharT, _Traits>& operator<<(std::basic_ostream<_CharT, _Traits>& out, const vec3& v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}


