#pragma once
#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

template <typename T>
class vec3 {
private:
    T e[3];
public:
    vec3();
    vec3(T e0, T e1, T e2);
    vec3(const vec3& v);

    T x() const;
    T y() const;
    T z() const;
    
    vec3<T> operator-() const;
    T operator[](int i) const;
    T& operator[](int i);

    
    vec3<T>& operator+=(const vec3<T>&);
    vec3<T>& operator-=(const vec3<T>&);
    vec3<T>& operator*=(const vec3<T>&);
    vec3<T>& operator/=(const vec3<T>&);
    vec3<T>& operator*=(const T t);
    vec3<T>& operator/=(const T t);
    template <typename U>
    friend std::ostream& operator<<(std::ostream&, const vec3<U>&);
    T length() const;
    T length_squared() const;
};

#endif // VEC3_H


template <typename T>
vec3<T>::vec3() {
    e[0] = 0;
    e[1] = 0;
    e[2] = 0;
}

template <typename T>
vec3<T>::vec3(T e0, T e1, T e2) {
    e[0] = e0;
    e[1] = e1;
    e[2] = e2;
}

template <typename T>
vec3<T>::vec3(const vec3<T>& v) {
    e[0] = v.x();
    e[1] = v.y();
    e[2] = v.z();
}

template <typename T>
T vec3<T>::x() const {
    return e[0];
}

template <typename T>
T vec3<T>::y() const {
    return e[1];
}

template <typename T>
T vec3<T>::z() const {
    return e[2];
}

template <typename T>
vec3<T> vec3<T>::operator-() const {
    return vec3<T>(-x(), -y(), -z());
}

template <typename T>
T vec3<T>::operator[](int i) const {
    return e[i];
}

template <typename T>
T& vec3<T>::operator[](int i) {
    return e[i];
}

template <typename T>
vec3<T>& vec3<T>::operator+=(const vec3<T>& v) {
    e[0] += v.x();
    e[1] += v.y();
    e[2] += v.z();
    return *this;
}

template <typename T>
vec3<T>& vec3<T>::operator-=(const vec3<T>& v) {
    e[0] -= v.x();
    e[1] -= v.y();
    e[2] -= v.z();
    return *this;
}

template <typename T>
vec3<T>& vec3<T>::operator*=(const vec3<T>& v) {
    e[0] *= v.x();
    e[1] *= v.y();
    e[2] *= v.z();
    return *this;
}

template <typename T>
vec3<T>& vec3<T>::operator/=(const vec3<T>& v) {
    e[0] /= v.x();
    e[1] /= v.y();
    e[2] /= v.z();
    return *this;
}

template <typename T>
vec3<T>& vec3<T>::operator*=(const T t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

template <typename T>
vec3<T>& vec3<T>::operator/=(const T t) {
    e[0] /= t;
    e[1] /= t;
    e[2] /= t;
    return *this;
}

template <typename T>
vec3<T> operator+(const vec3<T>& u, const vec3<T>& v) {
    return vec3<T>(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

template <typename T>
vec3<T> operator-(const vec3<T>& u, const vec3<T>& v) {
    return vec3<T>(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

template <typename T>
vec3<T> operator*(const vec3<T>& u, const vec3<T>& v) {
    return vec3<T>(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

template <typename T>
vec3<T> operator/(const vec3<T>& u, const vec3<T>& v) {
    return vec3<T>(u.x() / v.x(), u.y() / v.y(), u.z() / v.z());
}

template <typename T, typename U>
vec3<T> operator*(U t, const vec3<T>& v) {
    return vec3<T>(t * v.x(), t * v.y(), t * v.z());
}

template <typename T, typename U>
vec3<T> operator*(const vec3<T>& v, const U t) {
    return t * v;
}

template <typename T, typename U>
vec3<T> operator/(const vec3<T>& v, const U t) {
    return (1 / t) * v;
}

template <typename T>
T dot(const vec3<T>& u, const vec3<T>& v) {
    return u.x() * v.x()
        + u.y() * v.y()
        + u.z() * v.z();
}

template <typename T>
vec3<T> cross(const vec3<T>& u, const vec3<T>& v) {
    return vec3<T>(u.y() * v.z() - u.z() * v.y(),
        u.z() * v.x() - u.x() * v.z(),
        u.x() * v.y() - u.y() * v.x());
}

template <typename T>
vec3<T> unit_vector(vec3<T> v) {
    return v / v.length();
}

template <typename T>
T vec3<T>::length() const {
    return sqrt(length_squared());
}

template <typename T>
T vec3<T>::length_squared() const {
    return x() * x() + y() * y() + z() * z();
}


template <typename T>
std::ostream& operator<<(std::ostream& out, const vec3<T>& v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}
