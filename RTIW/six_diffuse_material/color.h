#pragma once

#ifndef COLOR_H
#define COLOR_H

#include <iostream>
#include "gVec3.h"

using color = vec3;

inline double linear_to_gamma(double linear_component)
{
    return sqrt(linear_component);
}

void write_color(unsigned char* image, int col, int row, int width, color pixel_color, int samples_per_pixel) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Divide the color by the number of samples.
    auto scale = 1.0 / samples_per_pixel;
    r *= scale;
    g *= scale;
    b *= scale;

    // Apply the linear to gamma transform.
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // Write the translated [0,255] value of each color component.
    static const interval intensity(0.000, 0.999);

    //fout << int(256 * intensity.clamp(r)) << " " << int(256 * intensity.clamp(g)) << " " << int(256 * intensity.clamp(b)) << fflush;
    image[(col * width + row) * 4 + 0] = static_cast<unsigned char>(256 * intensity.clamp(r));
    image[(col * width + row) * 4 + 1] = static_cast<unsigned char>(256 * intensity.clamp(g));
    image[(col * width + row) * 4 + 2] = static_cast<unsigned char>(256 * intensity.clamp(b));
    image[(col * width + row) * 4 + 3] = 255;
}

#endif // !COLOR_H

