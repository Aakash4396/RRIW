#pragma once

#ifndef COLOR_H
#define COLOR_H

#include <iostream>
#include "gVec3.h"

using color = vec3;

void write_color(unsigned char* image, int col, int row, int width, color pxl) {
    image[(col * width + row) * 4 + 0] = static_cast<unsigned char>(255.999 * pxl.x());
    image[(col * width + row) * 4 + 1] = static_cast<unsigned char>(255.999 * pxl.y());
    image[(col * width + row) * 4 + 2] = static_cast<unsigned char>(255.999 * pxl.z());
    image[(col * width + row) * 4 + 3] = 255;
    //fout << (int)image[(col * width + row) * 4 + 0] << " " << (int)image[(col * width + row) * 4 + 1] << " " << (int)image[(col * width + row) * 4 + 2] << fflush;
}

#endif // !COLOR_H

