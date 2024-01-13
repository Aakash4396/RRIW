#include <iostream>
#include "vec3.h"
#include "point.h"
#include "color.h"
#include "Renderer.h"

#define width 512
#define height 512

void createImage(GLubyte* image) {
    for (int j = 0; j < height; ++j) {
        std::clog << "\rScanlines remaining: " << (height - j) << ' ' << std::flush;
        for (int i = 0; i < width; ++i) {
            //auto pixel_color =  color((float)((i) / (width - 1)), (float)((j) / (height - 1)), 0.0f);
            float r = float(i) / (width);
            float g = float(j) / (height);
            float b = 0.0f;
            image[(j * width + i) * 4 + 0] = static_cast<unsigned char>(255.999 * r);
            image[(j * width + i) * 4 + 1] = static_cast<unsigned char>(255.999 * g);
            image[(j * width + i) * 4 + 2] = static_cast<unsigned char>(255.999 * b);
            image[(j * width + i) * 4 + 3] = 255;
        }
    }
    std::clog << "\rDone.                 \n";
}

int main() {

    
    GLubyte* imageData = new GLubyte[width*height*4];
    Renderer rdr;

    bool retVal = rdr.createWindow(L"RTIW", width, height);

    //std::cout << "P3\n" <<  width << ' ' <<  height << "\n255\n";
    createImage(imageData);
    rdr.drawImage(imageData);

    if (retVal) {
        rdr.RunMessageLoop();
    }

    return 0;
}


