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
            auto pixel_color =  color(float(i) / (width - 1), float(j) / (height - 1), 0.0f);
            write_color(image, j, i, width, pixel_color);
        }
    }
    std::clog << "\rDone.                 \n";
}

int main() {

    
    GLubyte* imageData = new GLubyte[width*height*4];
    Renderer rdr;

    bool retVal = rdr.createWindow(L"RTIW", width, height);

    
    createImage(imageData);
    rdr.drawImage(imageData);

    if (retVal) {
        rdr.RunMessageLoop();
    }

    return 0;
}


