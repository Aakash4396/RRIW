#include <iostream>

#include "Renderer.h"

#include "FileWriterUtility.h"
#include "rtiw.h"
#include "camera.h"
#include "hittable_list.h"
#include "sphere.h"


int main() {

    // World

    hittable_list world;
    
    world.add(new sphere(point(-1.0f, 0.0f, -1.0f), 0.5f));
    world.add(new sphere(point(1.0f, 0.0f, -1.0f), 0.5f));
    world.add(new sphere(point(0.0f, -100.5f, -1.0f), 100.0f));

    auto aspect_ratio = 16.0 / 9.0;
    int width = 800;
    int height = static_cast<int>(width / aspect_ratio);

    camera cam(width, height);

    cam.samples_per_pixel = 100;

    unsigned char* imageData = cam.createImage(world);

    Renderer rdr;
    bool retVal = rdr.createWindow(L"RTIW", width, height);

    if (retVal) {
        rdr.drawImage(imageData);
        rdr.RunMessageLoop();
    }

    fpclose;
    return 0;
}


