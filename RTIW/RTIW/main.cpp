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

    world.add(make_shared<sphere>(point(0.0f, 0.0f, -1.0f), 0.5f));
    world.add(make_shared<sphere>(point(0.0f, -100.5f, -1.0f), 100.0f));

    auto aspect_ratio = 16.0 / 9.0;
    int width = 800;
    int height = static_cast<int>(width / aspect_ratio);

    camera cam(width, height);

    Renderer rdr;
    bool retVal = rdr.createWindow(L"RTIW", width, height);
    
    unsigned char* imageData = cam.createImage(world);

    if (retVal) {
        rdr.drawImage(imageData);
        rdr.RunMessageLoop();
    }

    fpclose;
    return 0;
}


