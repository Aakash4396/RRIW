#include <iostream>

#include "Renderer.h"

#include "FileWriterUtility.h"
#include "rtiw.h"
#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"


int main() {

    // World

    hittable_list world;

    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left = make_shared<dielectric>(1.5);
    auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);


    world.add(make_shared<sphere>(point(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<sphere>(point(0.0, 0.0, -1.0), 0.5, material_center));
    world.add(make_shared<sphere>(point(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(make_shared<sphere>(point(-1.0, 0.0, -1.0), -0.4, material_left));
    world.add(make_shared<sphere>(point(1.0, 0.0, -1.0), 0.5, material_right));

    auto aspect_ratio = 16.0 / 9.0;
    int width = 800;
    int height = static_cast<int>(width / aspect_ratio);

    camera cam(width, height);

    cam.samples_per_pixel = 100;
    cam.max_depth = 50;

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


