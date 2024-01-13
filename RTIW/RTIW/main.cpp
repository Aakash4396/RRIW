#include <iostream>
#include "vec3.h"
#include "point.h"
#include "color.h"
#include "ray.h"
#include "Renderer.h"

auto aspect_ratio = 16.0 / 9.0;
int width = 1260;
int height = 1;

color ray_color(const Ray& r) {

    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
}

void createImage(GLubyte* image) {

    vec3 camera_center = point(0.0f, 0.0f, 0.0f);

    auto focal_length = 1.0f;
    auto viewport_height = 2.0f;
    auto viewport_width = viewport_height * (static_cast<float>(width/height));

    auto viewport_u = vec3(viewport_width, 0.0f, 0.0f);
    auto viewport_v = vec3(0.0f, -viewport_height, 0.0f);
    
    auto pixel_delta_u = viewport_u / width;
    auto pixel_delta_v = viewport_v / height;

    auto viewport_upper_left = camera_center - vec3(0.0, 0.0, focal_length) - viewport_u / 2 - viewport_v / 2;
    auto pixel00_loc = viewport_upper_left + (0.5f) * (pixel_delta_u + pixel_delta_v);


    for (int j = 0; j < height; ++j) {
        std::clog << "\rScanlines remaining: " << (height - j) << ' ' << std::flush;
        for (int i = 0; i < width; ++i) {
            auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            auto ray_direction = pixel_center - camera_center;
            Ray r(camera_center, ray_direction);
            color pixel_color = ray_color(r);
            write_color(image, j, i, width, pixel_color);
        }
    }
    std::clog << "\rDone.                 \n";
}

int main() {

    height = static_cast<int>(width / aspect_ratio);
    height = (height < 1) ? 1 : height;

    GLubyte* imageData = new GLubyte[width*height*4];
    Renderer rdr;

    bool retVal = rdr.createWindow(L"RTIW", width, height);

    
    createImage(imageData);
    rdr.drawImage(imageData);

    if (retVal) {
        rdr.RunMessageLoop();
    }

    if (imageData) {
        delete[] imageData;
        imageData = nullptr;
    }
    return 0;
}


