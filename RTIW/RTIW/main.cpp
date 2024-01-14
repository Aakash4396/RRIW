#include <iostream>
#include "FileWriterUtility.h"
#include "vec3.h"
#include "point.h"
#include "color.h"
#include "ray.h"
#include "Renderer.h"


auto aspect_ratio = 16.0 / 9.0;
int width = 800;
int height = 1;


float hit_sphere(const point& center, float radius, const Ray& r) {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;
    
    if (discriminant < 0) {
        return -1.0f;
    } else {
        return (-half_b - sqrt(discriminant)) / a;
    }
}


color ray_color(const Ray& r) {
    auto t = hit_sphere(point(0.0f, 0.0f, -1.0f), 0.5f, r);
    if (t > 0.0f) {
        vec3 N = unit_vector(r.at(t) - vec3(0.0f, 0.0f, -1.0f));
        return 0.5f * color(N.x() + 1, N.y() + 1, N.z() + 1);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
}

void createImage(unsigned char* image) {

    vec3 camera_center = point(0.0f, 0.0f, 0.0f);

    auto focal_length = 1.0f;
    float viewport_height = 2.0f;
    float viewport_width = viewport_height * ((float)width / (float)height);

    auto viewport_u = vec3(viewport_width, 0.0f, 0.0f);
    auto viewport_v = vec3(0.0f, -viewport_height, 0.0f);

    auto pixel_delta_u = viewport_u / width;
    auto pixel_delta_v = viewport_v / height;

    auto viewport_upper_left = camera_center - vec3(0.0f, 0.0f, focal_length) - viewport_u / 2.0f - viewport_v / 2.0f;

    auto pixel00_loc = viewport_upper_left + (0.5f) * (pixel_delta_u + pixel_delta_v);

    //fout << "P3\n" << width << " " << height << "\n255\n";

    for (int j = 0; j < height; ++j) {
        std::clog << "\rScanlines remaining: " << (height - j) << ' ' << std::flush;
        for (int i = 0; i < width; ++i) {
            auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            auto ray_direction = pixel_center - camera_center;
            Ray r(camera_center, ray_direction);
            color pixel_color = ray_color(r);
            //fout << int(255.99 * pixel_color.x()) << " " << int(255.99 * pixel_color.y()) << " " << int(255.99 * pixel_color.z()) << fflush;
            write_color(image, j, i, width, pixel_color);
        }
    }
    std::clog << "\rDone.                 \n";
}

int main() {

    height = static_cast<int>(width / aspect_ratio);
    height = (height < 1) ? 1 : height;

    unsigned char* imageData = new unsigned char[width * height * 4];
    Renderer rdr;

    createImage(imageData);

    bool retVal = rdr.createWindow(L"RTIW", width, height);
    rdr.drawImage(imageData);

    if (retVal) {
        rdr.RunMessageLoop();
    }

    if (imageData) {
        delete[] imageData;
        imageData = nullptr;
    }
    fpclose;
    return 0;
}


