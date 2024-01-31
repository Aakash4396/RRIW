#pragma once
#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "PStash.h"
#include "sphere.h"


class hittable_list {
public:
    PStash<hittable> objects;

    hittable_list() {}
    hittable_list(hittable* object) { add(object); }
    
    void add(hittable* object) {
        objects.add(object);
    }

    bool hit(const Ray& r, interval ray_t, hit_record& rec) {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.imax;
        
        for (auto object : objects) {
            if (object->hit(r, interval(ray_t.imin, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

};


#endif // !HITTABLE_LIST_H
