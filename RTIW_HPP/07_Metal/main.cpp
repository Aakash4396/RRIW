#include <iostream>
#include "kernel.cuh"
#include "Renderer.h"


int main() {

    auto aspect_ratio = 16.0 / 9.0;
    int width = IMG_WIDTH;
    int height = static_cast<int>(width / aspect_ratio);

    unsigned char* imageData = new unsigned char[width*height*4];
    

    CudaWrapper::cudaMain(imageData, width, height);

    Renderer rdr;
    bool retVal = rdr.createWindow(L"RTIW", width, height);
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


