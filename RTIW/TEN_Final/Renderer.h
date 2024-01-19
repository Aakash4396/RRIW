// Renderer.h

#pragma once

#ifndef RENDERER_H
#define RENDERER_H

#include <Windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>


// OpenGL macros
#pragma comment(lib,"OpenGL32.lib")
#pragma comment(lib,"glu32.lib")

class Renderer {
public:

    Renderer();
    ~Renderer();

    bool createWindow(LPCWSTR title, int width, int height);
    void Render();  // Add your rendering logic here
    void RunMessageLoop();
    void drawImage(unsigned char*);

private:
    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
    void ToggleFullscreen();
    int initializeOpenGL();
    void uninitializeOpenGL();
    void resize(int, int);
    HINSTANCE hInstance_;
    HWND hWnd_;
    LPCWSTR title_;
    int width_;
    int height_;
    HDC hdc_;
    HGLRC hrc_;
    bool fullscreen_;

    static Renderer* currentInstance_;
};

#endif // RENDERER_H
