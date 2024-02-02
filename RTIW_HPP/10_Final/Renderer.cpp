// Renderer.cpp

#include "Renderer.h"

GLuint textureID;
Renderer* Renderer::currentInstance_ = nullptr;

Renderer::Renderer() {
    hInstance_ = GetModuleHandle(nullptr);
    title_ = L"My Graphics Window";
    width_ = 512;
    height_ = 512;
    fullscreen_ = false;
    currentInstance_ = this;
}

Renderer::~Renderer() {
    uninitializeOpenGL();
    if (hWnd_) {
        DestroyWindow(hWnd_);
    }
}

bool Renderer::createWindow(LPCWSTR title, int width, int height) {
    title_ = title;
    width_ = width;
    height_ = height;
    WNDCLASSEX wcex;
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wcex.lpfnWndProc = WindowProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance_;
    wcex.hIcon = LoadIcon(hInstance_, IDI_APPLICATION);
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wcex.lpszMenuName = nullptr;
    wcex.lpszClassName = TEXT("RendererClass");
    wcex.hIconSm = LoadIcon(wcex.hInstance, IDI_APPLICATION);

    if (!RegisterClassEx(&wcex)) {
        MessageBox(nullptr, TEXT("Failed to register window class"), TEXT("Error"), MB_OK | MB_ICONERROR);
        return false;
    }

    hWnd_ = CreateWindowExW(
        WS_EX_APPWINDOW,
        L"RendererClass",
        title_,
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
        (GetSystemMetrics(SM_CXSCREEN) - width_) / 2,
        (GetSystemMetrics(SM_CYSCREEN) - height_) / 2,
        width_,
        height_,
        nullptr,
        nullptr,
        hInstance_,
        nullptr
    );

    int retVal = initializeOpenGL();

    return hWnd_ != nullptr;
}

void Renderer::RunMessageLoop() {
    MSG msg;
    while (true) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT)
                break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else {
            Render();
        }
    }
}

LRESULT CALLBACK Renderer::WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    case WM_CHAR:
        switch (wParam) {
        case 'F':
        case 'f':
            // 'F' or 'f' key pressed
            currentInstance_->ToggleFullscreen();
            break;
        }
        break;
    case WM_KEYDOWN:
        switch (wParam) {
        case 27:
            DestroyWindow(currentInstance_->hWnd_);
            break;
        }
        break;
    case WM_SIZE:
        currentInstance_->resize(LOWORD(lParam), HIWORD(lParam));
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

void Renderer::ToggleFullscreen() {
    // variable declarations
    static DWORD dwStyle;
    static WINDOWPLACEMENT wp;
    MONITORINFO mi;

    // code
    wp.length = sizeof(WINDOWPLACEMENT);
    if (fullscreen_ == FALSE) {
        dwStyle = GetWindowLong(hWnd_, GWL_STYLE);
        if (dwStyle & WS_OVERLAPPEDWINDOW) {
            mi.cbSize = sizeof(MONITORINFO);
            if (GetWindowPlacement(hWnd_, &wp) && GetMonitorInfo(MonitorFromWindow(hWnd_, MONITORINFOF_PRIMARY), &mi)) {
                SetWindowLong(hWnd_, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
                SetWindowPos(hWnd_, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
            }
            ShowCursor(FALSE);
        }
    }
    else {
        SetWindowLong(hWnd_, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(hWnd_, &wp);
        SetWindowPos(hWnd_, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
        ShowCursor(TRUE);
    }

    fullscreen_ = !fullscreen_;
}

int Renderer::initializeOpenGL() {
    // Function declarations

    // Variable declarations
    PIXELFORMATDESCRIPTOR pfd;
    int iPixelFormatIndex = 0;

    // Code
    ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));
    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cRedBits = 8;
    pfd.cGreenBits = 8;
    pfd.cBlueBits = 8;
    pfd.cAlphaBits = 8;

    // GetDC
    hdc_ = GetDC(hWnd_);

    // choose pixel format
    iPixelFormatIndex = ChoosePixelFormat(hdc_, &pfd);
    if (iPixelFormatIndex == 0) {
        return -1;
    }

    // set choosen pixel format
    if (SetPixelFormat(hdc_, iPixelFormatIndex, &pfd) == FALSE) {
        return -2;
    }

    // Create OpenGL Rendering Context
    hrc_ = wglCreateContext(hdc_);
    if (hrc_ == NULL) {
        return -3;
    }

    // Make Rendering Context as Current Context
    if (wglMakeCurrent(hdc_, hrc_) == FALSE) {
        return -4;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    resize(width_, height_);
    return 0;
}

void Renderer::resize(int width, int height) {
    // Code
    if (height == 0)
        height = 1;	// to avoid divide by zero in future call

    glViewport(0, 0, (GLsizei)width, (GLsizei)height);
}


void Renderer::Render() {

    // Implement your rendering logic here
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    
    SwapBuffers(hdc_);
}


void Renderer::drawImage(unsigned char* image) {

    glGenTextures(1, &textureID);	// gen texture and get id in a variable
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
    //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, 0);
}



void Renderer::uninitializeOpenGL() {
    if (wglGetCurrentContext() == hrc_) {
        wglMakeCurrent(NULL, NULL);
    }

    if (hrc_) {
        wglDeleteContext(hrc_);
        hrc_ = NULL;
    }

    if (hdc_) {
        ReleaseDC(hWnd_, hdc_);
        hdc_ = NULL;
    }
}
