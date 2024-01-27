Rem Run on x64 command prompt

nvcc -c -o kernel.obj kernel.cu

cl.exe /c /EHsc -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include" main.cpp Renderer.cpp

link main.obj Renderer.obj kernel.obj "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64\cudart.lib" user32.lib gdi32.lib

