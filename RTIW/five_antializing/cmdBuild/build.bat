cl /c /EHsc ../Renderer.cpp ../main.cpp

link main.obj Renderer.obj user32.lib gdi32.lib ole32.lib
