## compile & link 

    /usr/bin/g++-7 -g  CMakeFiles/scene.dir/scene.cpp.o -o bin/scene   
    -L/home/curt/.conan/data/pangolin/20200520/curt/testing/package/c7c7564d1d413e9a2ee043b49887c4a9fe665af7/lib  
    -L/home/curt/.conan/data/glew/2.2.0/_/_/package/2236b2cb703ee4f00965194fb773deafecabac53/lib  
    -Wl,
    -rpath,/home/curt/.conan/data/pangolin/20200520/curt/testing/package/c7c7564d1d413e9a2ee043b49887c4a9fe665af7/lib
    :/home/curt/.conan/data/glew/2.2.0/_/_/package/2236b2cb703ee4f00965194fb773deafecabac53/lib 
    -lpangolin 
    -lGLEW 
    -lm 
    -lGLU 
    -lGL 
    -lX11
