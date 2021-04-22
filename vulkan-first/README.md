https://vulkan-tutorial.com/code/04_logical_device.cpp

https://vulkan.lunarg.com/doc/view/1.1.114.0/linux/getting_started_ubuntu.html

## howto
glslc shader.vert -o cmake-build-debug/vert.spv
glslc shader.frag -o cmake-build-debug/frag.spv


## 환경설정

    # driver설정 위치
    /usr/share/vulkan/icd.d/

    # sudo apt-get install libvulkan-dev
    # apt를 통해서 설치할 수 있는것은 Library 파일들과 Vulkan validation layers 용 .json manifest 파일들이다
    
    export VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d/
