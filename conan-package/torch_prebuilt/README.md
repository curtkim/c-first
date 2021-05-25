## HOWTO

    conan create . 1.7.1@curt/prebuilt -o torch:cuda=11.0
    conan create . 1.8.1@curt/prebuilt



    conan source . --source-folder=tmp/source
    conan install . --install-folder=tmp/build
    CONAN_CPU_COUNT=10 conan build . --source-folder=tmp/source --build-folder=tmp/build
    conan package . --source-folder=tmp/source --build-folder=tmp/build --package-folder=tmp/package

    conan export-pkg . curt/prebuilt --package-folder=tmp/package
    conan test test_package torch/1.6.0@curt/prebuilt


## Debug

    torch cxx11을 받았기 때문에 profile에 아래가 적용되어 있어야한다. 
    
    compiler.libcxx=libstdc++11