
    conan source . --source-folder=tmp/source
    conan install . --install-folder=tmp/build -o torchvision:with_cuda=True
    CONAN_CPU_COUNT=20 conan build . --source-folder=tmp/source --build-folder=tmp/build
    conan package . --source-folder=tmp/source --build-folder=tmp/build --package-folder=tmp/package

    conan export-pkg . demo/testing --package-folder=tmp/package -o torchvision:with_cuda=True
    conan test test_package torchvision/0.9.1@curt/testing -o torchvision:with_cuda=True -o torch:cuda=11.1


    conan create . curt/testing -o torchvision:with_cuda=True -o torch:cuda=11.1
