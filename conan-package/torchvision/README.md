
    conan source . --source-folder=tmp/source
    conan install . --install-folder=tmp/build
    CONAN_CPU_COUNT=10 conan build . --source-folder=tmp/source --build-folder=tmp/build
    conan package . --source-folder=tmp/source --build-folder=tmp/build --package-folder=tmp/package

    conan export-pkg . demo/testing --package-folder=tmp/package
    conan test test_package torchvision/0.6.0@demo/testing
    conan create . demo/testing -o torchvision:with_cuda=True