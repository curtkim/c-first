## note
테스트에서 header파일을 찾지 못해서 실패함.

## howto

    conan source . --source-folder=tmp/source
    conan install . --install-folder=tmp/build
    cp -r tmp/source/* tmp/build/
    CONAN_CPU_COUNT=10 conan build . --source-folder=tmp/source --build-folder=tmp/build
    conan package . --source-folder=tmp/source --build-folder=tmp/build --package-folder=tmp/package

    conan export-pkg . curt/testing --package-folder=tmp/build/package
    conan test test_package Ipopt/3.12.7@curt/testing
    conan create . curt/testing
