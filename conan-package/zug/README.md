
    conan source . --source-folder=tmp/source
    conan install . --install-folder=tmp/build
    CONAN_CPU_COUNT=10 conan build . --source-folder=tmp/source --build-folder=tmp/build --profile=gcc10
    conan package . --source-folder=tmp/source --build-folder=tmp/build --package-folder=tmp/package

    conan export-pkg . curt/testing --package-folder=tmp/package
    conan test test_package zug/20210324@curt/testing
    conan create . curt/testing

