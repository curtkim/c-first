
    conan source . --source-folder=tmp/source
    conan install . --install-folder=tmp/build
    conan package . --source-folder=tmp/source --build-folder=tmp/build --package-folder=tmp/package

    conan export-pkg . curt/testing --package-folder=tmp/package
    conan test test_package cutlass/2.8.0@curt/testing
    conan create . curt/testing

