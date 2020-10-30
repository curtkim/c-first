
    conan source . --source-folder=tmp/source
    conan install . --install-folder=tmp/build
    conan package . --source-folder=tmp/source --build-folder=tmp/build --package-folder=tmp/package

    conan export-pkg . demo/testing --package-folder=tmp/package
    conan test test_package avro/1.9.2@demo/testing
