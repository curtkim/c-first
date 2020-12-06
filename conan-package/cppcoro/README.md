
    conan source . --source-folder=tmp/source
    conan install . --install-folder=tmp/build --profile=gcc10
    CONAN_CPU_COUNT=10 conan build . --source-folder=tmp/source --build-folder=tmp/build --profile=gcc10
    conan package . --source-folder=tmp/source --build-folder=tmp/build --package-folder=tmp/package --profile=gcc10

    conan export-pkg . demo/testing --package-folder=tmp/package --profile=gcc10
    conan test test_package cppcoro/20201020@demo/testing --profile=gcc10

