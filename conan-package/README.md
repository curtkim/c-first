https://docs.conan.io/en/latest/creating_packages/getting_started.html


## HOWTO
   
    conan export . demo/testing
    conan install rpclib/2.2.1@demo/testing --build rpclib
    conan test test_package rpclib/2.2.1@demo/testing

    = 

    conan create . demo/testing


## pcl local test

    conan source --source-folder=tmp/source .
    conan build . --source-folder=tmp/source --build-folder=tmp/build
    conan package . --source-folder=tmp/source --build-folder=tmp/build --package-folder=tmp/package


    conan create . demo/testing
    conan test test_package pcl/1.10.1@demo/testing