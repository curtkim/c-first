https://docs.conan.io/en/latest/creating_packages/getting_started.html


## HOWTO
   
    conan export . demo/testing
    conan install rpclib/2.2.1@demo/testing --build rpclib
    conan test test_package rpclib/2.2.1@demo/testing

    = 

    conan create . demo/testing
