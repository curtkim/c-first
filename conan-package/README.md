## case
- pipes : cmake를 최대한 이용하는, source에서 self.run을 사용
- source_subfolder를 사용하고 wrap CMakeLists.txt를 사용하는 case( 아닌 경우는 mypkg mypkg는 replace_in_file를 사용한다.)
- genertor가 cmake가 아닌 경우(pl_mpeg)
- tools.Git를 사용(libunifex), self.run을 사용
- AutoToolsBuildEnvironment사용하는 경우(freeimage)


## creating_package
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


## With Artifactory

    docker pull docker.bintray.io/jfrog/artifactory-cpp-ce
    docker run -v $(pwd)/artifactory_data:/var/opt/jfrog/artifactory -p 8081:8081 -p 8082:8082 docker.bintray.io/jfrog/artifactory-cpp-ce:latest
    
    conan remote add omega-stable http://localhost:8081/artifactory/api/conan/omega-stable
    conan upload carla-client/0.9.9.4@demo/testing -r omega-stable --all

## Artifactory Reference
- https://www.jfrog.com/confluence/display/RTF6X/Installing+with+Docker
- https://www.jfrog.com/confluence/display/JFROG/Conan+Repositories    