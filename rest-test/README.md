
	conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan
	conan search cpprestsdk -r bincrafters
	
    cd $CMAKE_BUILD
    conan install .. --build cpprestsdk
