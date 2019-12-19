from conans import ConanFile, CMake, tools


class GeosConan(ConanFile):
    name = "geos"
    version = "3.8"
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of Geos here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = {"shared": False}
    generators = "cmake"

    def source(self):
        self.run("git clone --branch 3.8.0 https://github.com/libgeos/geos.git")
        # This small hack might be useful to guarantee proper /MT /MD linkage
        # in MSVC if the packaged project doesn't have variables to set it
        # properly
        
        tools.replace_in_file("geos/CMakeLists.txt", 
        '''project(GEOS VERSION "${_version_major}.${_version_minor}.${_version_patch}"
  LANGUAGES C CXX
  ${_project_info})''', 
        '''project(GEOS VERSION "${_version_major}.${_version_minor}.${_version_patch}"
  LANGUAGES C CXX
  ${_project_info})
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()''')

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_folder="geos")
        cmake.build()

        # Explicit way:
        # self.run('cmake %s/hello %s'
        #          % (self.source_folder, cmake.command_line))
        # self.run("cmake --build . %s" % cmake.build_config)

    def package(self):
        self.copy("*.h", src="include/geos", dst="include", keep_path=True) # from build
        self.copy("*.h", src="geos/include", dst="include", keep_path=True) # from source
        #self.copy("*.dll", dst="bin", keep_path=False)
        #self.copy("*.so", dst="lib", keep_path=False)
        #self.copy("*.dylib", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["geos"]

