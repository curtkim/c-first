import os
from conans import ConanFile, CMake, tools


class PclConan(ConanFile):
    name = "pcl"
    version = "1.10.1"
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "https://github.com/conan-io/conan-center-index"
    description = "<Description of Pcl here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")
    homepage = "https://github.com/curtkim/conan-package"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_cuda": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": True,
        "with_cuda": False,
    }

    generators = "cmake"
    exports_sources = ["CMakeLists.txt"]
    source_subfolder = "source_subfolder"

    def requirements(self):
        self.requires("eigen/3.3.7")
        self.requires("boost/1.72.0")
        self.requires("zlib/1.2.11")
        self.requires("flann/1.9.1")
        self.requires("libpng/1.6.37")

    def source(self):
        tools.get("https://github.com/PointCloudLibrary/pcl/archive/pcl-1.10.1.zip")
        os.rename("pcl-pcl-1.10.1", self.source_subfolder)


    def _configure_cmake(self):

        cmake = CMake(self)
        cmake.definitions["BUILD_apps"] = "OFF"
        cmake.definitions["BUILD_examples"] = "OFF"
        cmake.definitions["BUILD_common"] = "ON"
        cmake.definitions["BUILD_2d"] = "ON"
        cmake.definitions["BUILD_features"] = "ON"
        cmake.definitions["BUILD_filters"] = "ON"
        cmake.definitions["BUILD_geometry"] = "ON"
        cmake.definitions["BUILD_io"] = "ON"
        cmake.definitions["BUILD_kdtree"] = "ON"
        cmake.definitions["BUILD_octree"] = "ON"
        cmake.definitions["BUILD_sample_consensus"] = "ON"
        cmake.definitions["BUILD_search"] = "ON"
        cmake.definitions["BUILD_tools"] = "OFF"
        cmake.definitions["PCL_BUILD_WITH_BOOST_DYNAMIC_LINKING_WIN32"] = "ON"
        cmake.definitions["PCL_SHARED_LIBS"] = "OFF"
        cmake.definitions["WITH_PCAP"] = "OFF"
        cmake.definitions["WITH_DAVIDSDK"] = "OFF"
        cmake.definitions["WITH_ENSENSO"] = "OFF"
        cmake.definitions["WITH_OPENNI"] = "OFF"
        cmake.definitions["WITH_OPENNI2"] = "OFF"
        cmake.definitions["WITH_RSSDK"] = "OFF"
        cmake.definitions["WITH_QHULL"] = "OFF"
        cmake.definitions["BUILD_TESTS"] = "OFF"
        cmake.definitions["BUILD_ml"] = "ON"
        cmake.definitions["BUILD_simulation"] = "OFF"
        cmake.definitions["BUILD_segmentation"] = "ON"
        cmake.definitions["BUILD_registration"] = "ON"

        cmake.definitions["WITH_CUDA"] = "OFF"
        cmake.definitions["WITH_OPENGL"] = "OFF"
        
        cmake.configure()
        return cmake

    def build(self):
        cmake = self._configure_cmake()    
        cmake.build()

        # Explicit way:
        # self.run('cmake %s/hello %s'
        #          % (self.source_folder, cmake.command_line))
        # self.run("cmake --build . %s" % cmake.build_config)

    def package(self):
        cmake = self._configure_cmake()    
        cmake.install()

        #self.copy("*.h", dst="include", src="hello")
        #self.copy("*hello.lib", dst="lib", keep_path=False)
        #self.copy("*.dll", dst="bin", keep_path=False)
        #self.copy("*.so", dst="lib", keep_path=False)
        #self.copy("*.dylib", dst="lib", keep_path=False)
        #self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)

