from conans import ConanFile, tools, CMake
from conans.tools import os_info, SystemPackageTool
import os
import shutil

class Selene(ConanFile):
    name = "selene"
    version = "20200307"
    url = "https://github.com/kmhofmann/selene"
    homepage = "https://github.com/kmhofmann/selene"
    description = "A C++17 image representation, processing and I/O library"
    settings = "os", "compiler", "build_type", "arch"
    #exports_sources = ["CMakeLists.txt"]
    options = {
        #"cuda_static_runtime": [True, False],
    }
    default_options = {
        #"cuda_static_runtime": False,
    }

    requires = (
        "libjpeg/9d", #"libjpeg-turbo/2.0.6",
        "libpng/1.6.37",
        #"libtiff/4.2.0",
    )
    generators = "cmake"
    _cmake = None

    @property
    def _source_subfolder(self):
        return "source_subfolder"


    def source(self):
        #if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
        #    shutil.rmtree(self._source_subfolder)

        self.run("git clone https://github.com/kmhofmann/selene.git selene_dir")
        self.run("cd selene_dir && git checkout 11718e1a7de6ff6251c46e4ef429a7cfb1bdb9eb")
        self.run("mv selene_dir/* .")        
        #os.rename(extracted_name, self._source_subfolder)

        tools.replace_in_file("CMakeLists.txt", "# User-settable options", '''
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()''')

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()


    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)

        self._cmake.definitions["SELENE_BUILD_TESTS"] = False
        self._cmake.definitions["SELENE_BUILD_EXAMPLES"] = False
        self._cmake.definitions["SELENE_BUILD_BENCHMARKS"] = False
        #self._cmake.definitions["CUDA_STATIC_RUNTIME"] = self.options.cuda_static_runtime
        
        self._cmake.configure()
        return self._cmake

    def package(self):
        self.copy("LICENSE", dst="licenses")
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)        
