import os
import shutil
from conans import ConanFile, tools, CMake


class Torchvision(ConanFile):

    ## for test
    version = "0.7.0"

    name = "torchvision"
    url = ""
    homepage = "https://github.com/pytorch/vision"
    description = "Datasets, Transforms and Models specific to Computer Vision"
    topics = ("conan", "vision", "torch")
    license = "BSD 3-Clause License"
    #exports_sources = ["CMakeLists.txt", "patches/*"]
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "fPIC": [True, False],
        "with_cuda": [True, False],
    }
    default_options = {
        "fPIC": True,
        "with_cuda": False,
    }
    generators = "cmake"

    requires = (
        "torch/1.6.0@curt/prebuilt",
    )

    _cmake = None

    @property
    def _source_subfolder(self):
        return "source_subfolder"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def source(self):
        #print(self.version)
        #print(self.conan_data)
        #print(os.getcwd())
        tools.get(**self.conan_data["sources"][self.version])
        extracted_name = "vision-" + self.version

        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)
        os.rename(extracted_name, self._source_subfolder)

        tools.replace_in_file("source_subfolder/CMakeLists.txt", 
'''project(torchvision)''', 
'''project(torchvision)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()''')

        # patch 
        #for patch in self.conan_data["patches"][self.version]:
        #    tools.patch(**patch)

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)
        #self._cmake.definitions["BUILD_TESTING"] = False
        self._cmake.configure(source_folder=self._source_subfolder)
        return self._cmake

    def package(self):
        self.copy("LICENSE", dst="licenses", src=self._source_subfolder)
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = self.collect_libs()
