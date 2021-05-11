from conans import ConanFile, tools, CMake
from conans.tools import os_info, SystemPackageTool
import os
import shutil

class Stdgpu(ConanFile):
    name = "stdgpu"
    version = "1.3.0"
    url = "https://github.com/stotko/stdgpu"
    homepage = "https://github.com/stotko/stdgpu"
    description = "Efficient STL-like Data Structures on the GPU"
    settings = "os", "compiler", "build_type", "arch"
    exports_sources = ["CMakeLists.txt"]
    options = {
        "shared": [True, False], 
        "fPIC": [True, False]
    }
    default_options = "shared=True", "fPIC=True"

    requires = (
    )
    generators = "cmake"
    _cmake = None

    @property
    def _source_subfolder(self):
        return "source_subfolder"


    def configure(self):
        if self.settings.os == "Windows":
            self.options.remove("fPIC")

    def source(self):
        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)

        tools.get(**self.conan_data["sources"][self.version])
        extracted_name = self.name + '-' + self.version
        os.rename(extracted_name, self._source_subfolder)


    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)

        self._cmake.definitions["STDGPU_BUILD_SHARED_LIBS"] = self.options.shared
        self._cmake.definitions["STDGPU_BUILD_EXAMPLES"] = False
        self._cmake.definitions["STDGPU_BUILD_TESTS"] = False


        self._cmake.configure()
        return self._cmake

    def package(self):
        self.copy("LICENSE", dst="licenses", src=self._source_subfolder)
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["stdgpu"]
