import os
from conans import ConanFile, tools, CMake
import shutil

class Cutlass(ConanFile):
    name = "cutlass"
    version = "2.8.0"
    url = "https://github.com/NVIDIA/cutlass"
    homepage = "https://github.com/NVIDIA/cutlass"
    description = "CUDA Templates for Linear Algebra Subroutines"
    topics = ("conan", "cuda", "template", "gemm")
    settings = "os"
    license = "BSD3 License"
    exports_sources = ["CMakeLists.txt"]

    _cmake = None
    
    _source_subfolder = "source_subfolder"

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

        self._cmake.definitions["CUTLASS_ENABLE_EXAMPLES_INIT"] = False
        self._cmake.definitions["CUTLASS_ENABLE_HEADERS_ONLY"] = True
        self._cmake.definitions["CUTLASS_ENABLE_EXAMPLES"] = False
        self._cmake.definitions["CUTLASS_ENABLE_TOOLS"] = True

        self._cmake.configure()
        return self._cmake

    def package(self):
        self.copy("LICENSE", dst="licenses", src=self._source_subfolder)
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["cutlass"]

