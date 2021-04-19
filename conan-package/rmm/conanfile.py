from conans import ConanFile, tools, CMake
from conans.tools import os_info, SystemPackageTool
import os
import shutil

class Rmm(ConanFile):
    name = "rmm"
    version = "0.20.0a"
    url = "https://github.com/rapidsai/rmm"
    homepage = "https://github.com/rapidsai/rmm"
    description = "RAPIDS Memory Manager"
    settings = "os", "compiler", "build_type", "arch"
    exports_sources = ["CMakeLists.txt"]
    options = {
        "cuda_static_runtime": [True, False],
    }
    default_options = {
        "cuda_static_runtime": False,
    }

    requires = (
        "spdlog/1.8.5",
        "thrust/1.9.5",
    )
    generators = "cmake"
    _cmake = None

    @property
    def _source_subfolder(self):
        return "source_subfolder"


    def source(self):
        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)

        tools.get(**self.conan_data["sources"][self.version])
        extracted_name = self.name + '-' + self.version
        os.rename(extracted_name, self._source_subfolder)

        tools.replace_in_file(self._source_subfolder + "/CMakeLists.txt", "include(cmake/Modules/CPM.cmake)",'')
        tools.replace_in_file(self._source_subfolder + "/CMakeLists.txt", "include(cmake/Modules/RMM_thirdparty.cmake)",'')

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()


    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)

        self._cmake.definitions["BUILD_TESTS"] = False
        self._cmake.definitions["BUILD_BENCHMARKS"] = False
        self._cmake.definitions["CUDA_STATIC_RUNTIME"] = self.options.cuda_static_runtime
        
        self._cmake.configure()
        return self._cmake

    def package(self):
        self.copy("LICENSE", dst="licenses", src=self._source_subfolder)
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
        self.cpp_info.libs.append('CUDA::cudart_static' if self.options.cuda_static_runtime else 'CUDA::cudart')
