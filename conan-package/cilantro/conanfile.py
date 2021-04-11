from conans import ConanFile, tools, CMake
from conans.tools import os_info, SystemPackageTool
import os
import shutil

class Cilantro(ConanFile):
    name = "cilantro"
    version = "20210308"
    url = "https://github.com/kzampog/cilantro"
    homepage = "https://github.com/kzampog/cilantro"
    description = "A lean C++ library for working with point cloud data"
    settings = "os", "compiler", "build_type", "arch"
    exports_sources = ["CMakeLists.txt"]
    options = {
        "shared": [True, False], 
        "fPIC": [True, False]
    }
    default_options = "shared=True", "fPIC=False"

    requires = (
        "eigen/3.3.9",
        "pangolin/20200520@curt/testing"
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

        git = tools.Git(folder=self._source_subfolder)
        git.clone("https://github.com/kzampog/cilantro.git", "master")
        git.run("reset --hard " + "c92b0d784f96e2f1ebb54101ad9e5137b123a61f")


    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)

        #self._cmake.definitions["BUILD_EXAMPLES"] = False
        #self._cmake.definitions["BUILD_TESTS"] = False
        #self._cmake.definitions["BUILD_TOOLS"] = False
        #self._cmake.definitions["BUILD_SHARED_LIBS"] = self.options.shared
        
        self._cmake.configure()
        return self._cmake

    def package(self):
        self.copy("LICENSE", dst="licenses", src=self._source_subfolder)
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["cilantro"]
