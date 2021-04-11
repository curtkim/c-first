from conans import ConanFile, tools, CMake
from conans.tools import os_info, SystemPackageTool
import os
import shutil

class Pangolin(ConanFile):
    name = "pangolin"
    version = "20200520"
    url = "https://github.com/stevenlovegrove/Pangolin"
    homepage = "https://github.com/stevenlovegrove/Pangolin"
    description = "Pangolin is a lightweight portable rapid development library for managing OpenGL display / interaction and abstracting video input."
    settings = "os", "compiler", "build_type", "arch"
    exports_sources = ["CMakeLists.txt"]
    options = {
        "shared": [True, False], 
        "fPIC": [True, False]
    }
    default_options = "shared=True", "fPIC=False"

    requires = (
        "glew/2.2.0",
        "eigen/3.3.9",
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
        git.clone("https://github.com/stevenlovegrove/Pangolin.git", "master")
        git.run("reset --hard " + "86eb4975fc4fc8b5d92148c2e370045ae9bf9f5d")


    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)

        self._cmake.definitions["BUILD_EXAMPLES"] = False
        self._cmake.definitions["BUILD_TESTS"] = False
        self._cmake.definitions["BUILD_TOOLS"] = False
        self._cmake.definitions["BUILD_SHARED_LIBS"] = self.options.shared
        
        self._cmake.configure()
        return self._cmake

    def package(self):
        self.copy("LICENSE", dst="licenses", src=self._source_subfolder)
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        if str(self.settings.os) in ["Linux", "Android"]:
            self.cpp_info.libs.append('pthread')
        self.cpp_info.libs = ["pangolin"]
