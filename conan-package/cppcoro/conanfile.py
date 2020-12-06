import os
import shutil
from conans import ConanFile, tools, CMake


class Pipes(ConanFile):

    version = "20201020"

    name = "cppcoro"
    url = ""
    homepage = "https://github.com/Garcia6l20/cppcoro"
    description = "A library of C++ coroutine abstractions for the coroutines TS"    
    topics = ("coroutine", "async")
    license = ""
    exports_sources = ["CMakeLists.txt"]
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"

    _cmake = None

    @property
    def _source_subfolder(self):
        return "source_subfolder"

    def source(self):
        #print(self.version)
        #print(self.conan_data)
        #print(os.getcwd())

        self.run("git clone https://github.com/Garcia6l20/cppcoro.git")
        self.run("cd cppcoro && git checkout 297ee3830b63d69962695255721e91495b754da7")

        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)
        os.rename("cppcoro", self._source_subfolder)


    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)
        self._cmake.configure()
        return self._cmake

    def package(self):
        #self.copy("LICENSE", dst="licenses", src=self._source_subfolder+"/lang/c++")
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["cppcoro"]

