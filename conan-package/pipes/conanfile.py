import os
import shutil
from conans import ConanFile, tools, CMake


class Pipes(ConanFile):

    version = "20201012"

    name = "pipes"
    url = ""
    homepage = "https://github.com/joboccara/pipes"
    description = "Pipelines for expressive code on collections in C++"    
    topics = ("pipe", "collection")
    license = "MIT License"
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

        self.run("git clone https://github.com/joboccara/pipes")
        self.run("cd pipes && git checkout 56f7e335deb89236918f16e60526bbc0702f6b80")

        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)
        os.rename("pipes", self._source_subfolder)


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

    def package_id(self):
        self.info.header_only()
