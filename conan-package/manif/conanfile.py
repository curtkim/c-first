import os
import shutil
from conans import ConanFile, tools, CMake


class Manif(ConanFile):

    version = "0.0.4"

    name = "manif"
    url = "https://github.com/artivis/manif"
    homepage = "artivis.github.io/manif"
    description = "A small C++11 header-only library for Lie theory."    
    topics = ("lie group", "lie algebra")
    license = "MIT License"
    exports_sources = ["CMakeLists.txt"]
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"

    requires = (
        "eigen/3.4.0"
    )

    _cmake = None

    @property
    def _source_subfolder(self):
        return "source_subfolder"

    def source(self):
        #print(self.version)
        #print(self.conan_data)
        #print(os.getcwd())

        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)
        tools.get(**self.conan_data["sources"][self.version])
        extracted_name = self.name + '-' + self.version
        os.rename(extracted_name, self._source_subfolder)


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
