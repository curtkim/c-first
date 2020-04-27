import os
import shutil
from conans import ConanFile, tools, CMake


class Avro(ConanFile):

    ## for test
    version = "1.9.2"

    name = "avro"
    url = ""
    homepage = "https://avro.apache.org/"
    description = "Apache Avro is a data serialization system"
    topics = ("conan", "avro", "serialization")
    license = "Apache License, Version 2.0"
    exports_sources = ["CMakeLists.txt", "patches/*"]
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "fPIC": [True, False],
    }
    default_options = {
        "fPIC": True,
    }
    generators = "cmake"

    requires = (
        "boost/1.69.0",
        "snappy/1.1.8",
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
        extracted_name = "avro-release-" + self.version

        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)
        os.rename(extracted_name, self._source_subfolder)

        # patch 
        for patch in self.conan_data["patches"][self.version]:
            tools.patch(**patch)

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)
        #self._cmake.definitions["BUILD_TESTING"] = False
        self._cmake.configure()
        return self._cmake

    def package(self):
        self.copy("LICENSE", dst="licenses", src=self._source_subfolder+"/lang/c++")
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["avrocpp_s", "avrocpp"]
