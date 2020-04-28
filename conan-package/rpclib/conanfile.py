import os
import shutil
from conans import ConanFile, tools, CMake


class Rpclib(ConanFile):

    ## for test
    version = "2.2.1"

    name = "rpclib"
    url = ""
    homepage = "http://rpclib.net"
    description = "Apache Avro is a data serialization system"
    topics = ("conan", "avro", "serialization")
    license = "MIT License"
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
        extracted_name = "rpclib-" + self.version

        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)
        os.rename(extracted_name, self._source_subfolder)

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
        self._cmake.configure()
        return self._cmake

    def package(self):
        self.copy("LICENSE.md", dst="licenses", src=self._source_subfolder)
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["rpc"]
