import os
import shutil
from conans import ConanFile, tools, CMake


class Libunifex(ConanFile):

    version = "20220210"

    name = "libunifex"
    url = ""
    homepage = "https://github.com/facebookexperimental/libunifex"
    description = "Apache Avro is a data serialization system"
    topics = ("conan", "libunifex", "coroutine", "io_uring")
    license = "Apache License, Version 2.0"
    exports_sources = ["CMakeLists.txt", "patches/*"]
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "fPIC": [True, False],
        "io_uring": [True, False],
    }
    default_options = {
        "fPIC": True,
        "io_uring": False
    }
    generators = "cmake"

    requires = (
        #"boost/1.69.0",
        #"snappy/1.1.8",
    )    

    _cmake = None

    @property
    def _source_subfolder(self):
        return "source_subfolder"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def source(self):
        if os.path.exists(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)

        git = tools.Git(folder=self._source_subfolder)

        git.clone("https://github.com/facebookexperimental/libunifex.git")
        git.run("reset --hard " + "7c646b063ce55a8c999c906e7845dd5c53d88f76")

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)
        self._cmake.definitions["UNIFEX_NO_LIBURING"] = "OFF" if self.options.io_uring else "ON"
        self._cmake.configure()
        return self._cmake

    def package(self):
        #self.copy("LICENSE", dst="licenses", src=self._source_subfolder+"/lang/c++")
        #cmake = self._configure_cmake()
        #cmake.install()
        self.copy("*.a", dst="lib", keep_path=False)
        #self.copy("**/*.h", src=self._source_subfolder + "/LibCarla/source", dst="include", keep_path=True, excludes=("compiler/*", "test/*")) # from source        
        self.copy("**/*.hpp", src=self._source_subfolder + "/include", dst="include", keep_path=True) # from source        

    def package_info(self):
        self.cpp_info.libs = ["unifex"]
        if str(self.settings.os) in ["Linux", "Android"]:
            self.cpp_info.libs.append('pthread')

        #self.cpp_info.libs = tools.collect_libs(self)
