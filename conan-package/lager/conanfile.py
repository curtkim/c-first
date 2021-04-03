import os
from conans import ConanFile, tools, CMake
import shutil

class Lager(ConanFile):
    name = "lager"
    version = "20210401"
    url = "https://github.com/arximboldi/lager"
    homepage = "https://github.com/arximboldi/immer"
    description = "C++ library for value-oriented design using the unidirectional data-flow architecture — Redux for C++"
    topics = ("conan", "value-oriented", "unidirectional data-flow", "redux")
    settings = "os"
    license = "BSL-1.0 License"
    exports_sources = ["CMakeLists.txt"]

    requires = (
        "immer/0.6.2@curt/testing",
        "zug/20210324@curt/testing",
        "cereal/1.3.0",
    )


    _cmake = None

    
    _source_subfolder = "source_subfolder"

    def source(self):
        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)

        self.run("git clone https://github.com/arximboldi/lager.git")
        self.run("cd lager && git checkout b94705ed180c0844523c845c9e3c668040de328e")
        os.rename(self.name, self._source_subfolder)

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)
        #self._cmake.definitions["UNIFEX_NO_LIBURING"] = "OFF" if self.options.io_uring else "ON"
        self._cmake.configure()
        return self._cmake

    def package(self):
        #cmake = self._configure_cmake()
        #cmake.install()
        root_dir = os.path.join(self._source_subfolder, self.name)
        self.copy(pattern="LICENSE", dst="licenses", src=root_dir)
        self.copy("*.hpp", src=self._source_subfolder + "/lager", dst="include/lager", keep_path=True) # from source        

    def package_id(self):
        self.info.header_only()