import os
from conans import ConanFile, tools, CMake
import shutil

class Zug(ConanFile):
    name = "zug"
    version = "20210324"
    url = "https://github.com/arximboldi/zug"
    homepage = "https://github.com/arximboldi/zug"
    description = "Transducers for C++ — Clojure style higher order push/pull sequence transformations"
    topics = ("conan", "sequence", "transducers")
    settings = "os"
    license = "BSL-1.0 License"
    exports_sources = ["CMakeLists.txt"]

    _cmake = None

    
    _source_subfolder = "source_subfolder"

    def source(self):
        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)

        self.run("git clone https://github.com/arximboldi/zug.git")
        self.run("cd zug && git checkout b28a04ce6ed457dd9cff2232b344499b1b48e884")
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
        self.copy("*.hpp", src=self._source_subfolder + "/zug", dst="include/zug", keep_path=True) # from source        

    def package_id(self):
        self.info.header_only()
