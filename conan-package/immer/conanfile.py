import os
from conans import ConanFile, tools, CMake
import shutil

class Immer(ConanFile):
    name = "immer"
    version = "0.6.2"
    url = "https://github.com/arximboldi/immer"
    homepage = "https://github.com/arximboldi/immer"
    description = "Postmodern immutable and persistent data structures for C++ — value semantics at scale"
    topics = ("conan", "immutable", "data structure")
    settings = "os"
    license = "BSL-1.0 License"
    exports_sources = ["CMakeLists.txt"]

    _cmake = None

    
    _source_subfolder = "source_subfolder"

    def source(self):
        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)

        tools.get(**self.conan_data["sources"][self.version])
        extracted_name = self.name + '-' + self.version
        os.rename(extracted_name, self._source_subfolder)

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
        self.copy("*.hpp", src=self._source_subfolder + "/immer", dst="include/immer", keep_path=True) # from source        

    def package_id(self):
        self.info.header_only()