import os
from conans import ConanFile, tools, CMake


class CppLazy(ConanFile):
    name = "cpp-lazy"
    version = "2.3.3"
    url = "https://github.com/MarcDirven/cpp-lazy"
    homepage = "https://github.com/MarcDirven/cpp-lazy"
    description = "C++11/14/17/20 library for lazy evaluation"
    topics = ("conan", "lazy")
    settings = "os"
    license = "MIT License"
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
        self.copy(pattern="LICENSE.md", dst="licenses", src=root_dir)
        self.copy("**/*.hpp", src=self._source_subfolder + "/include", dst="include", keep_path=True) # from source        


    #def package_info(self):
        #self.cpp_info.defines.append('ASIO_STANDALONE')
        #if str(self.settings.os) in ["Linux", "Android"]:
        #    self.cpp_info.libs.append('pthread')

    def package_id(self):
        self.info.header_only()