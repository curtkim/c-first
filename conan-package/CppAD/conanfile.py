import os
from conans import ConanFile, tools, CMake


class CppAD(ConanFile):
    name = "CppAD"
    version = "20200000.3"
    url = "https://github.com/coin-or/CppAD"
    homepage = "https://github.com/coin-or/CppAD"
    description = "A C++ Algorithmic Differentiation Package"
    topics = ("conan", "cppad", "Algorithmic Differentiation")
    settings = "os"
    license = ""
    #exports_sources = ["CMakeLists.txt"]

    _cmake = None

    #_source_subfolder = "source_subfolder"

    def source(self):
        tools.get(**self.conan_data["sources"][self.version])
        self.run("mv {}-{}/* .".format(self.name, self.version))

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)
        #self._cmake.definitions["UNIFEX_NO_LIBURING"] = "OFF" if self.options.io_uring else "ON"
        self._cmake.configure()
        return self._cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def package(self):
        cmake = self._configure_cmake()
        cmake.install()
        self.copy("**", dst="", src="package", keep_path=True)
        #self.copy(pattern="LICENSE.md", dst="licenses", src=root_dir)
        #self.copy("**/*.hpp", src=self._source_subfolder + "/include", dst="include", keep_path=True) # from source        

    #def package_info(self):
        #self.cpp_info.defines.append('ASIO_STANDALONE')
        #if str(self.settings.os) in ["Linux", "Android"]:
        #    self.cpp_info.libs.append('pthread')

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)