import os
import shutil
from conans import ConanFile, tools, CMake


class Libunifex(ConanFile):

    version = "3.1.0"

    name = "glbinding"
    url = ""
    homepage = "https://github.com/cginternals/glbinding"
    description = "A C++ binding for the OpenGL API"
    topics = ("conan", "opengl")
    license = "MIT License"
    exports_sources = ["CMakeLists.txt"]
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }
    default_options = {
        "fPIC": True,
        "shared": False,
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

        tools.get("https://github.com/cginternals/glbinding/archive/v3.1.0.tar.gz")
        extracted_name = self.name + "-" + self.version        
        os.rename(extracted_name, self._source_subfolder)

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        cmake = CMake(self)
        cmake.definitions["BUILD_SHARED_LIBS"] = "ON" if self.options.shared else "OFF"
        cmake.definitions["OPTION_BUILD_TEST"] = "OFF"
        cmake.definitions["OPTION_BUILD_DOCS"] = "OFF"
        cmake.definitions["OPTION_BUILD_TOOLS"] = "ON"
        cmake.definitions["OPTION_BUILD_EXAMPLES"] = "OFF"

        cmake.configure()
        self._cmake = cmake
        return self._cmake

    def package(self):
        #self.copy("LICENSE", dst="licenses", src=self._source_subfolder+"/lang/c++")
        cmake = self._configure_cmake()
        cmake.install()
        #self.copy("*.a", dst="lib", keep_path=False)
        #self.copy("**/*.h", src=self._source_subfolder + "/LibCarla/source", dst="include", keep_path=True, excludes=("compiler/*", "test/*")) # from source        
        #self.copy("**/*.hpp", src=self._source_subfolder + "/include", dst="include", keep_path=True) # from source        

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)