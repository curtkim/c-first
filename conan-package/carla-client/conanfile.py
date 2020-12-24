import os
import shutil
from conans import ConanFile, tools, CMake


class CarlaClient(ConanFile):

    ## for test
    version = "0.9.11"

    name = "carla-client"
    url = ""
    homepage = "http://carla.org"
    description = "Open-source simulator for autonomous driving research"
    topics = ("conan", "carla", "autonomouse", "driving")
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
        "boost/1.72.0",
        "rpclib/2.2.1@demo/testing",
        "carla-recast/20190509@demo/testing",        
        "libpng/1.6.37",
        "libtiff/4.1.0",
        "libjpeg/9d",
    )

    _cmake = None

    @property
    def _source_subfolder(self):
        return "source_subfolder"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def source(self):
        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)

        tools.get(**self.conan_data["sources"][self.version])
        #tools.unzip("/data/Downloads/carla-0.9.9.tar.gz")
        extracted_name = "carla-" + self.version
        os.rename(extracted_name, self._source_subfolder)

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)
        self._cmake.definitions["CARLA_VERSION"] = self.version
        self._cmake.definitions["LIBCARLA_BUILD_RELEASE"] = True
        self._cmake.definitions["LIBCARLA_BUILD_DEBUG"] = False
        
        self._cmake.configure()
        return self._cmake

    def package(self):
        self.copy("LICENSE", dst="licenses", src=self._source_subfolder)
        #cmake = self._configure_cmake()
        #cmake.install()
        self.copy("*.a", dst="lib", keep_path=False)
        self.copy("**/*.h", src=self._source_subfolder + "/LibCarla/source", dst="include", keep_path=True, excludes=("compiler/*", "test/*")) # from source        
        self.copy("**/*.hpp", src=self._source_subfolder + "/LibCarla/source", dst="include", keep_path=True) # from source        

    def package_info(self):
        self.cpp_info.libs = ["carla_client"]
