import os
import shutil
from conans import ConanFile, tools, CMake


class Libigl(ConanFile):

    ## for test
    version = "2.2.0"

    name = "libigl"
    url = ""
    homepage = "http://libigl.github.io/libigl/"
    description = "Simple C++ geometry processing library"
    topics = ("geometry", "matrices", "algorithms")
    license = "GNU GENERAL PUBLIC LICENSE"
    exports_sources = ["CMakeLists.txt", "patches/*"]
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "fPIC": [True, False],
        "static_library": [True, False],
    }
    default_options = {
        "fPIC": True,
        "static_library": True,
    }
    generators = "cmake"

    requires = (
        "eigen/3.3.7"
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
        extracted_name = self.name + "-" + self.version

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

        self._cmake.definitions["LIBIGL_EXPORT_TARGETS"] = "ON"
        if self.options.static_library:
            self._cmake.definitions["LIBIGL_USE_STATIC_LIBRARY"] = "ON"
        else:
            self._cmake.definitions["LIBIGL_USE_STATIC_LIBRARY"] = "OFF"

        # All these dependencies are needed to build the examples or the tests
        self._cmake.definitions["LIBIGL_BUILD_TUTORIALS"] = "OFF"
        self._cmake.definitions["LIBIGL_BUILD_TESTS"] = "OFF"
        self._cmake.definitions["LIBIGL_BUILD_PYTHON"] = "OFF"

        self._cmake.definitions["LIBIGL_WITH_CGAL"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_COMISO"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_CORK"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_EMBREE"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_MATLAB"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_MOSEK"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_OPENGL"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_OPENGL_GLFW"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_OPENGL_GLFW_IMGUI"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_PNG"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_TETGEN"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_TRIANGLE"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_XML"] = "OFF"
        self._cmake.definitions["LIBIGL_WITH_PYTHON"] = "OFF"

        self._cmake.configure()
        return self._cmake

    def package(self):
        self.copy("LICENSE*", dst="licenses", src=self._source_subfolder)
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        #self.cpp_info.libs = ["igl"]
        self.cpp_info.cppflags = ["-pthread"]        

        if self.options.static_library:
            self.cpp_info.libdirs = ["lib"]
            self.cpp_info.libs = ["libigl.a"]
            self.cpp_info.cppflags += ["-DIGL_STATIC_LIBRARY"]
