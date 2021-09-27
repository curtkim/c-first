from conans import ConanFile, tools, CMake
import os
import shutil


class AutodiffConan(ConanFile):
    name = "autodiff"
    version = "0.6.4"
    description = "automatic differentiation made easier for C++"
    topics = ("conan", "autodiff", "numerical", "math")
    url = "https://github.com/autodiff/autodiff"
    homepage = "https://github.com/autodiff/autodiff"
    license = "MIT"
    generators = "cmake"
    options = {
        "fPIC": [True, False],
        "static_library": [True, False],
    }
    default_options = {
        "fPIC": True,
        "static_library": True,
    }

    exports_sources = ["CMakeLists.txt"]

    @property
    def _source_subfolder(self):
        return "source_subfolder"

    requires = (
        "eigen/3.4.0",
    )
    _cmake = None


    def source(self):
        tools.get(**self.conan_data["sources"][self.version])
        extracted_name = self.name + "-" + self.version

        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)
        os.rename(extracted_name, self._source_subfolder)


    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)
        self._cmake.definitions["AUTODIFF_BUILD_TESTS"] = "OFF"
        self._cmake.definitions["AUTODIFF_BUILD_PYTHON"] = "OFF"
        self._cmake.definitions["AUTODIFF_BUILD_EXAMPLES"] = "OFF"
        self._cmake.definitions["AUTODIFF_BUILD_DOCS"] = "OFF"
        self._cmake.configure()

        return self._cmake

    def package(self):
        #self.copy("LICENSE", dst="licenses", src=self._source_subfolder+"/lang/c++")
        cmake = self._configure_cmake()
        cmake.install()

    def package_id(self):
        self.info.header_only()
