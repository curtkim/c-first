import os
import shutil
from conans import ConanFile, tools, CMake


class CarlaRecast(ConanFile):

    ## for test
    version = "20190509"

    name = "carla-recast"
    url = ""
    homepage = "https://github.com/carla-simulator/recastnavigation"
    description = "Navigation-mesh Toolset for Games"
    topics = ("conan", "navigation")
    license = ""
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

        git = tools.Git(folder=self._source_subfolder)
        git.clone("https://github.com/carla-simulator/recastnavigation.git", "master")
        git.run("reset --hard " + "cdce4e1a270fdf1f3942d4485954cc5e136df1df")

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
        self._cmake.definitions["RECASTNAVIGATION_DEMO"] = False
        self._cmake.definitions["RECASTNAVIGATION_BUILDER"] = False
        self._cmake.definitions["RECASTNAVIGATION_TESTS"] = False
        self._cmake.definitions["RECASTNAVIGATION_EXAMPLES"] = False
        
        self._cmake.configure()
        return self._cmake

    def package(self):
        self.copy("License.txt", dst="licenses", src=self._source_subfolder)
        self.copy("*.a", dst="lib", keep_path=False)
        self.copy("*.h", src=self._source_subfolder + "/Recast/Include", dst="include/recast", keep_path=False) # from source
        
    def package_info(self):
        self.cpp_info.libs = ["Recast", "DetourTileCache", "DetourCrowd", "Detour", "DebugUtils"]

        #self.copy("License.txt", dst="licenses", src=self._source_subfolder)
        #self.copy("*.h", src=self._source_subfolder+"/Recast/include", dst="include/recast", keep_path=False) # from source
        #self.copy("*.a", dst="lib", keep_path=False)        
