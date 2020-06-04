import os
import shutil
from conans import ConanFile, tools, CMake


class LodePNG(ConanFile):

    ## for test
    version = "20200520"

    name = "lodepng"
    url = ""
    homepage = "https://github.com/lvandeve/lodepng"
    description = "PNG encoder and decoder in C and C++."    
    topics = ("png", "encoder", "decoder")
    license = "zlib License"
    exports_sources = ["CMakeLists.txt"]
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"

    def source(self):

        if os.path.exists(self.name) and os.path.isdir(self.name):
            shutil.rmtree(self.name)

        self.run("git clone https://github.com/lvandeve/lodepng.git")
        self.run("cd lodepng && git checkout 486d165ed70999cd575a9996a7f2551c7b238c81")

        #tools.get(**self.conan_data["sources"][self.version])
        #extracted_name = self.name + "-" + self.version

        #if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
        #    shutil.rmtree(self._source_subfolder)
        #os.rename(extracted_name, self._source_subfolder)

        # patch 
        #for patch in self.conan_data["patches"][self.version]:
        #    tools.patch(**patch)

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_dir=self.source_folder)
        cmake.build()

    def package(self):
        self.copy("*.h", dst="include", keep_path=False)
        self.copy("*.lib", dst="lib", keep_path=False)
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.dylib", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["lodepng"]