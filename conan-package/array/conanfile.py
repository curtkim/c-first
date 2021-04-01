import os
import shutil
from conans import ConanFile, tools, CMake


class Array(ConanFile):

    version = "20210210"

    name = "array"
    url = ""
    homepage = "https://github.com/dsharlet/array"
    description = "C++ multidimensional arrays in the spirit of the STL"    
    topics = ("multidimensional arrays", "stl", "cuda")
    license = "Apache-2.0 License"
    settings = "os", "compiler", "build_type", "arch"

    @property
    def _source_subfolder(self):
        return "source_subfolder"

    def source(self):
        if os.path.exists(self.name) and os.path.isdir(self.name):
            shutil.rmtree(self.name)

        self.run("git clone https://github.com/dsharlet/array.git")
        self.run("cd array && git checkout 656ee8e4dae7dbf09b72e8020f94dc3fc7d5be77")

    def package(self):
        self.copy(pattern="LICENSE", dst="licenses", src=self.name)
        self.copy("*.h", src=self.name, dst="include/array", excludes=("examples/*", "test/*"))

    def package_id(self):
        self.info.header_only()
