import os
from conans import ConanFile, CMake, tools

class TorchConan(ConanFile):
    name = "torch"
    version = "1.5.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {"cuda": ["9.2", "10.1", "10.2"]}
    default_options = {"cuda": "10.2"}

    def build(self):
        cuda_version = "102"
        if self.options.cuda == "10.1":
            cuda_version = "101"
        elif self.options.cuda == '9.2':        
            cuda_version = "92"

        url = f"https://download.pytorch.org/libtorch/cu{cuda_version}/libtorch-cxx11-abi-shared-with-deps-{self.version}.zip"
        tools.get(url)

    def package(self):
        self.copy("*", src="libtorch/lib", dst="lib", keep_path=True)
        self.copy("*", src="libtorch/include", dst="include", keep_path=True)
        self.copy("*", src="libtorch/share", dst="share", keep_path=True)

    def package_info(self):
        self.cpp_info.libs = self.collect_libs()
        #self.cpp_info.libs = ["hello"]