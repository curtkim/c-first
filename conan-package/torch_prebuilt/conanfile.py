import os
from conans import ConanFile, CMake, tools
import tempfile

class TorchConan(ConanFile):
    name = "torch"
    version = "1.8.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {"cuda": ["9.2", "10.1", "10.2", "None"]}
    default_options = {"cuda": "10.1"}

    def build(self):
        cuda_version = "cu102"
        url_tail = ""
        if self.options.cuda == "10.1":
            cuda_version = "cu101"
            url_tail = "%2Bcu101"
        elif self.options.cuda == '9.2':        
            cuda_version = "cu92"
            url_tail = "%2Bcu92"
        elif self.options.cuda == 'None':
            cuda_version = "cpu"
            url_tail = "%2Bcpu"

        name = f"libtorch-cxx11-abi-shared-with-deps-{self.version}{url_tail}.zip"
        targetfile = os.path.join(tempfile.gettempdir(), name)

        if os.path.exists(targetfile) and not tools.get_env('TORCH_FORCE_DOWNLOAD', False):
            self.output.info(f'Skipping download. Using cached {targetfile}')
        else:
            url = f"https://download.pytorch.org/libtorch/{cuda_version}/{name}"
            self.output.info(f'Downloading libtorch from {url} to {targetfile}')
            tools.download(url, targetfile)
        tools.unzip(targetfile)
        #url = f"https://download.pytorch.org/libtorch/cu{cuda_version}/libtorch-cxx11-abi-shared-with-deps-{self.version}{url_tail}.zip"
        #tools.get(url)

    def package(self):
        self.copy("*", src="libtorch")
        #self.copy("*", src="libtorch/lib", dst="lib", keep_path=True)
        #self.copy("*", src="libtorch/include", dst="include", keep_path=True)
        #self.copy("*", src="libtorch/share", dst="share", keep_path=True)

    def package_info(self):
        self.cpp_info.libs = ['torch', 'torch_cpu', 'c10', 'pthread']
        self.cpp_info.includedirs = ['include', 'include/torch/csrc/api/include']
        self.cpp_info.bindirs = ['bin']
        self.cpp_info.libdirs = ['lib']
        if self.options.cuda != 'None':
            self.cpp_info.libs.extend(['torch_cuda', 'c10_cuda'])

        #self.cpp_info.libs = tools.collect_libs(self)
        #self.cpp_info.includedirs = ['include', 'include/torch/csrc/api/include']
