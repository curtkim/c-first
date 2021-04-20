import os
from conans import ConanFile, tools, CMake
import shutil

class Klein(ConanFile):
    name = "klein"
    version = "2.2.1"
    url = "https://github.com/jeremyong/klein"
    homepage = "https://github.com/jeremyong/klein"
    description = "P(R*_{3, 0, 1}) specialized SIMD Geometric Algebra Library"
    topics = ("conan", "geometric algebra", "SIMD")
    settings = "os"
    license = "MIT License"
    exports_sources = ["CMakeLists.txt"]

    _cmake = None
    
    _source_subfolder = "source_subfolder"

    def source(self):
        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)

        tools.get(**self.conan_data["sources"][self.version])
        extracted_name = self.name + '-' + self.version
        os.rename(extracted_name, self._source_subfolder)


    def package(self):
        root_dir = os.path.join(self._source_subfolder, self.name)
        self.copy(pattern="LICENSE", dst="licenses", src=root_dir)
        self.copy("*.hpp", src=self._source_subfolder+"/public", dst="include", keep_path=True, excludes=("test/*")) # from source

    def package_id(self):
        self.info.header_only()