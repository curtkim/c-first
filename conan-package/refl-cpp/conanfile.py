import os
from conans import ConanFile, tools, CMake
import shutil

class ReflCpp(ConanFile):
    name = "refl-cpp"
    version = "20210405"
    url = "https://github.com/veselink1/refl-cpp"
    homepage = "https://github.com/veselink1/refl-cpp"
    description = "A modern compile-time reflection library for C++ with support for overloads, templates, attributes and proxies"
    topics = ("conan", "metaprogramming", "reflection")
    settings = "os"
    license = "MIT License"
    exports_sources = ["CMakeLists.txt"]

    _cmake = None

    
    _source_subfolder = "source_subfolder"

    def source(self):
        '''
        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)

        tools.get(**self.conan_data["sources"][self.version])
        extracted_name = self.name + '-' + self.version
        os.rename(extracted_name, self._source_subfolder)
        '''
        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)

        self.run("git clone https://github.com/veselink1/refl-cpp.git")
        self.run("cd refl-cpp && git checkout 4684ff412bda23db3c757307f4fa16b54a5c5788")
        os.rename(self.name, self._source_subfolder)


    def package(self):
        root_dir = os.path.join(self._source_subfolder, self.name)
        self.copy(pattern="LICENSE", dst="licenses", src=root_dir)
        self.copy("*.hpp", src=self._source_subfolder+"/include", dst="include/refl-cpp", keep_path=True, excludes=("test/*")) # from source

    def package_id(self):
        self.info.header_only()