import os
import shutil
from conans import ConanFile, tools, CMake

class HFSM2(ConanFile):
    name = "hfsm2"
    version = "1.7.3"
    url = "https://github.com/andrew-gresyk/HFSM2"
    homepage = "https://github.com/andrew-gresyk/HFSM2"
    description = "High-Performance Hierarchical Finite State Machine Framework"
    topics = ("Finite State Machine", "FSM")
    settings = "os"
    license = "MIT License"

    _source_subfolder = "source_subfolder"

    def source(self):
        tools.get(**self.conan_data["sources"][self.version])
        extracted_name = self.name.upper() + "-" + self.version.replace('.', '_')

        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)
        os.rename(extracted_name, self._source_subfolder)

    def package(self):
        root_dir = self._source_subfolder
        include_dir = os.path.join(root_dir, "include")
        self.copy(pattern="LICENSE", dst="licenses", src=root_dir)
        self.copy(pattern="*.hpp", dst="include", src=include_dir)
        self.copy(pattern="*.inl", dst="include", src=include_dir)

    def package_id(self):
        self.info.header_only()