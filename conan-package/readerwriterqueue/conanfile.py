import os
import shutil
from conans import ConanFile, tools, CMake


class ReaderWriterQueue(ConanFile):

    version = "1.0.2"

    name = "readerwriterqueue"
    url = ""
    homepage = "https://github.com/cameron314/readerwriterqueue"
    description = "A single-producer, single-consumer lock-free queue for C++"    
    topics = ("lockfree", "queue")
    license = "Simplified BSD License"
    settings = "os", "compiler", "build_type", "arch"
    
    _source_subfolder = "source_subfolder"
    # header only library에서 package할때 build_dir대신에 source_dir에서 copy하도록 한다.
    no_copy_source = True

    def source(self):
        tools.get(**self.conan_data["sources"][self.version])
        extracted_name = self.name + "-" + self.version

        if os.path.exists(self._source_subfolder) and os.path.isdir(self._source_subfolder):
            shutil.rmtree(self._source_subfolder)
        os.rename(extracted_name, self._source_subfolder)

    def package(self):    
        root_dir = self._source_subfolder
        self.copy(pattern="LICENSE.md", dst="licenses", src=root_dir)
        self.copy(pattern="atomicops.h", dst="include", src=root_dir)
        self.copy(pattern="readerwriterqueue.h", dst="include", src=root_dir)

    def package_id(self):
        self.info.header_only()
