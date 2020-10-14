import os
from conans import ConanFile, tools


class Asio(ConanFile):
    name = "pl_mpeg"
    version = "20200510"
    url = "https://github.com/conan-io/conan-center-index"
    homepage = "https://github.com/phoboslab/pl_mpeg"
    description = "Single file C library for decoding MPEG1 Video and MP2 Audio"
    topics = ("conan", "mpeg", "video", "mp2", "audio")
    settings = "os"
    license = ""

    def source(self):
        self.run("git clone https://github.com/phoboslab/pl_mpeg.git")

    def package(self):
        self.copy(pattern="pl_mpeg.h", dst="include", src=self.name)

    def package_id(self):
        self.info.header_only()