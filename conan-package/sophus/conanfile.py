from conans import ConanFile, tools
import os


class SophusConan(ConanFile):
    name = "sophus"
    version = "20210517"
    description = "C++ implementation of Lie Groups using Eigen."
    topics = ("conan", "eigen", "numerical", "math")
    url = "https://github.com/conan-io/conan-center-index"
    homepage = "https://strasdat.github.io/Sophus/"
    license = "MIT"
    no_copy_source = True

    requires = (
        "eigen/3.3.7",
        "fmt/7.1.3"
    )

    def source(self):
        if os.path.exists(self.name) and os.path.isdir(self.name):
            shutil.rmtree(self.name)

        self.run("git clone https://github.com/strasdat/Sophus.git")
        self.run("cd Sophus && git checkout 9252e2d2f89fd40dd595c00129d452e1527e0d01")

    def package(self):
        print('os.getcwd()', os.getcwd())
        self.copy("LICENSE.txt", src="Sophus", dst="licenses")
        self.copy(pattern="*.hpp", src="Sophus/sophus", dst=os.path.join("include", "sophus"), keep_path=False)

    def package_id(self):
        self.info.header_only()

    def package_info(self):
        self.cpp_info.names["cmake_find_package"] = "Sophus"
        self.cpp_info.names["cmake_find_package_multi"] = "Sophus"