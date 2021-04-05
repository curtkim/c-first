from conans import ConanFile, AutoToolsBuildEnvironment, tools
import os



class CoinOrMumpsConan(ConanFile):
    name = "coin-or-mumps"
    version = "4.10.0"
    license = "proprietary" # license of MUMPS, not the build scripts
    author = "<Put your name here> <And your email here>"
    url = "https://github.com/coin-or-tools/ThirdParty-Mumps"
    description = "MUMPS for COIN-OR projects"
    #topics = ("<Put some tag here>", "<here>", "<and here>")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}
    #generators = "cmake"

    #requires = "openblas/0.3.12"

    @property
    def _source_subfolder(self):
        return "source-subfolder"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        self.options["openblas"].build_lapack = True

    def source(self):
        self.run("git clone {}".format(CoinOrMumpsConan.url))
        os.rename("ThirdParty-Mumps", self._source_subfolder)
        with tools.chdir(self._source_subfolder):
            self.run("./get.Mumps")

    def build(self):
        with tools.chdir(self._source_subfolder):
            autotools = AutoToolsBuildEnvironment(self)
            autotools.libs.append("m")
            autotools.configure()
            autotools.make()
            autotools.install()

    def package_info(self):
        self.cpp_info.libs = ["coinmumps"]

