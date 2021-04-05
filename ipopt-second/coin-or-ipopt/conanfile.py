from conans import ConanFile, AutoToolsBuildEnvironment, tools
import os



class IPOPTConan(ConanFile):
    name = "ipopt"
    version = "3.13.3"
    license = "EPL-2.0"
    #author = "<Put your name here> <And your email here>"
    url = "https://github.com/coin-or/Ipopt"
    description = "COIN-OR Interior Point Optimizer IPOPT"
    #topics = ("<Put some tag here>", "<here>", "<and here>")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}
    #generators = "cmake"

    requires = "metis/5.1.0", "coin-or-mumps/4.10.0"

    @property
    def _source_subfolder(self):
        return "Ipopt"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def source(self):
        self.run("git clone {} --branch releases/3.13.3".format(IPOPTConan.url))

    def build(self):
        with tools.chdir(self._source_subfolder):
            autotools = AutoToolsBuildEnvironment(self)
            autotools.libs.append("m")
            autotools.configure()
            autotools.make()
            autotools.install()

    def package_info(self):
        self.cpp_info.libs = ["ipopt"]
        #self.cpp_info.includedirs = ['include/coin-or']

