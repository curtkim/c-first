from conans import ConanFile, AutoToolsBuildEnvironment, tools
import os
import shutil

class Ipopt(ConanFile):
    name = "Ipopt"
    version = "3.12.7"
    url = "https://github.com/coin-or/Ipopt"
    homepage = "https://github.com/coin-or/Ipopt"
    description = "COIN-OR Interior Point Optimizer"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False], 
        "fPIC": [True, False]
    }
    default_options = "shared=False", "fPIC=False"

    requires = (
        "CppAD/20200000.3@curt/testing",
    )

    def configure(self):
        if self.settings.os == "Windows":
            self.options.remove("fPIC")

    def source(self):
        self.run("rm -rf ./*")
        tools.get(**self.conan_data["sources"][self.version])
        self.run("mv {}-{}/* .".format(self.name, self.version))
        self.run("chmod +x configure")
        self.run('find . -name "install-sh" -exec chmod +x {} \;')                
        self.run("chmod +x get.Mumps && ./get.Mumps", cwd='ThirdParty/Mumps')

    def build(self):    
        autotools = AutoToolsBuildEnvironment(self)
        autotools.libs.append("m")
        autotools.configure()
        autotools.make()
        autotools.install()

    '''
    def build(self):
        print(os.getcwd())

        autotools = AutoToolsBuildEnvironment(self)
        env_build_vars = autotools.vars
        env_build_vars['DESTDIR'] = self.package_folder
        autotools.configure()
        autotools.make(vars=env_build_vars)

    def package(self):
        autotools = AutoToolsBuildEnvironment(self)
        env_build_vars = autotools.vars
        env_build_vars['DESTDIR'] = self.package_folder
        autotools.install(vars=env_build_vars)

        #with tools.environment_append(autotools.vars):
        #    self.run(
        #        "mv {}/usr/* {}/".format(self.package_folder, self.package_folder))
        #    self.run("rm -rf {}/usr".format(self.package_folder))
    '''

    def package(self):
        self.copy('*', src='package', dst='.', keep_path=True)


    def package_info(self):
        #self.cpp_info.includedirs = ["include"]
        #self.cpp_info.libdirs = ["lib"]
        self.cpp_info.libs = ["ipopt", "coinmumps"]
