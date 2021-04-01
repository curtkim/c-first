from conans import ConanFile, AutoToolsBuildEnvironment, tools


class FreeImage(ConanFile):
    name = "freeimage"
    version = "3.18.0"
    license = "https://gitlab.lrde.epita.fr/olena/freeimage-mirror/blob/master/license-fi.txt"
    url = "https://gitlab.lrde.epita.fr/olena/conan-freeimage.git"
    homepage = "https://gitlab.lrde.epita.fr/olena/mirror-freeimage"
    description = "FreeImage is an Open Source library project for developers who would like to support popular graphics image formats like PNG, BMP, JPEG, TIFF and others as needed by today's multimedia applications."
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False], 
        "fPIC": [True, False]
    }
    default_options = "shared=False", "fPIC=False"
    generators = "cmake"

    def configure(self):
        if self.settings.os == "Windows":
            self.options.remove("fPIC")

    def source(self):
        git = tools.Git()
        git.clone("{}.git".format(self.homepage))
        git.checkout("{}".format(self.version))
        # added
        tools.replace_in_file("Makefile.gnu", "-o root -g root ", "")

    def build(self):
        autotools = AutoToolsBuildEnvironment(self)
        env_build_vars = autotools.vars
        env_build_vars['DESTDIR'] = self.package_folder
        autotools.make(vars=env_build_vars)

    def package(self):
        autotools = AutoToolsBuildEnvironment(self)
        env_build_vars = autotools.vars
        env_build_vars['DESTDIR'] = self.package_folder
        autotools.install(vars=env_build_vars)

        with tools.environment_append(autotools.vars):
            self.run(
                "mv {}/usr/* {}/".format(self.package_folder, self.package_folder))
            self.run("rm -rf {}/usr".format(self.package_folder))

    def package_info(self):
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.libdirs = ["lib"]
        self.cpp_info.libs = ["freeimage"]
