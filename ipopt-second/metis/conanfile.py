from conans import ConanFile, CMake, tools
import os

class MetisConan(ConanFile):
    name = "metis"
    version = "5.1.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "remove_rint_fix": [True, False],
        "remove_infinity_fix": [True, False],
		"REALTYPEWIDTH": ["32", "64"],
		"IDXTYPEWIDTH": ["32", "64"],
    }
    default_options = "remove_rint_fix=True", "remove_infinity_fix=True", "REALTYPEWIDTH=64", "IDXTYPEWIDTH=32"

    def source(self):
        archive_filename = "metis-5.1.0.tar.gz"
        tools.download("http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz", archive_filename)
        tools.untargz(archive_filename)
        if self.options.remove_rint_fix == True:
            tools.replace_in_file("./metis-5.1.0/GKlib/gk_arch.h", "#define rint(x) ((int)((x)+0.5))", "//#define rint(x) ((int)((x)+0.5))")
        if self.options.remove_infinity_fix == True:
            tools.replace_in_file("./metis-5.1.0/GKlib/gk_arch.h", "#define INFINITY FLT_MAX", "//#define INFINITY FLT_MAX")

        tools.replace_in_file("./metis-5.1.0/include/metis.h", "#define IDXTYPEWIDTH 32", "#define IDXTYPEWIDTH {}".format(self.options.IDXTYPEWIDTH))
        tools.replace_in_file("./metis-5.1.0/include/metis.h", "#define REALTYPEWIDTH 32", "#define REALTYPEWIDTH {}".format(self.options.REALTYPEWIDTH))
                    
    def build(self):
        cmake = CMake(self)
        cmake.configure(source_folder="./metis-5.1.0",
                defs={"GKLIB_PATH": os.path.abspath("./metis-5.1.0/GKlib")}
            )
        cmake.build()    

    def package(self):
        self.copy("*metis.h", dst="include", keep_path=False)
        self.copy("*metis.lib", dst="lib", keep_path=False)
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.dylib", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.includedirs = ['./include']
        self.cpp_info.libs = ["metis"]
