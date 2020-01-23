from conans import ConanFile, CMake, tools


class RpclibConan(ConanFile):
    name = "rpclib"
    version = "2.2.1"
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of Rpclib here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = {"shared": False}
    generators = "cmake"

    def source(self):
        self.run("git clone https://github.com/rpclib/rpclib.git")
        # This small hack might be useful to guarantee proper /MT /MD linkage
        # in MSVC if the packaged project doesn't have variables to set it
        # properly
        tools.replace_in_file("rpclib/CMakeLists.txt", "project(rpc VERSION 2.2.1)",
                              '''project(rpc VERSION 2.2.1)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()''')

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_folder="rpclib")
        cmake.build()

        # Explicit way:
        # self.run('cmake %s/hello %s'
        #          % (self.source_folder, cmake.command_line))
        # self.run("cmake --build . %s" % cmake.build_config)

    def package(self):
        #self.run('pwd')
        self.copy("*.h", src="rpclib/include", dst="include", keep_path=True)
        self.copy("*.hpp", src="rpclib/include", dst="include", keep_path=True)
        self.copy("*.inl", src="rpclib/include", dst="include", keep_path=True)
        self.copy("*.a", src="lib", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["rpclib"]

        self.cpp_info.static.libs = ["librpc.a"]

