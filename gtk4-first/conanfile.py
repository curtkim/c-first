from conans import ConanFile, Meson

class GtkConanMeson(ConanFile):
    name = "gtkconanmeson"
    version = "0.1"
    settings = "os", "compiler", "build_type", "arch"
    generators = "pkg_config"
    requires = "gtk/4.4.0"
    exports_sources = "src/*"

    def build(self):
        meson = Meson(self)
        meson.configure(source_folder="%s/src" % self.source_folder,
                        build_folder="%s/dist" % self.source_folder)
        meson.build()


