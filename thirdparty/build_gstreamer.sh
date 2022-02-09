cd gstreamer
meson --prefix=${PWD}/outInstall builddir
ninja -C builddir
meson install -C builddir