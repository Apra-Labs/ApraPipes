cd gstreamer
meson --prefix=${PWD}/outInstall builddir -Dgpl=enabled
ninja -C builddir
meson install -C builddir
cd ..