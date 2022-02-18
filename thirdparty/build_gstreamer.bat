cd gst-build
python gst-worktree.py add gst-build-1.16 origin/1.16
cd gst-build-1.16
git checkout 1.16.2
set PKG_CONFIG_PATH=C:\src\ApraPipes\vcpkg\installed\x64-windows\lib\pkgconfig
meson --prefix=%cd%\outInstall builddir -Dpython=disabled
meson compile -C builddir
meson install -C builddir