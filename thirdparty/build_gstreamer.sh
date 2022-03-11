# meson from pip3 only
cd gst-build
./gst-worktree.py add gst-build-1.16 origin/1.16
cd gst-build-1.16
git checkout 1.16.2
meson --prefix=${PWD}/outInstall builddir -Dpython=disabled
ninja -C builddir
sudo meson install -C builddir
rm -rf builddir
cd ../..
