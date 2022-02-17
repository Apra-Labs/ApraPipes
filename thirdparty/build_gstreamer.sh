cd gst-build
./gst-worktree.py add gst-build-1.16 origin/1.16
cd gst-build-1.16
git checkout 1.16.2
sudo /root/.local/bin/meson --prefix=${PWD}/outInstall builddir -Dpython=disabled
ninja -C builddir
sudo /root/.local/bin/meson install -C builddir
cd .. 