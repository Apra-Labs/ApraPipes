cd gst-build
python gst-worktree.py add gst-build-1.16 origin/1.16
cd gst-build-1.16
git checkout 1.16.2
dir
set VSLANG=1033
meson --prefix=%cd%\outInstall builddir -Dpython=disabled -Ddevtools=disabled
meson compile -C builddir
meson install -C builddir