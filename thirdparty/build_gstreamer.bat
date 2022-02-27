cd gst-build
python gst-worktree.py add gst-build-1.16 origin/1.16
cd gst-build-1.16
git checkout 1.16.2
dir
set VSLANG=1033
CC=clang-cl CC_LD=link meson --prefix=%cd%\outInstall builddir -Dpython=disabled -Ddevtools=disabled
CC=clang-cl CC_LD=link meson compile -C builddir
CC=clang-cl CC_LD=link meson install -C builddir