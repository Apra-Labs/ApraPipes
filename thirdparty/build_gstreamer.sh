#!/bin/bash
pwd 
ls
cd gst-build
git worktree list

rm -rf gst-build-1.16 || true
git worktree prune
git worktree list

dos2unix ./gst-worktree.py

./gst-worktree.py add gst-build-1.16 origin/1.16
cd gst-build-1.16
git checkout 1.16.2
meson --prefix=${PWD}/outInstall builddir -Dpython=disabled
ninja -C builddir
meson install -C builddir
rm -rf builddir
cd ../..
