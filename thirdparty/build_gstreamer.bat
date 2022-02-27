cd gst-build
python gst-worktree.py add gst-build-1.16 origin/1.16
cd gst-build-1.16
git checkout 1.16.2
dir
set VSLANG=1033
set CC=clang
set CC_LD=link
echo "%link% %PATH%"
set link="C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64\link.exe"
echo "%VSLANG% %CC% %CC_LD% %link%"
meson --prefix=%cd%\outInstall builddir -Dpython=disabled -Ddevtools=disabled
meson compile -C builddir
meson install -C builddir