for /F "tokens=*" %%A in ('"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath') do SET VCVARSPATH="%%A\VC\Auxiliary\Build\"
pushd %VCVARSPATH%
call vcvars64.bat
popd 
@echo on 

cd gst-build

rem use on GHA to cleanup the folder
IF /I NOT "%1"=="clean" GOTO NOCLEAN
rmdir /s /q gst-build-1.16

:NOCLEAN
python gst-worktree.py add gst-build-1.16 origin/1.16
cd gst-build-1.16

IF ERRORLEVEL 1 (echo "we can not cd into the above dir. Better to skip the build and fail" ) && ( cd..) && ( EXIT /B 1)

git checkout 1.16.2

set VSLANG=1033
meson --prefix=%cd%\outInstall builddir -Dpython=disabled -Ddevtools=disabled 
meson compile -C builddir "--vs-args=/MP"
meson install -C builddir
del /S /Q builddir
del /S /Q subprojects
dir
