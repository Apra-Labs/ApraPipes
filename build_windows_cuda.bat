@echo off
set batdir=%~dp0
cd %batdir%/base
powershell -nologo -executionpolicy bypass -File build_dependencies_windows_cuda.ps1
cd ..

@echo off
set batdir=%~dp0
cd %batdir%/vcpkg

call bootstrap-vcpkg.bat
@echo on 
vcpkg.exe integrate install
cd ..



SET VCPKG_ARGS=-DENABLE_CUDA=ON -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base

mkdir _build
cd _build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo %VCPKG_ARGS%
cmake --build .
cd ..
rem goto :EOF
mkdir _debugbuild
cd _debugbuild
cmake -DCMAKE_BUILD_TYPE=Debug %VCPKG_ARGS%
cmake --build .