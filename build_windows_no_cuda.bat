@echo off
set batdir=%~dp0
cd %batdir%/build_scripts
powershell -nologo -executionpolicy bypass -File build_dependencies_windows_no_cuda.ps1
cd ..

@echo off
set batdir=%~dp0
cd %batdir%/base
powershell -nologo -executionpolicy bypass -File fix-vcpkg-json.ps1 -removeCUDA
cd ..

IF "%1"=="--build-doc" (
    @echo off
    sh .\build_documentation.sh
)

cd vcpkg
call bootstrap-vcpkg.bat
vcpkg.exe integrate install
cd ..

mkdir _buildNoCuda
cd _buildNoCuda
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_CUDA=OFF -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base 
cmake --build . --config RelWithDebInfo
cd ..

rem goto :EOF
mkdir _debugbuildNoCuda
cd _debugbuildNoCuda
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=OFF -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base 
cmake --build . --config Debug