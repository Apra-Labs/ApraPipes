cd vcpkg
call bootstrap-vcpkg.bat
vcpkg.exe integrate install
cd ..

mkdir _build
cd _build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_CUDA=ON -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base 
cmake --build .
cd ..
pause

@REM mkdir _debugbuild
@REM cd _debugbuild
@REM cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=ON -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base 
@REM cmake --build .
@REM pause