copy base\vcpkg.nocuda.json base\vcpkg.json
cd vcpkg
call bootstrap-vcpkg.bat
vcpkg.exe integrate install
cd ..

mkdir _buildNoCuda
cd _buildNoCuda
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_CUDA=OFF -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base 
cmake --build .
cd ..

rem goto :EOF
mkdir _debugbuildNoCuda
cd _debugbuildNoCuda
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=OFF -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base 
cmake --build .