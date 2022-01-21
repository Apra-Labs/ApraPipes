CMAKE_THCOUNT=$(sh ./checkProc.sh)
mkdir -p _build
cd _build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../base 
cmake --build . -- -j "$CMAKE_THCOUNT"

cd ..

mkdir -p _debugbuild
cd _debugbuild
cmake -DCMAKE_BUILD_TYPE=Debug ../base 
cmake --build . -- -j "$CMAKE_THCOUNT"