mkdir -p _build
cd _build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_CUDA=OFF  ../base 
cmake --build . -- -j "$(($(nproc) - 1))"
cd ..

mkdir -p _debugbuild
cd _debugbuild
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=OFF ../base 
cmake --build . -- -j "$(($(nproc) - 1))"
