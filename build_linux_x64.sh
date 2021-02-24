mkdir -p _build
cd _build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_CUDA=ON -DENABLE_LINUX=ON../base 
cmake --build . -- -j "$(nproc)"

cd ..

mkdir -p _debugbuild
cd _debugbuild
cmake -DCMAKE_BUILD_TYPE=Debug ../base 
cmake --build . -- -j "$(nproc)"
