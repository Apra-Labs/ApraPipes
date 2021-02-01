mkdir -p _build
cd _build
# cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DENABLE_ARM64=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo ../base 
cmake -DENABLE_ARM64=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo ../base 
cmake --build . -- -j "$(nproc)"