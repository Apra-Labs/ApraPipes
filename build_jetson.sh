mkdir -p _build
cd _build
cmake -DENABLE_ARM64=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo ../base 
cmake --build .