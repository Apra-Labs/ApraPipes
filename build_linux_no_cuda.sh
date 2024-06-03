sudo apt-get install clang-format
clang-format -style=llvm -dump-config > .clang-format
if ! command -v pip &> /dev/null; then
    # If pip is not available, download and install pip
    curl -O https://bootstrap.pypa.io/get-pip.py
    python3 get-pip.py
fi
pip install pre-commit
pre-commit install

chmod +x build_scripts/build_dependencies_linux_no_cuda.sh
sudo ./build_scripts/build_dependencies_linux_no_cuda.sh

chmod +x base/fix-vcpkg-json.sh
./base/fix-vcpkg-json.sh true false false

if [[ $1 == "--build-doc" ]]; then
    chmod +x build_documentation.sh
    ./build_documentation.sh
fi

cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
cd ..

CMAKE_THCOUNT=$(sh ./checkProc.sh)
mkdir -p _build
cd _build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_CUDA=OFF  ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$CMAKE_THCOUNT"
cd ..

mkdir -p _debugbuild
cd _debugbuild
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=OFF ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$CMAKE_THCOUNT"
