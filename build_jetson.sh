sudo apt-get install clang-format
clang-format -style=llvm -dump-config > .clang-format
if ! command -v pip &> /dev/null; then
    # If pip is not available, download and install pip
    curl -O https://bootstrap.pypa.io/get-pip.py
    python3 get-pip.py
fi
pip install pre-commit
pre-commit install

chmod +x build_scripts/build_dependencies_jetson_cuda.sh
sudo ./build_scripts/build_dependencies_jetson_cuda.sh

if nvcc --version; then
  USER_NAME=$(whoami)
  TARGET_USER="$USER_NAME"
  TARGET_HOME=$(eval echo ~$TARGET_USER)

  # Append lines to the target user's ~/.bashrc
  if ! grep -qxF 'export VCPKG_FORCE_SYSTEM_BINARIES=1' $TARGET_HOME/.bashrc; then
    echo 'export VCPKG_FORCE_SYSTEM_BINARIES=1' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
    echo "VCPKG_FORCE_SYSTEM_BINARIES flag added in .bashrc"
  else
    echo "VCPKG_FORCE_SYSTEM_BINARIES flag already exists in .bashrc"
  fi

  if ! grep -qxF 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' $TARGET_HOME/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
    echo "CUDA Binary Path added to .bashrc"
  else
    echo "CUDA Binary Path already exists in .bashrc"
  fi

  if ! grep -qxF 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' $TARGET_HOME/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
    echo "CUDA Library Path added to .bashrc"
  else
    echo "CUDA Library Path already exists in .bashrc"
  fi
  
  echo "Appended paths to ~/.bashrc and saved changes."
  source ~/.bashrc
  echo "Reloaded ~/.bashrc"
fi

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
export VCPKG_FORCE_SYSTEM_BINARIES=1 && export VCPKG_OVERLAY_PORTS=../thirdparty/custom-overlay && cmake -B . -DENABLE_ARM64=ON -DENABLE_WINDOWS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$(($(nproc) - 1))"
