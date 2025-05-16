sudo apt-get install clang-format
clang-format -style=llvm -dump-config > .clang-format
if ! command -v pip &> /dev/null; then
    # If pip is not available, download and install pip
    curl -O https://bootstrap.pypa.io/get-pip.py
    python3 get-pip.py
fi
pip install pre-commit
pre-commit install

chmod +x build_scripts/build_dependencies_linux_cuda.sh
sudo ./build_scripts/build_dependencies_linux_cuda.sh

if ! sudo nvcc --version &>/dev/null; then
  USER_NAME=$(whoami)
  cudnn_archives="/home/$USER_NAME/Downloads/cudnn-*.tar.xz"

  for archive in $cudnn_archives; do
    if [ -e "$archive" ]; then
      extracted_folder="/home/$USER_NAME/Downloads/$(basename "$archive" .tar.xz)"
      if [ ! -d "$extracted_folder" ]; then
        echo "Extracting $archive..."
        tar xf "$archive" -C "/home/$USER_NAME/Downloads/"
      else
        echo "Archive already extracted: $extracted_folder"
      fi
    fi
  done

  echo "Copying files..."
  sudo cp -r /home/$USER_NAME/Downloads/cudnn-*/include/* /usr/local/cuda/include/
  sudo cp -r /home/$USER_NAME/Downloads/cudnn-*/lib/* /usr/local/cuda/lib64/

  TARGET_USER="$USER_NAME"
  TARGET_HOME=$(eval echo ~$TARGET_USER)

  # Append lines to the target user's ~/.bashrc
  if ! grep -qxF 'export PATH=/usr/local/cuda/bin:${PATH}' $TARGET_HOME/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin:${PATH}' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
    echo "CUDA Binary Path added to .bashrc"
  else
    echo "CUDA Binary Path already exists in .bashrc"
  fi

  if ! grep -qxF 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' $TARGET_HOME/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' | sudo -u $TARGET_USER tee -a $TARGET_HOME/.bashrc
    echo "CUDA Library Path added to .bashrc"
  else
    echo "CUDA Library Path already exists in .bashrc"
  fi

  # Reload .bashrc
  source ~/.bashrc

  echo "Appended line to ~/.bashrc and saved changes."
  echo "Reloaded ~/.bashrc"
fi

if [[ $1 == "--build-doc" ]]; then
    chmod +x build_documentation.sh
    ./build_documentation.sh
fi

cd vcpkg
./bootstrap-vcpkg.sh
vcpkg integrate install
cd ..

CMAKE_THCOUNT=$(sh ./checkProc.sh)
mkdir -p _build
cd _build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$CMAKE_THCOUNT"

cd ..

mkdir -p _debugbuild
cd _debugbuild
cmake -DCMAKE_BUILD_TYPE=Debug ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$CMAKE_THCOUNT"
