#!/bin/bash
pwd 
ls
cd lib_re
cmake -B build -DCMAKE_BUILD_TYPE=Release 
cmake --build build -j
sudo cmake --install build
sudo ldconfig
cd ../..