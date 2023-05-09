#!/bin/bash
sudo -v
pwd 
ls
cd lib_re    
cmake -B build -DCMAKE_BUILD_TYPE=Release 
cmake --build build -j
cmake --install build
ldconfig
cd ../..