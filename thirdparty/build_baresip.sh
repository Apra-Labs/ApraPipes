#!/bin/bash
sudo -v
pwd 
ls
cd baresip
cmake -B build -DCMAKE_BUILD_TYPE=Release 
cmake --build build -j
cmake --install build
cd ../..