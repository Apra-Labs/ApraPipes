#!/bin/bash
pwd 
ls
cd baresip
cmake -B build -DCMAKE_BUILD_TYPE=Release 
cmake --build build -j
sudo cmake --install build
cd ../..