apt-get install gcc-8 g++-8
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 100
update-alternatives --config gcc