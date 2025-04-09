# Use an official Ubuntu image as the base
FROM ubuntu:20.04

# Disable interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install build tools and dependencies
RUN apt-get update
RUN apt-get install -y dos2unix curl zip unzip tar git autoconf \
    python3 automake autopoint build-essential \
    flex git-core git-lfs libass-dev libfreetype6-dev libgnutls28-dev libmp3lame-dev \
    libsdl2-dev libssl-dev libtool libsoup-gnome2.4-dev libncurses5-dev libva-dev \
    libvdpau-dev libvorbis-dev libxcb1-dev libxdamage-dev libxcursor-dev libxinerama-dev \
    libx11-dev libgles2-mesa-dev libxcb-shm0-dev libxcb-xfixes0-dev pkg-config \
    texinfo wget yasm zlib1g-dev nasm gperf bison python3-pip doxygen graphviz \
    libxi-dev libgl1-mesa-dev libglu1-mesa-dev mesa-common-dev libxrandr-dev libxxf86vm-dev \
    libxtst-dev libudev-dev libgl1-mesa-dev python3-jinja2 nlohmann-json3-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \    
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \   
    gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa \
    gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio libgstrtspserver-1.0-dev gstreamer1.0-rtsp \
    sudo \
    python3 \
    python3-pip \
    keyboard-configuration \
    console-setup \
 && rm -rf /var/lib/apt/lists/*
 RUN wget https://github.com/Kitware/CMake/releases/download/v3.23.0/cmake-3.23.0-linux-x86_64.sh -O /tmp/cmake-install.sh && \
 chmod +x /tmp/cmake-install.sh && \
 /tmp/cmake-install.sh --skip-license --prefix=/usr/local && \
 rm /tmp/cmake-install.sh

RUN pip3 install meson
# Set the working directory inside the container
WORKDIR /app

# Copy the entire repository into the container
COPY . /app

# Ensure the build script has execute permissions
RUN chmod +x build_linux_no_cuda_docker.sh \
    && chmod +x build_scripts/build_dependencies_linux_no_cuda.sh \
    && chmod +x base/fix-vcpkg-json.sh

# Execute the build script to generate the base image
RUN bash ./build_linux_no_cuda_docker.sh

# Optionally, specify a default command or entrypoint if needed
CMD ["bash"]
