#pragma once

#include <linux/videodev2.h>
#include <cstdint>

class AV4L2PlaneInfo
{
public:
    AV4L2PlaneInfo() : fd(-1), data(nullptr) {}

public:
    int fd;
    uint8_t *data;
};

class AV4L2Buffer
{
public:
    AV4L2Buffer(uint32_t index, uint32_t type, uint32_t memType, uint32_t numPlanes);
    ~AV4L2Buffer();

    void map();
    void unmap();

    uint32_t getIndex();
    uint32_t getNumPlanes();

public:
    struct v4l2_buffer v4l2_buf;
    AV4L2PlaneInfo *planesInfo;

private:
    uint32_t mNumPlanes;
    uint32_t mIndex;
};
