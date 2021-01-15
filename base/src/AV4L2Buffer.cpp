#include "AV4L2Buffer.h"
#include "AIPExceptions.h"

#include <sys/mman.h>

AV4L2Buffer::AV4L2Buffer(uint32_t index, uint32_t type, uint32_t memType, uint32_t numPlanes) : mNumPlanes(numPlanes)
{
    memset(&v4l2_buf, 0, sizeof(struct v4l2_buffer));
    v4l2_buf.index = index;
    v4l2_buf.type = type;
    v4l2_buf.memory = memType;
    v4l2_buf.length = numPlanes;

    v4l2_buf.m.planes = new struct v4l2_plane[numPlanes];
    for (auto i = 0; i < numPlanes; i++)
    {
        memset(&v4l2_buf.m.planes[i], 0, sizeof(struct v4l2_plane));
    }
    planesInfo = new AV4L2PlaneInfo[numPlanes];
}

AV4L2Buffer::~AV4L2Buffer()
{
    delete[] planesInfo;
    delete[] v4l2_buf.m.planes;
}

void AV4L2Buffer::map()
{
    auto planes = v4l2_buf.m.planes;
    for (auto j = 0; j < mNumPlanes; j++)
    {
        planesInfo[j].data = (uint8_t *)mmap(NULL,
                                             planes[j].length,
                                             PROT_READ | PROT_WRITE,
                                             MAP_SHARED,
                                             planesInfo[j].fd,
                                             planes[j].m.mem_offset);

        if (planesInfo[j].data == MAP_FAILED)
        {
            throw AIPException(AIP_FATAL, "Could not map buffer ");
        }
    }
}

void AV4L2Buffer::unmap()
{
    auto planes = v4l2_buf.m.planes;
    for (uint32_t j = 0; j < mNumPlanes; j++)
    {
        if (planesInfo[j].data)
        {
            munmap(planesInfo[j].data, planes[j].length);
        }
        planesInfo[j].data = nullptr;
    }
}

uint32_t AV4L2Buffer::getIndex()
{
    return v4l2_buf.index;
}

uint32_t AV4L2Buffer::getNumPlanes()
{
    return mNumPlanes;
}