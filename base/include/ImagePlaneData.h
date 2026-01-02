#pragma once

#include <cstddef>
#include <functional>
#include <vector>
#include <memory>
#include <cstring>

class ImagePlaneData
{
public:
    ImagePlaneData(size_t _size, size_t _step, size_t _rowSize, int _width, int _height) : size(_size),
                                                                                           step(_step),
                                                                                           rowSize(_rowSize),
                                                                                           width(_width),
                                                                                           height(_height)
    {
        imageSize = rowSize*height;

        if (step == rowSize)
        {
            mCopyToData = copyFromImagePlane;
        }
        else
        {
            mCopyToData = copyFromImagePlaneByLine;
        }
    }

public:
    static void copyFromImagePlane(ImagePlaneData *plane, void *dst)
    {
        memcpy(dst, plane->data, plane->imageSize);
    }

    static void copyFromImagePlaneByLine(ImagePlaneData *plane, void *dst)
    {
        auto dstPtr = static_cast<uint8_t *>(dst);
        auto srcPtr = static_cast<uint8_t *>(plane->data);
        for (auto i = 0; i < plane->height; i++)
        {
            memcpy(dstPtr, srcPtr, plane->rowSize);
            dstPtr += plane->rowSize;
            srcPtr += plane->step;
        }
    }

public:
    typedef std::function<void(ImagePlaneData *, void *)> Copy;

public:
    void *data;
    size_t size;
    size_t imageSize;
    size_t step;
    size_t rowSize;
    int width;
    int height;

    Copy mCopyToData;
};

typedef std::vector<std::shared_ptr<ImagePlaneData>> ImagePlanes;