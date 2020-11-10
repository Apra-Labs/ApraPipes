#pragma once

#include <opencv2/core/types_c.h>
#include "FrameMetadata.h"

class ApraLines
{
public:
    // no lines - count 0 should also work
    ApraLines(void *buffer, size_t size);

    int size();

    /*! element access */
    cv::Vec4i &operator[](int i);

protected:
    int count;
    cv::Vec4i *lines;
};

class ApraLinesMetadata : public FrameMetadata
{
public:
    ApraLinesMetadata() : FrameMetadata(FrameType::APRA_LINES) {}

    framemetadata_sp getParentMetadata()
    {
        return parentMetadata;
    }

    void setParentMetadata(framemetadata_sp &metadata)
    {
        parentMetadata = metadata;
    }

private:
    framemetadata_sp parentMetadata;
};