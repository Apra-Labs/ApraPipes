#include "ApraLines.h"
#include "AIPExceptions.h"

ApraLines::ApraLines(void *buffer, size_t size)
{
    count = static_cast<int>(size >> 4);
    lines = reinterpret_cast<cv::Vec4i*>(buffer);
}

int ApraLines::size()
{
    return count;
}

cv::Vec4i& ApraLines::operator [](int i)
{
    if(i >= count)
    {
        throw AIPException(AIP_PARAM_OUTOFRANGE, "out of range access");
    }

    return lines[i];
}