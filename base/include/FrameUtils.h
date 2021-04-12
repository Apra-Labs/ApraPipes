#pragma once

#include "Frame.h"
#include "FrameMetadata.h"
#ifdef ARM64
#include "DMAFDWrapper.h"
#endif

class FrameUtils
{
public:
    typedef std::function<void *(frame_sp &)> GetDataPtr;

public:
    static GetDataPtr getDataPtrFunction(FrameMetadata::MemType memType)
    {
        switch (memType)
        {
#ifdef ARM64
        case FrameMetadata::MemType::DMABUF:
            return getDMAFDHostDataPtr;
#endif
        default:
            return getHostDataPtr;
        }
    }

    static void *getHostDataPtr(frame_sp &frame)
    {
        return frame->data();
    }

#ifdef ARM64
    static void *getDMAFDHostDataPtr(frame_sp &frame)
    {
        auto ptr = static_cast<DMAFDWrapper *>(frame->data());
        return ptr->getHostPtr();
    }
#endif
};