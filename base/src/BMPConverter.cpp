#include "BMPConverter.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"

#include "opencv2/core.hpp"
#include <memory>

// https://stackoverflow.com/a/23303847/1694369

#pragma pack(push,1)

struct FileHeader
{
    uint8_t signature[2];
    uint32_t filesize;
    uint32_t reserved;
    uint32_t fileoffset_to_pixelarray;
};

struct BitmapInfoHeader
{
    uint32_t dibheadersize;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bitsperpixel;
    uint32_t compression;
    uint32_t imagesize;
    uint32_t ypixelpermeter;
    uint32_t xpixelpermeter;
    uint32_t numcolorspallette;
    uint32_t mostimpcolor;
};

struct BitmapHeader
{
    FileHeader fileHeader;
    BitmapInfoHeader infoHeader;
};

#pragma pack(pop)

class BMPConverter::Detail
{
public:
    Detail(BMPConverterProps &_props) : props(_props), outSize(0), width(0), height(0), bitmapHeaderSize(sizeof(BitmapHeader)), imageSize(0)
    {
    }

    ~Detail()
    {
    }

    void getImageSize(int &_width, int &_height)
    {
        _width = width;
        _height = height;
    }

    void setMetadata(framemetadata_sp &metadata)
    {
        auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
        width = rawImageMetadata->getWidth();
        height = rawImageMetadata->getHeight();

        if(rawImageMetadata->getImageType() != ImageMetadata::RGB)
        {
            throw AIPException(AIP_NOTIMPLEMENTED, "ImageType is expected to be RGB. Actual<" + std::to_string(rawImageMetadata->getImageType()) + ">");
        }

        if(rawImageMetadata->getStep() != width*3)
        {
            throw AIPException(AIP_FATAL, "step is expected to be width*3. width<" + std::to_string(width) + "> step<" + std::to_string(rawImageMetadata->getStep()) + ">");
        }

        if(width % 4 != 0)
        {
            throw AIPException(AIP_FATAL, "width is expected to be multiple of 4. width<" + std::to_string(width) + ">");
        }

        inImage = Utils::getMatHeader(rawImageMetadata);
        outImage = Utils::getMatHeader(rawImageMetadata);

        imageSize = rawImageMetadata->getDataSize();
        outSize =  bitmapHeaderSize + imageSize;

        bitmapHeader = std::make_unique<char[]>(bitmapHeaderSize);
        auto pBitmapHeader = reinterpret_cast<BitmapHeader *>(bitmapHeader.get());

        pBitmapHeader->fileHeader.signature[0] = 0x42; //B
        pBitmapHeader->fileHeader.signature[1] = 0x4d; //M
        pBitmapHeader->fileHeader.filesize = outSize;
        pBitmapHeader->fileHeader.fileoffset_to_pixelarray = bitmapHeaderSize;
        pBitmapHeader->infoHeader.dibheadersize = sizeof(BitmapInfoHeader);
        pBitmapHeader->infoHeader.width = width;
        pBitmapHeader->infoHeader.height = -1 * height; // for top-down bitmap height must be negative
        pBitmapHeader->infoHeader.planes = 1;
        pBitmapHeader->infoHeader.bitsperpixel = 24;
        pBitmapHeader->infoHeader.compression = 0;
        pBitmapHeader->infoHeader.imagesize = imageSize;
        pBitmapHeader->infoHeader.ypixelpermeter = 0;
        pBitmapHeader->infoHeader.xpixelpermeter = 0;
        pBitmapHeader->infoHeader.numcolorspallette = 0;
    }

    bool compute(void *buffer, void *outBuffer)
    {
        memcpy(outBuffer, bitmapHeader.get(), bitmapHeaderSize);

        auto imageStart = static_cast<uint8_t*>(outBuffer) + bitmapHeaderSize;        
        inImage.data = static_cast<uint8_t*>(buffer);
        outImage.data = imageStart;
        cv::cvtColor(inImage, outImage, cv::COLOR_RGB2BGR);

        return true;
    }


    size_t outSize;
private:
    BMPConverterProps props;

    std::unique_ptr<char[]> bitmapHeader;
    size_t bitmapHeaderSize;
    size_t imageSize;

    cv::Mat inImage;
    cv::Mat outImage;

    int width;
    int height;
};

BMPConverter::BMPConverter(BMPConverterProps _props) : Module(TRANSFORM, "BMPConverter", _props)
{
    mDetail.reset(new Detail(_props));
    mOutputMetadata = framemetadata_sp(new FrameMetadata(FrameMetadata::BMP_IMAGE));
    mOutputPinId = addOutputPin(mOutputMetadata);
}

BMPConverter::~BMPConverter() {}

bool BMPConverter::validateInputPins()
{
    if (getNumberOfInputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstInputMetadata();
    FrameMetadata::FrameType frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::RAW_IMAGE)
    {
        LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE of type RGB. Actual FrameType<" << frameType << ">";
        return false;
    }

    return true;
}

bool BMPConverter::validateOutputPins()
{
    if (getNumberOfOutputPins() != 1)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfOutputPins() << ">";
        return false;
    }

    framemetadata_sp metadata = getFirstOutputMetadata();
    FrameMetadata::FrameType frameType = metadata->getFrameType();
    if (frameType != FrameMetadata::BMP_IMAGE)
    {
        LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be BMP_IMAGE. Actual<" << frameType << ">";
        return false;
    }

    return true;
}

bool BMPConverter::init()
{
    if (!Module::init())
    {
        return false;
    }

    auto metadata = getFirstInputMetadata();
    if (metadata->isSet())
    {
        mDetail->setMetadata(metadata);
    }

    return true;
}

bool BMPConverter::term()
{
    return Module::term();
}

bool BMPConverter::process(frame_container &frames)
{
    auto frame = frames.cbegin()->second;
    auto outFrame = makeFrame(mDetail->outSize, mOutputMetadata);

    mDetail->compute(frame->data(), outFrame->data());

    frames.insert(make_pair(mOutputPinId, outFrame));
    send(frames);

    return true;
}

bool BMPConverter::processSOS(frame_sp &frame)
{
    auto metadata = frame->getMetadata();
    mDetail->setMetadata(metadata);

    return true;
}

bool BMPConverter::shouldTriggerSOS()
{
    return mDetail->outSize == 0;
}

bool BMPConverter::processEOS(string &pinId)
{
    mDetail->outSize = 0;
    return true;
}

void BMPConverter::getImageSize(int &width, int &height)
{
    mDetail->getImageSize(width, height);
}