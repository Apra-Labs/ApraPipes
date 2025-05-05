#include "ThumbnailListGenerator.h"
#include "FrameMetadata.h"
#include "ImageMetadata.h"
#include "RawImageMetadata.h"
#include "RawImagePlanarMetadata.h"
#include "FrameMetadataFactory.h"
#include "Frame.h"
#include "Logger.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "Utils.h"
#include <vector>
#include <fstream>
#include <jpeglib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdio>
#include <exiv2/exiv2.hpp>

#include <openssl/md5.h>
#include <sstream>
#include <iomanip>

// #include <exif.h>

class ThumbnailListGenerator::Detail
{

public:
    Detail(ThumbnailListGeneratorProps &_props) : mProps(_props)
    {
        mOutSize = cv::Size(mProps.thumbnailWidth, mProps.thumbnailHeight);
        enableSOS = true;
        flags.push_back(cv::IMWRITE_JPEG_QUALITY);
        flags.push_back(90);
    }

    ~Detail() {}

    void initMatImages(framemetadata_sp &input)
    {
        mIImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(input));

        // auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(input);
        // m_height = rawMetadata->getHeight();
        // m_width = rawMetadata->getWidth();
        // m_step = rawMetadata->getStep();
    }

    void setProps(ThumbnailListGeneratorProps &props)
    {
        mProps = props;
    }

    cv::Mat mIImg;
    cv::Size mOutSize;
    bool enableSOS;
    ThumbnailListGeneratorProps mProps;
    int m_width;
    int m_height;
    int m_step;
    cv::Mat m_tempImage;
    int count = 0;
    vector<int> flags;
};

ThumbnailListGenerator::ThumbnailListGenerator(ThumbnailListGeneratorProps _props) : Module(SINK, "ThumbnailListGenerator", _props)
{
    mDetail.reset(new Detail(_props));
}

ThumbnailListGenerator::~ThumbnailListGenerator() {}

bool ThumbnailListGenerator::validateInputPins()
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
        LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
        return false;
    }

    return true;
}

bool ThumbnailListGenerator::init()
{
    if (!Module::init())
    {
        return false;
    }
    return true;
}

bool ThumbnailListGenerator::term()
{
    return Module::term();
}

// void ThumbnailListGenerator::addInputPin(framemetadata_sp &metadata, string &pinId)
// {
//     // Module::addInputPin(metadata, pinId);
//     mDetail->initMatImages(metadata); // should do inside SOS
// }

std::string ThumbnailListGenerator::calculateMD5(const unsigned char *data, size_t length)
{
    unsigned char md5Digest[MD5_DIGEST_LENGTH];
    MD5_CTX md5Context;
    MD5_Init(&md5Context);
    MD5_Update(&md5Context, data, length);
    MD5_Final(md5Digest, &md5Context);

    std::stringstream ss;
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i)
    {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)md5Digest[i];
    }
    return ss.str();
}

bool ThumbnailListGenerator::process(frame_container &frames)
{
    auto frame = getFrameByType(frames, FrameMetadata::RAW_IMAGE);
    if (isFrameEmpty(frame))
    {
        LOG_ERROR << "Got Empty Frames, returning";
        return true;
    }

    framemetadata_sp frameMeta = frame->getMetadata();
    auto rawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(frameMeta);
    auto height = rawMetadata->getHeight();
    auto width = rawMetadata->getWidth();

    unsigned char *frame_buffer = (unsigned char *)frame->data();
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    JSAMPROW row_pointer[1];
    unsigned char *jpegBuffer = nullptr; // Buffer to hold JPEG data
    unsigned long jpegSize = 0;          // Size of the JPEG data

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_mem_dest(&cinfo, &jpegBuffer, &jpegSize); // Store JPEG in memory

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 4;
    cinfo.in_color_space = JCS_EXT_RGBA;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 80, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    LOG_INFO << "Compressing image -> width<" << width << "> & height <" << height << ">";

    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = &frame_buffer[cinfo.next_scanline * width * 4];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    LOG_INFO << "JPEG Compression Done. Buffer Size: " << jpegSize;
    std::string md5Val = calculateMD5(jpegBuffer, jpegSize);
    LOG_ERROR << "================MD5 val is=====================" << md5Val;
    std::string filename = mDetail->mProps.fileToStore;
    FILE *outfile = fopen(filename.c_str(), "wb");
    if (!outfile)
    {
        LOG_ERROR << "Couldn't open file for writing: " << filename;
        free(jpegBuffer);
        return false;
    }

    fwrite(jpegBuffer, 1, jpegSize, outfile); // Write buffer to file
    fflush(outfile);
    fclose(outfile);

    LOG_INFO << "JPEG file written successfully: " << filename;

    // Free the allocated memory
    free(jpegBuffer);
    sync();
    if (m_callbackFunction)
    {
        m_callbackFunction();
    }
    return true;
}

void ThumbnailListGenerator::setProps(ThumbnailListGeneratorProps &props)
{
    Module::addPropsToQueue(props);
}

ThumbnailListGeneratorProps ThumbnailListGenerator::getProps()
{
    fillProps(mDetail->mProps);
    return mDetail->mProps;
}

bool ThumbnailListGenerator::handlePropsChange(frame_sp &frame)
{
    ThumbnailListGeneratorProps props(0, 0, "s");
    bool ret = Module::handlePropsChange(frame, props);
    mDetail->setProps(props);
    return ret;
}

void ThumbnailListGenerator::decompressFrame()
{
    FILE *file = fopen(mDetail->mProps.fileToStore.c_str(), "rb");
	if (!file)
	{
		std::cerr << "Error: Could not open the image file." << std::endl;
		perror("fopen error");
		return;
	}

	jpeg_decompress_struct jpegInfo;
	jpeg_error_mgr jpegErr;
	jpegInfo.err = jpeg_std_error(&jpegErr);
	jpeg_create_decompress(&jpegInfo);
	jpeg_stdio_src(&jpegInfo, file);
	jpeg_read_header(&jpegInfo, TRUE);

	jpegInfo.do_fancy_upsampling = FALSE;
	jpegInfo.do_block_smoothing = FALSE;

	jpeg_start_decompress(&jpegInfo);

	int32_t width = jpegInfo.output_width;
	int32_t height = jpegInfo.output_height;
	int32_t channels = 4;

	std::vector<unsigned char> imageData;
    imageData.resize(width * height * channels);
	unsigned char *rowPtr = imageData.data();
	while (jpegInfo.output_scanline < jpegInfo.output_height)
	{
		rowPtr = &imageData[jpegInfo.output_scanline * width * channels];
		if (rowPtr && &jpegInfo)
		{
			jpeg_read_scanlines(&jpegInfo, &rowPtr, 1);
		}
	}
    m_frameBuffer = imageData;
    if (m_callbackFunction)
    {
        m_callbackFunction();
    }
    fflush(file);
	jpeg_finish_decompress(&jpegInfo);
	jpeg_destroy_decompress(&jpegInfo); 
	fclose(file);
    sync();
    try
	{
        LOG_ERROR << "For " << mDetail->mProps.fileToStore.c_str() << ", Custom Metadata -> " << m_customMetadata.c_str();
		Exiv2::Image::AutoPtr mediaFile = Exiv2::ImageFactory::open(mDetail->mProps.fileToStore);
		if (!mediaFile.get() || m_customMetadata.empty())
		{
			std::cerr << "Could not open media file for EXIF modification."
					<< std::endl;
			return;
		}

		mediaFile->readMetadata();
		Exiv2::ExifData &exifData = mediaFile->exifData();

		exifData["Exif.Photo.UserComment"] = m_customMetadata;
		mediaFile->setExifData(exifData);
		mediaFile->writeMetadata();

        std::cout << "Hash saved in EXIF metadata successfully."
                << std::endl;
	} catch (Exiv2::Error &e)
	{
		std::cerr << "Error writing EXIF metadata: " << e.what() << std::endl;
	}
    // OPen Give Frame Buffer And Size to XCIF library, 
    // Calculate Hash
    // Store it in XCIF
}

std::vector<unsigned char> ThumbnailListGenerator::getFrameBuffer()
{
    return m_frameBuffer;
}

void ThumbnailListGenerator::setMetadata(std::string data)
{
    LOG_ERROR << "setCustomMetadata called -> "<< data.c_str();
    m_customMetadata = data;
}