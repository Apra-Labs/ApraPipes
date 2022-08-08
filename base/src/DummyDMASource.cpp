#include "DummyDMASource.h"
#include "DMAAllocator.h"
#include "FrameMetadata.h"
#include <iostream>
#include <opencv2/opencv.hpp>

DummyDMASource::DummyDMASource(DummyDMASourceProps props)
	: Module(SOURCE, "DummyDMASource", props), mProps(props)
{
    auto outputMetadata = framemetadata_sp(new RawImageMetadata(FrameMetadata::MemType::DMABUF));
    DMAAllocator::setMetadata(outputMetadata, static_cast<int>(mProps.width), static_cast<int>(mProps.height), ImageMetadata::ImageType::RGBA);
	mOutputPinId = addOutputPin(outputMetadata);	
}

DummyDMASource::~DummyDMASource() {}

bool DummyDMASource::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "1 output pin expected";
		return false;
	}

	return true;
}

bool DummyDMASource::init()
{
	if (!Module::init())
	{
		return false;
	}
	return true;
}

bool DummyDMASource::term()
{
	return  Module::term();
}

// bool DummyDMASource::produce()
// {
// 	frame_container frames;
//     auto frame = makeFrame();
//     cv::Mat image = imread(mProps.fileName, cv::IMREAD_GRAYSCALE);
// 	// frame->data() = image->data();
// 	// memcpy(frame->data(), image.data, static_cast<size_t>(image.size));
//     outputMetadata1 = framemetadata_sp(new RawImagePlanarMetadata(FrameMetadata::MemType::DMABUF));
// 	cv::Mat oImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(mOutputMetadata));
//     frame->data() = static_cast<void *>(image.data);
//     frames.insert(make_pair(mOutputPinId, frame));
//     send(frames);
// 	return true;
// }

bool DummyDMASource::produce()
{
	frame_container frames;
    auto frame = makeFrame();
    cv::Mat image = imread("./data/globe-scene-fish-bowl-pngcrush.png", cv::IMREAD_COLOR);
	auto size1 = image.step[0] * image.rows;
    LOG_ERROR << size1;
    memcpy(frame->data(), image.data, static_cast<size_t>(size1));
    
    frames.insert(make_pair(mOutputPinId, frame));
    send(frames);
	return true;
}