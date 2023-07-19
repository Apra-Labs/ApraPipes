#include <cstdint>
#include "opencv2/opencv.hpp"
#include <boost/foreach.hpp>
#include "OverlayModuleDec.h"
#include "Utils.h"
#include "FrameContainerQueue.h"
#include "RawImageMetadata.h"
#include "Overlay.h"
#include "CudaCommon.h"
#include "DMAFDWrapper.h"

OverlayModuleDec::OverlayModuleDec(OverlayModuleDecProps _props) : Module(TRANSFORM, "OverlayModuleDec", _props) {}

void OverlayModuleDec::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	if (metadata->getFrameType() == FrameMetadata::RAW_IMAGE)
	//mOutputPinId = addOutputPin(metadata);
	{
	{
		auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		auto mOutputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(1280, 720, ImageMetadata::BGRA, CV_8UC4, 0, CV_8U, FrameMetadata::DMABUF, true));
		mOutputPinId = addOutputPin(mOutputMetadata);
	}
	}
}


bool OverlayModuleDec::init()
{
	return Module::init();
}

bool OverlayModuleDec::term()
{
	return Module::term();
}

bool OverlayModuleDec::validateInputPins()
{
	pair<string, framemetadata_sp> me; // map element	
	auto inputMetadataByPin = getInputMetadata();
	BOOST_FOREACH(me, inputMetadataByPin)
	{
		FrameMetadata::FrameType frameType = me.second->getFrameType();
		if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::OVERLAY_INFO_IMAGE && frameType != FrameMetadata::H264_DATA)
		{
			LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE OR OVERLAY_INFO_IMAGE. Actual<" << frameType << ">";
			return false;
		}
	}
	return true;
}

bool OverlayModuleDec::validateOutputPins()
{
	auto outputMetadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = outputMetadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}
	return true;
}

bool OverlayModuleDec::shouldTriggerSOS()
{
	return true;
}

bool OverlayModuleDec::process(frame_container& frames)
{
	DrawingOverlay drawOverlay;
	for (auto it = frames.cbegin(); it != frames.cend(); it++)
	{
		auto metadata = it->second->getMetadata();
		auto frameType = metadata->getFrameType();
		frame_sp overlayframe = it->second;

		if (frameType == FrameMetadata::OVERLAY_INFO_IMAGE)
		{
			drawOverlay.deserialize(overlayframe);
		}
	}
	for (auto it = frames.cbegin(); it != frames.cend(); it++)
	{
		auto metadata = it->second->getMetadata();
		auto frameType = metadata->getFrameType();
		frame_sp frame = it->second;

		if (frameType == FrameMetadata::RAW_IMAGE)
		{
			//drawOverlay.draw(frame);
			auto outFrame = makeFrame(); // DMABUF
			auto dstPtr =  static_cast<DMAFDWrapper *>(outFrame->data())->getCudaPtr(); 
			
			auto inputMetadata = boost::shared_ptr<FrameMetadata>(new RawImageMetadata(1280, 720, ImageMetadata::BGR, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
			drawOverlay.draw(frame);

			
			auto input = FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata);
			cv::Mat matInputImg =  Utils::getMatHeader(input);

			matInputImg.data = static_cast<uchar*>(frame->data());
		
			cv::Mat bgraImg;
			cv::cvtColor(matInputImg, bgraImg, cv::COLOR_BGR2BGRA);
			//size_t bgraSize = reinterpret_cast<size_t>(bgraImg.);
			 auto stream = cudastream_sp(new ApraCudaStream);
			// void * tempHostPtr = (void*)malloc(outFrame->size()); 
			cudaMemcpyAsync(dstPtr, bgraImg.data, outFrame->size(), cudaMemcpyHostToDevice, stream->getCudaStream());
			//
			frame_container overlayConatiner;
			overlayConatiner.insert(make_pair(mOutputPinId, outFrame));
			send(overlayConatiner);
		}
	}
	return true;
}
