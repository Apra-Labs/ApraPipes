#include <cstdint>
#include <boost/foreach.hpp>
extern "C"
{
#include <libavutil/motion_vector.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}
#include "OverlayMotionVectors.h"
#include "Utils.h"

class OverlayDetailAbs
{
public:
	OverlayDetailAbs()
	{
	};
	~OverlayDetailAbs(){ }

	void setMetadata(RawImageMetadata* rawMetadata)
	{
		mImg = Utils::getMatHeader(rawMetadata);
		RawImageMetadata outputMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), ImageMetadata::BGR, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true);
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(rawOutputMetadata);
		rawOutMetadata->setData(outputMetadata);
	}

	virtual bool overlayMotionVectors(frame_container frames, frame_sp& outFrame) = 0;

public:
	framemetadata_sp rawOutputMetadata;
	cv::Mat mImg;
};

class DetailFFmpeg : public OverlayDetailAbs
{
public:
	DetailFFmpeg() {}
	~DetailFFmpeg() {}

	bool overlayMotionVectors(frame_container frames, frame_sp& outFrame);
};
class DetailOpenh264 : public OverlayDetailAbs
{
public:
	DetailOpenh264() {}
	~DetailOpenh264() {}

	bool overlayMotionVectors(frame_container frames, frame_sp& outFrame);

};

bool DetailFFmpeg::overlayMotionVectors(frame_container frames, frame_sp& outFrame)
{
	auto inRawImageFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::RAW_IMAGE);
	auto inMotionVectorFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::MOTION_VECTOR_DATA);

	if (inMotionVectorFrame->size() == 0 || !inRawImageFrame)
	{
		return false;
	}

	AVMotionVector* motionVectors = (AVMotionVector*)inMotionVectorFrame->data();
	mImg.data = static_cast<uint8_t*>(inRawImageFrame->data());
	for (int i = 0; i < inMotionVectorFrame->size() / sizeof(*motionVectors); i++)
	{
		AVMotionVector* MV = &motionVectors[i];

		if (std::abs(MV->motion_x) > 2 || std::abs(MV->motion_y) > 2)
		{
			cv::arrowedLine(mImg, cv::Point(int(MV->src_x), int(MV->src_y)), cv::Point(int(MV->dst_x), int(MV->dst_y)), cv::Scalar(0, 255, 0), 1); // arrowedLine will help to also show the direction of motion.
		}
	}
	outFrame = inRawImageFrame;
	return true;
}

bool DetailOpenh264::overlayMotionVectors(frame_container frames, frame_sp& outFrame)
{
	auto inRawImageFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::RAW_IMAGE);
	auto inMotionVectorFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::MOTION_VECTOR_DATA);

	if (!inMotionVectorFrame || !inRawImageFrame)
	{
		return false;
	}

	auto motionVectorData = static_cast<uint16_t*>(inMotionVectorFrame->data());

	mImg.data = static_cast<uint8_t*>(inRawImageFrame->data());

	for (int i = 0; i < inMotionVectorFrame->size(); i++)
	{
		auto motionX = motionVectorData[i];
		auto motionY = motionVectorData[i + 1];
		auto xOffset = motionVectorData[i + 2];
		auto yOffset = motionVectorData[i + 3];
		if(abs(motionX) > 3 || abs(motionY) > 3)
		cv::circle(mImg, cv::Point(int(xOffset), int(yOffset)), 1, cv::Scalar(0, 255, 0));
	}
	outFrame = inRawImageFrame;
	return true;
}

OverlayMotionVector::OverlayMotionVector(OverlayMotionVectorProps props) : Module(TRANSFORM, "OverlayMotionVectors", props)
{
	if (props.MVOverlay == OverlayMotionVectorProps::MVOverlayMethod::FFMPEG)
	{
		mDetail.reset(new DetailFFmpeg());
	}
	else if (props.MVOverlay == OverlayMotionVectorProps::MVOverlayMethod::OPENH264)
	{
		mDetail.reset(new DetailOpenh264());
	}
	mDetail->rawOutputMetadata = framemetadata_sp(new RawImageMetadata());
	mOutputPinId = Module::addOutputPin(mDetail->rawOutputMetadata);
}

bool OverlayMotionVector::init()
{
	return Module::init();
}

bool OverlayMotionVector::term()
{
	return Module::term();
}

bool OverlayMotionVector::validateInputPins()
{
	if (getNumberOfInputPins() > 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 2. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	pair<string, framemetadata_sp> me; // map element	
	auto inputMetadataByPin = getInputMetadata();
	BOOST_FOREACH(me, inputMetadataByPin)
	{
		FrameMetadata::FrameType frameType = me.second->getFrameType();
		if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::MOTION_VECTOR_DATA)
		{
			LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE OR MOTION_VECTOR_DATA. Actual<" << frameType << ">";
			return false;
		}
	}
	return true;
}

bool OverlayMotionVector::validateOutputPins()
{
	if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	auto outputMetadata = getFirstOutputMetadata();
	FrameMetadata::FrameType frameType = outputMetadata->getFrameType();
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}
	return true;
}

bool OverlayMotionVector::shouldTriggerSOS()
{
	return true;
}

bool OverlayMotionVector::process(frame_container& frames)
{
	frame_sp outFrame;
	if (mDetail->overlayMotionVectors(frames, outFrame))
	{
		frames.insert(make_pair(mOutputPinId, outFrame));
		send(frames);
	}
	return true;
}

bool OverlayMotionVector::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	if (metadata->getFrameType() == FrameMetadata::RAW_IMAGE)
	{
		mDetail->setMetadata(FrameMetadataFactory::downcast<RawImageMetadata>(metadata));
	}
	return true;
}