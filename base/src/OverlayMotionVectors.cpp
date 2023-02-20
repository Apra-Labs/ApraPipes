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

class OverlayMotionVector::Detail
{
public:
	Detail(OverlayMotionVectorProps props)
	{
	};
	~Detail()
	{
	}

	void setMatImg(RawImageMetadata* rawMetadata)
	{
		mImg = Utils::getMatHeader(rawMetadata);
	}

	void overlayMotionVectors(frame_container frames)
	{
		auto inRawImageFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::RAW_IMAGE);
		auto inMotionVectorFrame = Module::getFrameByType(frames, FrameMetadata::FrameType::MOTION_VECTOR_DATA);

		if (inMotionVectorFrame->size() == 0 || inRawImageFrame->size() == 0)
		{
			return;
		}

		AVMotionVector* motionVectors = (AVMotionVector*)inMotionVectorFrame->data();
		mImg.data = static_cast<uint8_t*>(inRawImageFrame->data());
		for (int i = 0; i < inMotionVectorFrame->size() / sizeof(*motionVectors); i++)
		{
			AVMotionVector* MV = &motionVectors[i];

			if (MV->motion_x > 2 || MV->motion_y > 2)
			{
				cv::arrowedLine(mImg, cv::Point(int(MV->src_x), int(MV->src_y)), cv::Point(int(MV->dst_x), int(MV->dst_y)), cv::Scalar(0, 255, 0), 1); // arrowedLine will help to also show the direction of motion.
			}
		}
		cv::imshow("frame", mImg);
		cv::waitKey(1);
	}

private:
	cv::Mat mImg;
};


OverlayMotionVector::OverlayMotionVector(OverlayMotionVectorProps props) : Module(SINK, "OverlayMotionVectors", props)
{
	mDetail.reset(new Detail(props));
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

bool OverlayMotionVector::shouldTriggerSOS()
{
	return true;
}

bool OverlayMotionVector::process(frame_container& frames)
{
	mDetail->overlayMotionVectors(frames);
	return true;
}

bool OverlayMotionVector::processSOS(frame_sp& frame)
{
	auto metadata = frame->getMetadata();
	if (metadata->getFrameType() == FrameMetadata::RAW_IMAGE)
	{
		mDetail->setMatImg(FrameMetadataFactory::downcast<RawImageMetadata>(metadata));
	}
	return true;
}