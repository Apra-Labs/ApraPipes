#include <cstdint>
#include <boost/foreach.hpp>
#include "OverlayModule.h"
#include "Utils.h"
#include "FrameContainerQueue.h"
#include "OverlayDataInfo.h"

class OverlayComponent
{
public:
	OverlayComponent() {}
	class OverlayComponent(Primitive& _shapeType, frame_sp inFrame)
	{
		
	}
	virtual void draw(frame_sp inRawFrame) = 0;
};

class Line : public OverlayComponent
{
public:
	Line(Primitive type, frame_sp frame)
	{

	}
	void draw(frame_sp inRawFrame)
	{
		printf("draw line");
	}
};

class Circle : public OverlayComponent
{
public:
	Circle(Primitive type, frame_sp frame)
	{

	}
	void draw(frame_sp inRawFrame)
	{
		printf("draw circle");
	}
};

class rectangle : public OverlayComponent
{
public:
	rectangle() {}
	rectangle(RectangleOverlay _rectOverlayObj)
	{
		rectOverlayObj = _rectOverlayObj;
	}

	void draw(frame_sp inRawFrame)
	{
		/*cmdObj.cvOverlayImage.data = static_cast<uint8_t*>(inRawFrame->data());
		cv::Point pt1(rectOverlayObj.x1, rectOverlayObj.y1);
		cv::Point pt2(rectOverlayObj.x2, rectOverlayObj.y2);
		cv::rectangle(cmdObj.cvOverlayImage, pt1, pt2 , cv::Scalar(0, 255, 0), 2);*/
	}
private:
	//OverlayCommand cmdObj;
	RectangleOverlay rectOverlayObj;
};

class Composite : public OverlayComponent
{
public:
	void add(OverlayComponent* componentObj)
	{
		gList.push_back(componentObj);
	}
	void draw(frame_sp inRawFrame)
	{
		printf("draw compisite");
	}

private:
	vector<OverlayComponent*> gList;
};

class OverlayCommand
{
public:
	OverlayCommand() {}
	
	virtual void execute(OverlayComponent* Overlay, frame_sp inRawFrame)
	{
		Overlay->draw(inRawFrame);
	}
	void setMetadata(RawImageMetadata* rawMetadata)
	{
		cvOverlayImage = Utils::getMatHeader(rawMetadata);
		RawImageMetadata outputMetadata(rawMetadata->getWidth(), rawMetadata->getHeight(), ImageMetadata::BGR, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true);
		rawOutputMetadata = framemetadata_sp(new RawImageMetadata());
		auto rawOutMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(rawOutputMetadata);
		rawOutMetadata->setData(outputMetadata);
	}
public:
	cv::Mat cvOverlayImage;
	framemetadata_sp rawOutputMetadata;
	//boost::shared_ptr<FrameContainerQueueOverlayAdapter> frameContainerOverlayAdapt;;
};


OverlayModule::OverlayModule(OverlayModuleProps props) : Module(TRANSFORM, "OverlayMotionVectors", props)
{
	mDetail.reset(new OverlayCommand());
	//mDetail->frameContainerOverlayAdapt.reset(new FrameContainerQueueOverlayAdapter([&](size_t size) -> frame_sp {return makeFrame(size); }));
}

void OverlayModule::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
	if(metadata->getFrameType() == FrameMetadata::RAW_IMAGE)
	mOutputPinId = addOutputPin(metadata);
}


bool OverlayModule::init()
{
	return Module::init();
}

bool OverlayModule::term()
{
	return Module::term();
}

bool OverlayModule::validateInputPins()
{
	/*if (getNumberOfInputPins() > 2)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 2. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}*/
	return true;
}

bool OverlayModule::validateOutputPins()
{
	/*if (getNumberOfOutputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateOutputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}*/
	return true;
}

bool OverlayModule::shouldTriggerSOS()
{
	return true;
}

bool OverlayModule::process(frame_container& frames)
{
	OverlayComponent* OverlayObj;
	for (auto it = frames.cbegin(); it != frames.cend(); it++)
	{
		auto metadata = it->second->getMetadata();
		auto frameTye = metadata->getFrameType();
		if (frameTye == FrameMetadata::OVERLAY_INFO_IMAGE)
		{
			frame_sp frame = it->second;
			//RectangleOverlay rectOverlay = RectangleOverlay::deSerialize(frame);
			
		}

		if (frameTye == FrameMetadata::FrameType::RAW_IMAGE)
		{
			frame_sp outFrame = it->second;
			mDetail->execute(OverlayObj, outFrame);
			frames.insert(make_pair(mOutputPinId, outFrame));
			send(frames);
		}
	}
	return true;
}

bool OverlayModule::processSOS(frame_sp& frame)
{
	
	auto metadata = frame->getMetadata();
	if (metadata->getFrameType() == FrameMetadata::RAW_IMAGE)
	{
		mDetail->setMetadata(FrameMetadataFactory::downcast<RawImageMetadata>(metadata));
	}
	return true;
}