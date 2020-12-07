#include "VirtualCameraSink.h"
#include "FrameMetadata.h"
#include "Frame.h"
#include "Logger.h"
#include "Utils.h"
#include "AIPExceptions.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

class VirtualCameraSink::Detail
{
public:
	Detail(VirtualCameraSinkProps &_props) : props(_props), dev_fd(0), imageSize(0)
	{
	}

	~Detail()
	{
		if (dev_fd)
		{
			close(dev_fd);
			dev_fd = 0;
		}
	}

	void setMetadata(framemetadata_sp &metadata)
	{
		if (metadata->getFrameType() != FrameMetadata::FrameType::RAW_IMAGE)
		{
			throw AIPException(AIP_FATAL, "Only RGB is supported. Wrong FrameType");
		}

		auto rawImageMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
		if (rawImageMetadata->getImageType() != ImageMetadata::RGB)
		{
			throw AIPException(AIP_FATAL, "Only RGB is supported. Wrong ImageType");
		}
		width = rawImageMetadata->getWidth();
		height = rawImageMetadata->getHeight();
		step = rawImageMetadata->getStep();
		imageSize = rawImageMetadata->getDataSize();
		if (step != width * 3)
		{
			throw AIPException(AIP_FATAL, "Not Implemented. step must be equal to width*3. width<" + std::to_string(width) + "> step<" + std::to_string(step) + ">");
		}

		init();
	}

	bool writeToDevice(void *data)
	{
		try
		{
			auto ret = write(dev_fd, data, imageSize);
			if(ret == -1)
			{
				LOG_ERROR << "FAILED TO WRITE TO DEVICE.";
			}
		}
		catch (...)
		{
			LOG_ERROR << "writing to device failed.";
		}
	}

	void getImageSize(int &_width, int &_height)
	{
		_width = width;
		_height = height;
	}

	VirtualCameraSinkProps props;
	size_t imageSize;

private:
	void init()
	{
		dev_fd = open(props.device.c_str(), O_RDWR);
		if (dev_fd == -1)
		{
			throw AIPException(AIP_FATAL, "cannot open video device<" + props.device + ">");
		}

		struct v4l2_format v;
		v.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
		if (ioctl(dev_fd, VIDIOC_G_FMT, &v) == -1)
		{
			throw AIPException(AIP_FATAL, "cannot setup video device<" + props.device + ">");
		}
		v.fmt.pix.width = width;
		v.fmt.pix.height = height;
		v.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
		v.fmt.pix.sizeimage = imageSize;
		v.fmt.pix.field = V4L2_FIELD_NONE;
		if (ioctl(dev_fd, VIDIOC_S_FMT, &v) == -1)
		{
			throw AIPException(AIP_FATAL, "cannot setup video device 2<" + props.device + ">");
		}
	}

	int dev_fd;
	int width;
	int height;
	size_t step;
};

VirtualCameraSink::VirtualCameraSink(VirtualCameraSinkProps props) : Module(SINK, "VirtualCameraSink", props)
{
	mDetail.reset(new Detail(props));
}

VirtualCameraSink::~VirtualCameraSink() {}

bool VirtualCameraSink::validateInputPins()
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

bool VirtualCameraSink::init()
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

bool VirtualCameraSink::term()
{
	return Module::term();
}

bool VirtualCameraSink::process(frame_container &frames)
{
	mDetail->writeToDevice(frames.cbegin()->second->data());

	return true;
}

bool VirtualCameraSink::processSOS(frame_sp &frame)
{
	auto metadata = frame->getMetadata();
	mDetail->setMetadata(metadata);

	return true;
}

bool VirtualCameraSink::shouldTriggerSOS()
{
	return mDetail->imageSize == 0;
}

bool VirtualCameraSink::processEOS(string &pinId)
{
	mDetail->imageSize = 0;
	return true;
}

void VirtualCameraSink::getImageSize(int &width, int &height)
{
	mDetail->getImageSize(width, height);
}