#include "VirtualCameraSink.h"
#include "FrameMetadata.h"
#include "DMAFDWrapper.h"
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
	Detail(VirtualCameraSinkProps &_props) : props(_props), dev_fd(0), imageSize(0), imageType(ImageMetadata::ImageType::RGB)
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
		auto frameType = metadata->getFrameType();
		switch (frameType)
		{
		case FrameMetadata::FrameType::RAW_IMAGE:
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImageMetadata>(metadata);
			width = inputRawMetadata->getWidth();
			height = inputRawMetadata->getHeight();
			auto step = inputRawMetadata->getStep();
			if (step != width * 3)
			{
				throw AIPException(AIP_FATAL, "Not Implemented. step must be equal to width*3. width<" + std::to_string(width) + "> step<" + std::to_string(step) + ">");
			}
			imageType = inputRawMetadata->getImageType();
		}
		break;
		case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
		{
			auto inputRawMetadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(metadata);
			width = inputRawMetadata->getWidth(0);
			height = inputRawMetadata->getHeight(0);
			auto channels = inputRawMetadata->getChannels();
			for (auto i = 0; i < channels; i++)
			{
				auto step = inputRawMetadata->getStep(i);
				auto _width = inputRawMetadata->getWidth(i);
				if (step != _width)
				{
					throw AIPException(AIP_FATAL, "Not Implemented. step must be equal to width. width<" + std::to_string(_width) + "> step<" + std::to_string(step) + ">");
				}
			}
			imageType = inputRawMetadata->getImageType();
		}
		break;
		default:
			throw AIPException(AIP_FATAL, "Expected Raw Image or RAW_IMAGE_PLANAR. Actual<" + std::to_string(frameType) + ">");
			break;
		}

		switch (imageType)
		{
		case ImageMetadata::RGB:
		case ImageMetadata::YUV420:
			break;
		default:
			throw AIPException(AIP_FATAL, "Expected ImageType RGB or YUV420. Actual<" + std::to_string(imageType) + ">");
		}

		auto memType = metadata->getMemType();
		switch (memType)
		{
		case FrameMetadata::MemType::HOST:
			getDataPtr = [&](frame_sp& frame) -> void* {
				return getHostDataPtr(frame); };
			break;
		case FrameMetadata::MemType::DMABUF:
			getDataPtr = [&](frame_sp& frame) -> void* {
				return getDMAFDHostDataPtr(frame); };
			break;
		default:
			throw AIPException(AIP_FATAL, "Expected MemType HOST or DMABUF. Actual<" + std::to_string(memType) + ">");
		}

		imageSize = metadata->getDataSize();

		init();
	}

	bool writeToDevice(frame_sp frame)
	{
		try
		{
			auto ret = write(dev_fd, getDataPtr(frame), imageSize);
			if (ret == -1)
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
		switch (imageType)
		{
		case ImageMetadata::ImageType::RGB:
			v.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
			break;
		case ImageMetadata::ImageType::YUV420:
			v.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;
			break;
		default:
			throw AIPException(AIP_NOTEXEPCTED, "RGB or YUV420 is expected.");
		}
		v.fmt.pix.sizeimage = imageSize;
		v.fmt.pix.field = V4L2_FIELD_NONE;
		if (ioctl(dev_fd, VIDIOC_S_FMT, &v) == -1)
		{
			throw AIPException(AIP_FATAL, "cannot setup video device 2<" + props.device + ">");
		}
	}

	void *getHostDataPtr(frame_sp &frame)
	{
		return frame->data();
	}

	void *getDMAFDHostDataPtr(frame_sp &frame)
	{
		auto ptr = static_cast<DMAFDWrapper *>(frame->data());
		return ptr->getHostPtr();
	}

	int dev_fd;
	int width;
	int height;
	ImageMetadata::ImageType imageType;

	typedef std::function<void *(frame_sp &)> GetDataPtr;
	GetDataPtr getDataPtr;
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
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
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

	return true;
}

bool VirtualCameraSink::term()
{
	return Module::term();
}

bool VirtualCameraSink::process(frame_container &frames)
{
	mDetail->writeToDevice(frames.cbegin()->second);

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