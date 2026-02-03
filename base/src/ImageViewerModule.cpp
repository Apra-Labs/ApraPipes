#include "stdafx.h"
#include "ImageViewerModule.h"
#include "Frame.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "FrameMetadata.h"
#include "Logger.h"
#include "Utils.h"

#if 1
#include "ApraNvEglRenderer.h"
#include "DMAFDWrapper.h"
#include "Command.h"
#include <drm/drm_fourcc.h>
#endif

class DetailRenderer
{

public:
	DetailRenderer(ImageViewerModuleProps &_props) : props(_props) {}

	~DetailRenderer()
	{
#if 1
		destroyWindow();
#endif
	}

	// arm:EGL Renderer , linux/windows:imshow

	virtual bool view() = 0;

	bool eglInitializer(uint32_t _height, uint32_t _width)
	{
#if 1
		uint32_t displayHeight, displayWidth;
		NvEglRenderer::getDisplayResolution(displayWidth, displayHeight);

		// Validate width and height before using
		uint32_t rendererWidth = props.width != 0 ? props.width : _width;
		uint32_t rendererHeight = props.height != 0 ? props.height : _height;
		if (rendererWidth == 0 || rendererHeight == 0) {
			LOG_ERROR << "Invalid renderer dimensions: width=" << rendererWidth << ", height=" << rendererHeight;
			return false;
		}

		props.x_offset += (displayWidth - rendererWidth) / 2;
		props.y_offset += (displayHeight - rendererHeight) / 2;

		renderer = NvEglRenderer::createEglRenderer(
			props.strTitle.c_str(),
			rendererWidth,
			rendererHeight,
			props.x_offset,
			props.y_offset,
			props.ttfFilePath.c_str(),
			props.message.c_str(),
			props.scale,
			props.r, props.g, props.b,
			props.fontSize,
			props.textPosX, props.textPosY,
			props.imagePath,
			props.imagePosX, props.imagePosY,
			props.imageWidth, props.imageHeight,
			props.opacity,
			props.mask,
			props.imageOpacity,
			props.textOpacity
		);

		if (!renderer)
		{
			LOG_ERROR << "Failed to create EGL renderer. Parameters: width=" << rendererWidth << ", height=" << rendererHeight << ", x_offset=" << props.x_offset << ", y_offset=" << props.y_offset;
			return false;
		}
#endif
		return true;
	}

	bool destroyWindow()
	{
#if 1
		if (renderer)
		{
			delete renderer;
		}
#else
		return true;
#endif
	}

	bool shouldTriggerSOS()
	{
#if 1
		return !renderer;
#else
		return !mImg.rows;
#endif
	}

	void setMatImg(RawImageMetadata *rawMetadata)
	{
		mImg = Utils::getMatHeader(rawMetadata);
	}

	void showImage(frame_sp &frame)
	{
		mImg.data = (uchar *)frame->data();
		cv::imshow(props.strTitle, mImg);
		cv::waitKey(1);
	}

public:
	frame_sp inputFrame;
	ImageViewerModuleProps props;
	NvEglRenderer *getRenderer() const { return renderer; }

protected:
	cv::Mat mImg;
#if 1
	NvEglRenderer *renderer = nullptr;
#endif
};

ImageViewerModule::ImageViewerModule(ImageViewerModuleProps _props) : Module(SINK, "ImageViewerModule", _props), mProps(_props) {}

ImageViewerModule::~ImageViewerModule() {}

class DetailEgl : public DetailRenderer
{
public:
	DetailEgl(ImageViewerModuleProps &_props) : DetailRenderer(_props) {}

	bool view()
	{
#if 1
		if (!inputFrame || !inputFrame->data()) {
			LOG_ERROR << "Input frame or data is null.";
			return false;
		}
		DMAFDWrapper* wrapper = static_cast<DMAFDWrapper *>(inputFrame->data());
		if (!wrapper) {
			LOG_ERROR << "DMAFDWrapper cast failed.";
			return false;
		}
		int fd = wrapper->getFd();
		if (fd <= 0) {
			LOG_ERROR << "Invalid DMAFDWrapper FD: " << fd;
			return false;
		}
		LOG_INFO << "Rendering DMAFDWrapper FD: " << fd;
		renderer->render(fd);
#endif
		return true;
	}
};

class DetailImageviewer : public DetailRenderer
{
public:
	DetailImageviewer(ImageViewerModuleProps &_props) : DetailRenderer(_props) {}

	bool view()
	{
		showImage(inputFrame);
		return true;
	}
};

bool ImageViewerModule::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}
	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	FrameMetadata::MemType inputMemType = metadata->getMemType();

#if 1
	if (inputMemType != FrameMetadata::MemType::DMABUF)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be DMABUF. Actual<" << inputMemType << ">";
		return false;
	}
	if (frameType != FrameMetadata::RAW_IMAGE && frameType != FrameMetadata::RAW_IMAGE_PLANAR)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE or RAW_IMAGE_PLANAR. Actual<" << frameType << ">";
		return false;
	}
#else
	if (frameType != FrameMetadata::RAW_IMAGE)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}
#endif
	return true;
}

void ImageViewerModule::addInputPin(framemetadata_sp &metadata, string &pinId)
{
	Module::addInputPin(metadata, pinId);
#if 1
	mDetail.reset(new DetailEgl(mProps));
#else
	mDetail.reset(new DetailImageviewer(mProps));
#endif
}

bool ImageViewerModule::init()
{
	if (!Module::init())
	{
		return false;
	}

	return true;
}

bool ImageViewerModule::term() { return Module::term(); }

bool ImageViewerModule::process(frame_container &frames)
{
	mDetail->inputFrame = frames.cbegin()->second;
	if (isFrameEmpty(mDetail->inputFrame))
	{
		return true;
	}
	mDetail->view();
	return true;
}

bool ImageViewerModule::processSOS(frame_sp &frame)
{
	auto inputMetadata = frame->getMetadata();
	auto frameType = inputMetadata->getFrameType();
	FrameMetadata::MemType mInputMemType = inputMetadata->getMemType();
#if 1
	int width = 0;
	int height = 0;
	int pitch = 0;
	int fourcc = 0;
	int offset0 = 0;
	int offset1 = 0;
	int offset2 = 0;
	int pitch1 = 0;
	int pitch2 = 0;
	switch (frameType)
	{
	case FrameMetadata::FrameType::RAW_IMAGE:
	{
		auto metadata = FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata);
		width = metadata->getWidth();
		height = metadata->getHeight();
		pitch = static_cast<int>(metadata->getStep());
		switch (metadata->getImageType())
		{
		case ImageMetadata::RGBA:
			fourcc = DRM_FORMAT_ABGR8888;
			break;
		case ImageMetadata::BGRA:
			fourcc = DRM_FORMAT_BGRA8888;
			break;
		case ImageMetadata::UYVY:
			fourcc = DRM_FORMAT_UYVY;
			break;
		case ImageMetadata::YUYV:
			fourcc = DRM_FORMAT_YUYV;
			break;
		default:
			fourcc = DRM_FORMAT_RGBA8888;
			break;
		}
	}
	break;
	case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
	{
		auto metadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(inputMetadata);
		width = metadata->getWidth(0);
		height = metadata->getHeight(0);
		auto dmaWrapper = static_cast<DMAFDWrapper *>(frame->data());
		if (!dmaWrapper)
		{
			LOG_ERROR << "DMAFDWrapper is null for planar frame.";
			return false;
		}
		auto surf = dmaWrapper->getNvBufSurface();
		if (!surf)
		{
			LOG_ERROR << "NvBufSurface is null for planar frame.";
			return false;
		}
		auto &planeParams = surf->surfaceList[0].planeParams;
		pitch = static_cast<int>(planeParams.pitch[0]);
		pitch1 = static_cast<int>(planeParams.pitch[1]);
		pitch2 = static_cast<int>(planeParams.pitch[2]);
		offset0 = static_cast<int>(planeParams.offset[0]);
		offset1 = static_cast<int>(planeParams.offset[1]);
		offset2 = static_cast<int>(planeParams.offset[2]);

		if (metadata->getImageType() == ImageMetadata::NV12)
		{
			fourcc = DRM_FORMAT_NV12;
		}
		else if (metadata->getImageType() == ImageMetadata::YUV420)
		{
			fourcc = DRM_FORMAT_YUV420;
		}
	}
	break;
	default:
		throw AIPException(AIP_FATAL, "Unsupported FrameType<" + std::to_string(frameType) + ">");
	}

	if (!mDetail->eglInitializer(height, width))
	{
		LOG_ERROR << "Failed to initialize EGL renderer.";
		return false;
	}

	auto renderer = mDetail->getRenderer();
	if (!renderer)
	{
		LOG_ERROR << "EGL renderer is not available after initialization.";
		return false;
	}

	if (fourcc != 0 && pitch != 0)
	{
		if (frameType == FrameMetadata::FrameType::RAW_IMAGE_PLANAR && (fourcc == DRM_FORMAT_NV12 || fourcc == DRM_FORMAT_YUV420))
		{
			int numPlanes = (fourcc == DRM_FORMAT_NV12) ? 2 : 3;
			renderer->setImportParamsPlanar(
				fourcc,
				width,
				height,
				pitch,
				offset0,
				pitch1,
				offset1,
				pitch2,
				offset2,
				numPlanes);
		}
		else
		{
			renderer->setImportParams(pitch, fourcc, 0, width, height);
		}
	}
#else
	mDetail->setMatImg(FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata));
#endif
	return true;
}

bool ImageViewerModule::shouldTriggerSOS()
{
	return mDetail->shouldTriggerSOS();
}

bool ImageViewerModule::handleCommand(Command::CommandType type, frame_sp &frame)
{
#if 1
	if (type == Command::CommandType::DeleteWindow)
	{
		mDetail->destroyWindow();
		return true;
	}
	else if (type == Command::CommandType::CreateWindow)
	{
		EglRendererCreateWindow cmd;
		getCommand(cmd, frame);
		mDetail->eglInitializer(cmd.width, cmd.height);
		return true;
	}
	return Module::handleCommand(type, frame);
#else
	return true;
#endif
}

bool ImageViewerModule::closeWindow()
{
#if 1
	EglRendererCloseWindow cmd;
	return queueCommand(cmd);
#else
	return true;
#endif
}

bool ImageViewerModule::createWindow(int width, int height)
{
#if 1
	EglRendererCreateWindow cmd;
	cmd.width = width;
	cmd.height = height;
	return queueCommand(cmd);
#else
	return true;
#endif
}