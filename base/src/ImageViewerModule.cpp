#include "stdafx.h"
#include "ImageViewerModule.h"
#include "Frame.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "FrameMetadata.h"
#include "Logger.h"
#include "Utils.h"

#if defined(__arm__) || defined(__aarch64__)
#include "ApraNvEglRenderer.h"
#include "DMAFDWrapper.h"
#include "Command.h"
#endif

class DetailRenderer
{

public:
	DetailRenderer(ImageViewerModuleProps &_props) : props(_props) {}

	~DetailRenderer()
	{
#if defined(__arm__) || defined(__aarch64__)
		destroyWindow();
#endif
	}

	// arm:EGL Renderer , linux/windows:imshow

	virtual bool view() = 0;

	bool eglInitializer(uint32_t _imageHeight,uint32_t _imageWidth)
	{
#if defined(__arm__) || defined(__aarch64__)
		NvEglRenderer::getDisplayResolution(displayWidth, displayHeight);
		x_offset = props.x_offset;
		y_offset = props.y_offset;
		if (props.height != 0 && props.width != 0)
		{
			props.x_offset += (displayWidth - props.width) / 2;
			props.y_offset += (displayHeight - props.height) / 2;
			renderer = NvEglRenderer::createEglRenderer(__TIMESTAMP__, props.width, props.height, props.x_offset, props.y_offset, props.displayOnTop);
		}
		else
		{
			props.x_offset += (displayWidth - _imageWidth) / 2;
			props.y_offset += (displayHeight - _imageHeight) / 2;
			renderer = NvEglRenderer::createEglRenderer(__TIMESTAMP__, _imageWidth, _imageHeight, props.x_offset, props.y_offset, props.displayOnTop);
		}
		if (!renderer)
		{
			LOG_ERROR << "Failed to create EGL renderer";
			return false;
		}
#endif
		return true;
	}

	bool destroyWindow()
	{
#if defined(__arm__) || defined(__aarch64__)
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
#if defined(__arm__) || defined(__aarch64__)
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

	bool eventHandler()
	{
#if defined(__arm__) || defined(__aarch64__)
		XEvent event;
		if (XCheckMaskEvent(renderer->x_display,
							ButtonPressMask |
								NoEventMask |
								KeyPressMask |
								KeyReleaseMask |
								ButtonReleaseMask |
								EnterWindowMask |
								LeaveWindowMask |
								PointerMotionMask |
								PointerMotionHintMask |
								Button1MotionMask |
								Button2MotionMask |
								Button3MotionMask |
								Button4MotionMask |
								Button5MotionMask |
								ButtonMotionMask |
								KeymapStateMask |
								ExposureMask |
								VisibilityChangeMask |
								StructureNotifyMask |
								ResizeRedirectMask |
								SubstructureNotifyMask |
								SubstructureRedirectMask |
								FocusChangeMask |
								PropertyChangeMask |
								ColormapChangeMask |
								OwnerGrabButtonMask,
							&event))
		{

			if (event.type == ButtonPress)
			{
				if (event.xbutton.button == Button1)
				{
					int current_time = event.xbutton.time;
					int time_difference = current_time - last_click_time;
					if (time_difference <= doubleClickInterval)
					{
						// Double click detected
						printf("Double click\n");
						num_clicks = 0;
						if (sync)
						{
							props.x_offset = x_offset;
							props.y_offset = y_offset;
							if (props.height != 0 && props.width != 0)
							{
								props.width = displayWidth;
								props.height = displayHeight;
							}
							else
							{
								imageHeight = displayHeight;
								imageWidth = displayWidth;
							}
							destroyWindow();
							eglInitializer(imageHeight, imageWidth);
							sync = false;
						}
						else
						{
							props.x_offset = x_offset;
							props.y_offset = y_offset;
							if (props.height != 0 && props.width != 0)
							{
								props.width = originalWidth;
								props.height = originalHeight;
							}
							else
							{
								imageHeight = originalHeight;
								imageWidth = originalWidth;
							}
							destroyWindow();
							eglInitializer(imageHeight, imageWidth);
							sync = true;
						}
					}
					else
					{
						// Single click detected
						printf("Single click\n");
						num_clicks++;

						if (num_clicks >= 2)
						{
							// Calculate the adaptive double-click threshold
							doubleClickInterval = time_difference / num_clicks;
						}
					}
					last_click_time = current_time;
				}
			}
		}
#endif
		return true;
	}

public:
	frame_sp inputFrame;
	ImageViewerModuleProps props;
	uint32_t imageHeight = 0;
	uint32_t imageWidth = 0;
	uint32_t originalWidth = 0;
	uint32_t originalHeight = 0;

protected:
    cv::Mat mImg;
	uint32_t x_offset = 0;
	uint32_t y_offset = 0;
	bool sync = true;
	uint32_t displayHeight, displayWidth;
	int last_click_time = 0;
	int doubleClickInterval = 0;
	int num_clicks = 0;
#if defined(__arm__) || defined(__aarch64__)
	NvEglRenderer *renderer = nullptr;
#endif
};

ImageViewerModule::ImageViewerModule(ImageViewerModuleProps _props) : Module(SINK, "ImageViewerModule", _props), mProps(_props) {}

ImageViewerModule::~ImageViewerModule() {}

class DetailEgl : public DetailRenderer
{
public:
	DetailEgl(ImageViewerModuleProps &_props) : DetailRenderer(_props)
	{
		
	}

	bool view()
	{
#if defined(__arm__) || defined(__aarch64__)
		renderer->render((static_cast<DMAFDWrapper *>(inputFrame->data()))->getFd());
		eventHandler();
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

#if defined(__arm__) || defined(__aarch64__)
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
#if defined(__arm__) || defined(__aarch64__)
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
#if defined(__arm__) || defined(__aarch64__)
	switch (frameType)
	{
	case FrameMetadata::FrameType::RAW_IMAGE:
	{
		auto metadata = FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata);
		mDetail->imageWidth = metadata->getWidth();
		mDetail->imageHeight = metadata->getHeight();
	}
	break;
	case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
	{
		auto metadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(inputMetadata);
		mDetail->imageWidth = metadata->getWidth(0);
		mDetail->imageHeight = metadata->getHeight(0);
	}
	break;
	default:
		throw AIPException(AIP_FATAL, "Unsupported FrameType<" + std::to_string(frameType) + ">");
	}

	if (mProps.height != 0 && mProps.width != 0)
	{
		mDetail->originalHeight = mProps.height;
		mDetail->originalWidth = mProps.width;
	}
	else
	{
		mDetail->originalHeight = mDetail->imageHeight;
		mDetail->originalWidth = mDetail->imageWidth;
	}
	mDetail->eglInitializer(mDetail->imageHeight, mDetail->imageWidth);
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
#if defined(__arm__) || defined(__aarch64__)
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
#if defined(__arm__) || defined(__aarch64__)
	EglRendererCloseWindow cmd;
	return queueCommand(cmd);
#else
	return true;
#endif
}

bool ImageViewerModule::createWindow(int width, int height)
{
#if defined(__arm__) || defined(__aarch64__)
	EglRendererCreateWindow cmd;
	cmd.width = width;
	cmd.height = height;
	return queueCommand(cmd);
#else
	return true;
#endif
}
