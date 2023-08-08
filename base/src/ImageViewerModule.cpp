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
	DetailRenderer(ImageViewerModuleProps& _props) : props(_props) {}

	~DetailRenderer() 
	{
		#if defined(__arm__) || defined(__aarch64__)
		 if(renderer)
        {
            delete renderer;
        }
		#endif
	}

	virtual bool compute() = 0;

	bool eglInitializer(uint32_t _height, uint32_t _width , bool _displayOnTop)
	{
	#if defined(__arm__) || defined(__aarch64__)
        uint32_t displayHeight, displayWidth;
        NvEglRenderer::getDisplayResolution(displayWidth,displayHeight);
        if(height!=0 && width!=0){
            x_offset += (displayWidth-width)/2;
            y_offset += (displayHeight-height)/2;
            renderer = NvEglRenderer::createEglRenderer(__TIMESTAMP__, width, height, x_offset, y_offset,displayOnTop);
        }else{
            x_offset += (displayWidth-_width)/2;
            y_offset += (displayHeight-_height)/2;
            renderer = NvEglRenderer::createEglRenderer(__TIMESTAMP__, _width, _height, x_offset, y_offset, displayOnTop);
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
        if(renderer)
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

	void setMatImg(RawImageMetadata* rawMetadata)
	{
		mImg = Utils::getMatHeader(rawMetadata);
	}

	void showImage(frame_sp& frame)
	{
		mImg.data = (uchar *)frame->data();
		cv::imshow(mStrTitle, mImg);
		cv::waitKey(1); //use 33 for linux Grrr
	}

public:
	cv::Mat mImg;	
	std::string mStrTitle;
    uint32_t x_offset,y_offset,width,height;
    bool displayOnTop;
	frame_sp inputFrame;
	frame_sp outputFrame;
	ImageViewerModuleProps props;

#if defined(__arm__) || defined(__aarch64__)
	NvEglRenderer *renderer = nullptr;
#endif
};

ImageViewerModule::ImageViewerModule(ImageViewerModuleProps _props) : Module(SINK, "ImageViewerModule", _props), mProps(_props) {}

ImageViewerModule::~ImageViewerModule() {}

class DetailEgl : public DetailRenderer
{
public:
	DetailEgl(ImageViewerModuleProps &_props) : DetailRenderer(_props) {}

	bool compute()
	{
		#if defined(__arm__) || defined(__aarch64__)
		renderer->render((static_cast<DMAFDWrapper *>(inputFrame->data()))->getFd());
		#endif
		return true;
	}
};

class DetailImageviewer : public DetailRenderer
{
public:
	DetailImageviewer(ImageViewerModuleProps &_props) : DetailRenderer(_props) {}

	bool compute()
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
	FrameMetadata::MemType inputMemType = metadata->getMemType();
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

bool ImageViewerModule::process(frame_container& frames)
{
	mDetail->inputFrame = frames.cbegin()->second;
	if (isFrameEmpty(mDetail->inputFrame))
	{
		return true;
	}
	mDetail->compute();
	return true;
}

bool ImageViewerModule::processSOS(frame_sp& frame)
{
	auto inputMetadata = frame->getMetadata();
	auto frameType = inputMetadata->getFrameType();
	FrameMetadata::MemType mInputMemType = inputMetadata->getMemType();
    #if defined(__arm__) || defined(__aarch64__)
	    int width = 0;
        int height =0;
		switch (frameType)
        {
        case FrameMetadata::FrameType::RAW_IMAGE:
        {
           auto metadata = FrameMetadataFactory::downcast<RawImageMetadata>(inputMetadata);
           width = metadata->getWidth();
           height = metadata->getHeight();
        }
        break;
        case FrameMetadata::FrameType::RAW_IMAGE_PLANAR:
        {
           auto metadata = FrameMetadataFactory::downcast<RawImagePlanarMetadata>(inputMetadata);
           width = metadata->getWidth(0);
           height = metadata->getHeight(0);
        }
        break;
        default:
        throw AIPException(AIP_FATAL, "Unsupported FrameType<" + std::to_string(frameType) + ">");
        }

		mDetail->eglInitializer(height,width,mDetail->displayOnTop);
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
        EglRendererCloseWindow cmd;
        getCommand(cmd, frame);
        mDetail->destroyWindow();
        return true;
    }
    else if (type == Command::CommandType::CreateWindow)
    {
        EglRendererCreateWindow cmd;
        getCommand(cmd, frame);
        mDetail->eglInitializer(cmd.width, cmd.height,mDetail->displayOnTop);
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