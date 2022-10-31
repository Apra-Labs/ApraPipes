#include <boost/test/unit_test.hpp>
#include <stdafx.h>
#include "NVRPipeline.h"
#include "PipeLine.h"
#include "EncodedImageMetadata.h"
#include "FrameContainerQueue.h"
#include "Module.h"
#include "Utils.h"
#include "NVRControlModule.h"
#include "WebCamSource.h"
#include "H264EncoderNVCodec.h"
#include"Mp4WriterSink.h"
#include "CudaMemCopy.h"
#include "CCNPPI.h"
#include "CudaStreamSynchronize.h"
#include "H264EncoderNVCodec.h"
#include "ResizeNPPI.h"


class NVRPipeline_Detail {
public:
	NVRPipeline_Detail()
	{
	}
	bool open()
	{
		bool ret = tryStartPipeline();
		return ret;
	}
	bool init(boost::shared_ptr<AbsControlModule>cModule)
	{
		p->init();
		cModule->init();
		return true;
	}
	bool close()
	{
		return tryStopPipeline();
	}
	bool pause()
	{
		try
		{
			p->pause();
		}
		catch (...)
		{
			LOG_ERROR << "Error occured while pausing <>";
			return false;
		}
		return true;
	}
	bool resume()
	{
		try
		{
			p->play();
		}
		catch (...)
		{
			LOG_ERROR << "Error occured while resuming <>";
			return false;
		}
		return true;
	}
	bool startPipeline()
	{
		auto cuContext = apracucontext_sp(new ApraCUcontext());
		uint32_t gopLength = 25;
		uint32_t bitRateKbps = 1000;
		uint32_t frameRate = 30;
		H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::MAIN;
		bool enableBFrames = true;
		auto width = 640;
		auto height = 360;
		WebCamSourceProps webCamSourceprops(0, 1920, 1080);
		webCam = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));
		//auto metadata = framemetadata_sp(new RawImagePlanarMetadata(width, height, ImageMetadata::ImageType::YUV420, size_t(0), CV_8U));
		//webCam->addOutputPin(metadata);
		cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
		auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
		auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
		webCam->setNext(copy);

		auto resize = boost::shared_ptr<Module>(new ResizeNPPI(ResizeNPPIProps(width >> 2, height >> 2, cudaStream_)));
		copy->setNext(resize);

		auto sync = boost::shared_ptr<Module>(new CudaStreamSynchronize(CudaStreamSynchronizeProps(cudaStream_)));
		resize->setNext(sync);
		encoder = boost::shared_ptr<H264EncoderNVCodec>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
		sync->setNext(encoder);
		std::string outFolderPath = "./data/testOutput/mp4_videos/24bpp/";
		auto mp4WriterSinkProps = Mp4WriterSinkProps(1, 1, 24, outFolderPath);
		mp4WriterSinkProps.logHealth = true;
		mp4WriterSinkProps.logHealthFrequency = 10;
		mp4Writer = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps));
		encoder->setNext(mp4Writer);
		mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));
		p = boost::shared_ptr<PipeLine>(new PipeLine("test"));
		p->appendModule(webCam);
		p->addControlModule(mControl);
		init(mControl);
		p->run_all_threaded();
		return true;
	}

	bool tryStartPipeline()
	{
		try
		{
			return startPipeline();
		}
		catch (...)
		{
			LOG_ERROR << "try_startInternal failed";
			return false;
		}
	}

	bool stopPipeline()
	{
		p->stop();
		p->term();
		p->wait_for_all();
		p.reset();
		return true;
	}

	bool tryStopPipeline()
	{
		try
		{
			return stopPipeline();
		}
		catch (...)
		{
			LOG_ERROR << "try_stopInternal failed";
			return false;
		}
	}
	boost::shared_ptr<PipeLine> p;
	boost::shared_ptr<WebCamSource>webCam;
	boost::shared_ptr<H264EncoderNVCodec>encoder;
	boost::shared_ptr<Mp4WriterSink>mp4Writer;
	boost::shared_ptr<NVRControlModule>mControl;
};

// NVRPipeline methods
NVRPipeline::NVRPipeline()
{
	mDetail = new NVRPipeline_Detail();
}
bool NVRPipeline::open()
{
	return mDetail->open();
}
bool NVRPipeline::close()
{
	return mDetail->close();
}
bool NVRPipeline::pause()
{
	return mDetail->pause();
}
bool NVRPipeline::resume()
{
	return mDetail->resume();
}