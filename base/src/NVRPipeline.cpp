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
#include "Mp4WriterSink.h"
#include "CudaMemCopy.h"
#include "CudaStreamSynchronize.h"
#include "H264EncoderNVCodec.h"
#include "ColorConversionXForm.h"
#include "ImageViewerModule.h"
#include "MultimediaQueueXform.h"
#include "ValveModule.h"

class NVRPipeline_Detail {
public:
	NVRPipeline_Detail()
	{
	}
	bool open()
	{
		if (!bPipelineStarted)
		{
			bool ret = tryStartPipeline();
			return ret;
		}
		return true;
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
		if (!bPipelineStarted)
		{
			return true;
		}
		if (pipelinePaused)
		{
			LOG_ERROR << "The pipeline is already paused !!";
		}
		try
		{
			p->pause();
			pipelinePaused = true;
		}
		catch (...)
		{
			LOG_ERROR << "Error occured while pausing !!";
			return false;
		}
		return true;
	}

	bool resume()
	{
		if (!bPipelineStarted)
		{
			return true;
		}
		if (!pipelinePaused)
		{
			return true;
		}
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

	void key_func(boost::shared_ptr<NVRControlModule>& mControl)
	{

		while (true) {
			int k;
			k = getchar();
			if (k == 97)
			{
				BOOST_LOG_TRIVIAL(info) << "Starting Render!";
				mControl->nvrView(true);
				mControl->step();
			}
			if (k == 100)
			{
				BOOST_LOG_TRIVIAL(info) << "Stopping Render!";
				mControl->nvrView(false);
				mControl->step();
			}
		}
	}

	bool startPipeline()
	{
		auto cuContext = apracucontext_sp(new ApraCUcontext());
		uint32_t gopLength = 25;
		uint32_t bitRateKbps = 1000;
		uint32_t frameRate = 30;
		H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::MAIN;
		bool enableBFrames = true;
		auto width = 1920;
		auto height = 1020;


		WebCamSourceProps webCamSourceprops(0, 1920, 1080);
		webCam = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));
		auto colorConvt = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::ConversionType::RGB_TO_YUV420PLANAR)));
		webCam->setNext(colorConvt);

		auto colorConvtView = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::ConversionType::RGB_TO_BGR)));
		webCam->setNext(colorConvtView);

		view = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(ImageViewerModuleProps("NVR-View")));
		colorConvtView->setNext(view);

		cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
		auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
		auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
		colorConvt->setNext(copy);

		encoder = boost::shared_ptr<H264EncoderNVCodec>(new H264EncoderNVCodec(H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames)));
		copy->setNext(encoder);

		std::string outFolderPath_1 = "./data/testOutput/mp4_videos/24bpp/";
		auto mp4WriterSinkProps_1 = Mp4WriterSinkProps(1, 1, 24, outFolderPath_1);
		mp4WriterSinkProps_1.logHealth = true;
		mp4WriterSinkProps_1.logHealthFrequency = 10;
		mp4Writer_1 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_1));
		encoder->setNext(mp4Writer_1);

		multiQue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(MultimediaQueueXformProps(10000, 5000, true)));
		//encoder->setNext(valve);

		valve = boost::shared_ptr<ValveModule>(new ValveModule(ValveModuleProps(-1)));
		//multiQue->setNext(valve);
		//encoder->setNext(valve);
		std::string outFolderPath_2 = "./data/testOutput/mp4_videos/ExportVids/";
		auto mp4WriterSinkProps_2 = Mp4WriterSinkProps(1, 1, 24, outFolderPath_2);
		mp4WriterSinkProps_2.logHealth = true;
		mp4WriterSinkProps_2.logHealthFrequency = 10;
		mp4Writer_2 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_2));
		//encoder->setNext(mp4Writer_2);

		mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(NVRControlModuleProps()));

		mControl->enrollModule("WebCamera", webCam);
		mControl->enrollModule("Renderer", view);
		mControl->enrollModule("Writer-1", mp4Writer_1);
		mControl->enrollModule("MultimediaQueue", multiQue);
		mControl->enrollModule("Valve", valve);
		mControl->enrollModule("Writer-2", mp4Writer_2);

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
			bPipelineStarted = true;
			return startPipeline();
		}
		catch (...)
		{
			LOG_ERROR << "starting pipeline failed!!";
			return false;
		}
	}

	bool startRecord()
	{
		if (!bPipelineStarted)
		{
			LOG_ERROR << "The pipeline is not started!!";
			return false;
		}
		if (pipelinePaused)
		{
			LOG_ERROR << "The pipeline is paused!!";
			return false;
		}
		mControl->nvrRecord(true);
		mControl->step();
		return true;
	}

	bool stopRecord()
	{
		if (!bPipelineStarted)
		{
			LOG_ERROR << "The pipeline is not started!!";
			return false;
		}
		if (pipelinePaused)
		{
			LOG_ERROR << "The pipeline is paused!!";
			return false;
		}
		mControl->nvrRecord(false);
		mControl->step();
		return true;
	}

	bool xport(uint64_t ts, uint64_t te)
	{
		if (!bPipelineStarted)
		{
			LOG_ERROR << "The pipeline is not started!!";
			return false;
		}
		if (pipelinePaused)
		{
			LOG_ERROR << "The pipeline is paused!!";
			return false;
		}
		mControl->nvrExport(ts, te);
		mControl->step();
		return true;
	}

	bool stopPipeline()
	{
		boost::this_thread::sleep_for(boost::chrono::seconds(30));
		p->stop();
		p->term();
		p->wait_for_all();
		inp.join();
		return true;
	}

	bool tryStopPipeline()
	{
		if (!bPipelineStarted)
		{
			return true;
		}
		if (pipelinePaused)
		{
			return true;
		}
		try
		{
			return stopPipeline();
		}
		catch (...)
		{
			LOG_ERROR << "Stopping pipeline failed!!";
			return false;
		}
	}


	bool bPipelineStarted = false;
	bool pipelinePaused = false;
	boost::shared_ptr<PipeLine> p;
	boost::shared_ptr<WebCamSource>webCam;
	boost::shared_ptr<H264EncoderNVCodec>encoder;
	boost::shared_ptr<Mp4WriterSink>mp4Writer_1;
	boost::shared_ptr<Mp4WriterSink>mp4Writer_2;
	boost::shared_ptr<NVRControlModule>mControl;
	boost::shared_ptr<ImageViewerModule>view;
	boost::shared_ptr<MultimediaQueueXform>multiQue;
	boost::shared_ptr<ValveModule>valve;
	boost::thread inp;

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

bool NVRPipeline::startRecording()
{
	return mDetail->startRecord();
}

bool NVRPipeline::stopRecording()
{
	return mDetail->stopRecord();
}

bool NVRPipeline::xport(uint64_t TS, uint64_t TE)
{
	return mDetail->xport(TS, TE);
}