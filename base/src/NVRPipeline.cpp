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
#include "Mp4ReaderSource.h"
#include "H264Metadata.h"
#include "Mp4VideoMetadata.h"

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
			if (k == 101)
			{
				BOOST_LOG_TRIVIAL(info) << "Starting Requested Export!!";
				boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
				auto now = (boost::posix_time::microsec_clock::universal_time() - time_epoch).total_milliseconds();
				uint64_t seekStartTS = now - 180000;
				uint64_t seekEndTS = now + 120000;
				mControl->nvrExport(seekStartTS, seekEndTS);
				mControl->step();
			}
		}
	}

	bool startPipeline()
	{
		Logger::setLogLevel(boost::log::trivial::severity_level::info);
		//Logger::initLogger(logprops);

		auto cuContext = apracucontext_sp(new ApraCUcontext());
		uint32_t gopLength = 25;
		uint32_t bitRateKbps = 1000;
		uint32_t frameRate = 30;
		H264EncoderNVCodecProps::H264CodecProfile profile = H264EncoderNVCodecProps::MAIN;
		bool enableBFrames = true;
		auto width = 640; //1920
		auto height = 360; //1020

		//WebCam
		WebCamSourceProps webCamSourceprops(0, 640, 360);
		webCamSourceprops.logHealth = true;
		webCamSourceprops.logHealthFrequency = 100;
		auto webCam = boost::shared_ptr<WebCamSource>(new WebCamSource(webCamSourceprops));

		//Color Conversion View
		auto colorProps1 = ColorConversionProps(ColorConversionProps::ConversionType::RGB_TO_BGR);
		colorProps1.logHealth = true;
		colorProps1.logHealthFrequency = 100;
		auto colorConvtView = boost::shared_ptr<ColorConversion>(new ColorConversion(colorProps1));
		//webCam->setNext(colorConvtView);

		//ImageViewer
		ImageViewerModuleProps imgViewerProps("NVR-View");
		imgViewerProps.logHealth = true;
		imgViewerProps.logHealthFrequency = 100;
		auto view = boost::shared_ptr<ImageViewerModule>(new ImageViewerModule(imgViewerProps));
		//colorConvtView->setNext(view);

		//Color Conversion to encoder
		auto colorProps2 = ColorConversionProps(ColorConversionProps::ConversionType::RGB_TO_YUV420PLANAR);
		colorProps2.logHealth = true;
		colorProps2.logHealthFrequency = 100;
		auto colorConvt = boost::shared_ptr<ColorConversion>(new ColorConversion(colorProps2));
		webCam->setNext(colorConvt); //WebCam->ColorConversion

		//Cuda Mem Copy
		cudastream_sp cudaStream_ = boost::shared_ptr<ApraCudaStream>(new ApraCudaStream());
		auto copyProps = CudaMemCopyProps(cudaMemcpyHostToDevice, cudaStream_);
		copyProps.logHealth = true;
		copyProps.logHealthFrequency = 100;
		auto copy = boost::shared_ptr<Module>(new CudaMemCopy(copyProps));
		colorConvt->setNext(copy);

		//H264 Encoder
		auto encoderProps = H264EncoderNVCodecProps(bitRateKbps, cuContext, gopLength, frameRate, profile, enableBFrames);
		encoderProps.logHealth = true;
		encoderProps.logHealthFrequency = 100;
		auto encoder = boost::shared_ptr<H264EncoderNVCodec>(new H264EncoderNVCodec(encoderProps));
		copy->setNext(encoder);

		//MP4 Writer-1 (24/7 writer)
		std::string outFolderPath_1 = "./data/testOutput/mp4_videos/24bpp/";
		auto mp4WriterSinkProps_1 = Mp4WriterSinkProps(1, 10, 24, outFolderPath_1);
		mp4WriterSinkProps_1.logHealth = true;
		mp4WriterSinkProps_1.logHealthFrequency = 100;
		auto mp4Writer_1 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_1));
		encoder->setNext(mp4Writer_1);

		//MultimediaQueue 
		auto multiProps = MultimediaQueueXformProps(120000, 30000, true);
		multiProps.logHealth = true;
		multiProps.logHealthFrequency = 100;
		multiProps.fps = 30;
		auto multiQue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(multiProps));
		encoder->setNext(multiQue);

		//auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/h264images/Raw_YUV420_640x360????.h264")));
		//multiQue->setNext(fileWriter);

		//MP4 Reader [Source]
		std::string startingVideoPath = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
		std::string outPath = "./data/testOutput/mp4_videos/24bpp";
		std::string changedVideoPath = "./data/testOutput/mp4_videos/24bpp/20221023/0017/";
		boost::filesystem::path file("frame_??????.h264");
		auto frameType = FrameMetadata::FrameType::H264_DATA;
		auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
		boost::filesystem::path dir(outPath);
		auto mp4ReaderProps = Mp4ReaderSourceProps(startingVideoPath, false, true);
		mp4ReaderProps.logHealth = true;
		mp4ReaderProps.logHealthFrequency = 100;
		mp4ReaderProps.fps = 30;
		auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
		mp4Reader->addOutPutPin(h264ImageMetadata);
		auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
		mp4Reader->addOutPutPin(mp4Metadata);

		//MP4 Writer-2 exports
		std::string outFolderPath_2 = "./data/testOutput/mp4_videos/ExportVids/";
		auto mp4WriterSinkProps_2 = Mp4WriterSinkProps(1, 10, 24, outFolderPath_2);
		mp4WriterSinkProps_2.logHealth = false;
		mp4WriterSinkProps_2.logHealthFrequency = 100;
		auto mp4Writer_2 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_2));
		multiQue->setNext(mp4Writer_2);
		boost::filesystem::path full_path = dir / file;
		LOG_INFO << full_path;
		mp4Reader->setNext(mp4Writer_2);

		//NVR ControlModule
		auto controlProps = NVRControlModuleProps();
		controlProps.logHealth = true;
		controlProps.logHealthFrequency = 100;
		auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(controlProps));
		Logger::setLogLevel(boost::log::trivial::severity_level::info);


		mControl->enrollModule("WebCamera", webCam);
		mControl->enrollModule("Renderer", view);
		mControl->enrollModule("Writer-1", mp4Writer_1);
		mControl->enrollModule("MultimediaQueue", multiQue);
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