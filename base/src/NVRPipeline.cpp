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
#include "CudaMemCopy.h"
#include "CudaStreamSynchronize.h"
#include "H264EncoderNVCodec.h"
#include "ColorConversionXForm.h"
#include "ImageViewerModule.h"
#include "H264Metadata.h"
#include "Mp4VideoMetadata.h"
#include "NVRControlModule.h"
#include "FileReaderModule.h"
#include "EglRenderer.h"
#include <boost/filesystem.hpp>
#include "NvV4L2Camera.h" //Jetson
#include "NvTransform.h"
#include "H264DecoderV4L2Helper.h"
#include "H264EncoderV4L2.h"
#include "DMAFDToHostCopy.h"
#include "H264Decoder.h"
#include "FrameMetadata.h"
#include "DMAFrameUtils.h"
#include "Frame.h"
#include "DMAFDToHostCopy.h"
#include "H264Utils.h"
#include "Mp4ReaderSource.h"
#include "RTSPClientSrc.h"
#include "Mp4VideoMetadata.h"
#include "Mp4WriterSink.h"
#include "Mp4WriterSinkUtils.h"
#include "EncodedImageMetadata.h"
#include "Module.h"
#include "PropsChangeMetadata.h"
#include "MultimediaQueueXform.h"
#include "ValveModule.h"
#include "H264Metadata.h"
#include "Command.h"
#include <boost/lexical_cast.hpp>
// #include <gtk/gtk.h>
// #include <gdk/gdkscreen.h>
// #include <cairo.h>
#include <math.h>
#include <ctype.h>
#include <bits/stdc++.h>
#include <stdlib.h>
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>
//#include <json/value.h>

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
	LoggerProps loggerProps;
	loggerProps.logLevel = boost::log::trivial::severity_level::info;
	loggerProps.enableFileLog = true;
	Logger::initLogger(loggerProps);

	//RTSP Source
	auto url=string("rtsp://vsi1.blub0x.com:5544/a0dce344-929b-4703-bd40-035c98572526");
	string empty;
	auto rtspProps = RTSPClientSrcProps(url, empty, empty);
	//rtspProps.logHealth = true;
	//rtspProps.logHealthFrequency = 100; 
	rtspProps.fps = 18;
	mLive = boost::shared_ptr<RTSPClientSrc>(new RTSPClientSrc(rtspProps));
	auto meta = framemetadata_sp(new H264Metadata());
	mLive->addOutputPin(meta);

	//Decoder - 1
	decoder_1 = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));//
	mLive->setNext(decoder_1);

	//EGL Renderer - 1
	renderer_1 = boost::shared_ptr<EglRenderer>(new EglRenderer(EglRendererProps(0, 0, 720, 480)));
	decoder_1->setNext(renderer_1);

	//MP4-Writer - 1 [24/7]
	std::string outFolderPath_1 = "./data/testOutput/mp4_videos/24bpp/";
	auto mp4WriterSinkProps_1 = Mp4WriterSinkProps(1, 10, 24, outFolderPath_1);
	//mp4WriterSinkProps_1.logHealth = true;
	mp4WriterSinkProps_1.logHealthFrequency = 100;
	mp4WriterSinkProps_1.fps = 30;
	mp4writer_1 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(Mp4WriterSinkProps(1, 10, 24, outFolderPath_1)));
	mLive->setNext(mp4writer_1);

	//MultimediaQueue
	auto multiProps = MultimediaQueueXformProps(60000, 30000, true);
	//multiProps.logHealth = true;
	multiProps.logHealthFrequency = 100;
	multiProps.fps = 30;
	multiQue = boost::shared_ptr<MultimediaQueueXform>(new MultimediaQueueXform(multiProps));
	mLive->setNext(multiQue);
	
	//MP4-Writer 2 [Export]
	std::string outFolderPath_2 = "./data/testOutput/mp4_videos/Export_Videos/";
	auto mp4WriterSinkProps_2 = Mp4WriterSinkProps(60, 10, 24, outFolderPath_2);
	//mp4WriterSinkProps_2.logHealth = true;
	mp4WriterSinkProps_2.logHealthFrequency = 100;
	mp4WriterSinkProps_2.fps = 30;
	mp4writer_2 = boost::shared_ptr<Mp4WriterSink>(new Mp4WriterSink(mp4WriterSinkProps_2));
	multiQue->setNext(mp4writer_2);
	
	//MP4 Reader - 1
	std::string startingVideoPath_1 = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	std::string outPath_1 = "./data/testOutput/mp4_videos/24bpp";
	std::string changedVideoPath_1 = "./data/testOutput/mp4_videos/24bpp/20221023/0011/";                              
	boost::filesystem::path file1("frame_??????.h264");
	auto frameType_1 = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata_1 = framemetadata_sp(new H264Metadata(0, 0));
	boost::filesystem::path dir1(outPath_1);
	auto mp4ReaderProps_1 = Mp4ReaderSourceProps(startingVideoPath_1, false, false);
	//mp4ReaderProps_1.logHealth = true;
	mp4ReaderProps_1.logHealthFrequency = 100;
	mp4ReaderProps_1.fps = 30;
	mp4Reader_1 = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps_1));
	mp4Reader_1->addOutPutPin(h264ImageMetadata_1);
	auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader_1->addOutPutPin(mp4Metadata);
	mp4Reader_1->setNext(mp4writer_2);


	//Disk -> Decoder -> rendering branch
	//MP4 Reader - 2
	std::string startingVideoPath_2 = "./data/Mp4_videos/h264_video/20221010/0012/1668064027062.mp4";
	std::string outPath_2 = "./data/testOutput/mp4_videos/24bpp";
	std::string changedVideoPath_2 = "./data/testOutput/mp4_videos/24bpp/20221023/0011/";                              
	boost::filesystem::path file2("frame_??????.h264");
	auto frameType_2 = FrameMetadata::FrameType::H264_DATA;
	auto h264ImageMetadata_2 = framemetadata_sp(new H264Metadata(0, 0));
	boost::filesystem::path dir2(outPath_2);
	auto mp4ReaderProps_2 = Mp4ReaderSourceProps(startingVideoPath_2, false, false);
	//mp4ReaderProps_2.logHealth = true;
	mp4ReaderProps_2.logHealthFrequency = 100;
	mp4ReaderProps_2.fps = 30;
	mp4Reader_2 = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps_2));
	mp4Reader_2->addOutPutPin(h264ImageMetadata_2);
	auto mp4Metadata_2 = framemetadata_sp(new Mp4VideoMetadata("v_1"));
	mp4Reader_2->addOutPutPin(mp4Metadata_2);

	//H264-V4L2 Decoder
	auto decoder_2 = boost::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));//
	mp4Reader_2->setNext(decoder_2);

	//EGL Renderer
	auto renderer_2 = boost::shared_ptr<Module>(new EglRenderer(EglRendererProps(300, 300, 360, 240)));
	decoder_2->setNext(renderer_2);


	//ControlModule 
	auto controlProps = NVRControlModuleProps();
	controlProps.logHealth = true;
	controlProps.logHealthFrequency = 100;
	controlProps.fps = 30;
	auto mControl = boost::shared_ptr<NVRControlModule>(new NVRControlModule(controlProps));

    //p("test");
	//std::thread inp(key_Read_func, std::ref(mControl), std::ref(mp4Reader_1));
	p->appendModule(mLive);
	p->appendModule(mp4Reader_1);
	p->appendModule(mp4Reader_2);
	p->addControlModule(mControl);

	mControl->enrollModule("Reader_1", mp4Reader_1);
	mControl->enrollModule("Reader_2", mp4Reader_2);
	mControl->enrollModule("Renderer", renderer_1);
	mControl->enrollModule("Writer-1", mp4writer_1);
	mControl->enrollModule("MultimediaQueue", multiQue);
	mControl->enrollModule("Writer-2", mp4writer_2);


	BOOST_TEST(p->init());
	mControl->init();
	mp4Reader_1->play(false);
	mp4Reader_2->play(false);

	p->run_all_threaded();
	boost::this_thread::sleep_for(boost::chrono::seconds(240));
	for (const auto& folder : boost::filesystem::recursive_directory_iterator(boost::filesystem::path("./data/testOutput/mp4_videos/24bpp/20230009/0013/")))
	{
		if (boost::filesystem::is_regular_file(folder))
		{
			boost::filesystem::path p = folder.path();

			changedVideoPath_1 = p.string();
			changedVideoPath_2 = p.string();
			LOG_ERROR<<"|-|-|-|-START OPERATIONS-|-|-|-|";
			break;
		}
	}
	Mp4ReaderSourceProps propsChange_1(changedVideoPath_1, true);
	Mp4ReaderSourceProps propsChange_2(changedVideoPath_2, true);
	mp4Reader_1->setProps(propsChange_1);
	mp4Reader_2->setProps(propsChange_2);
	boost::this_thread::sleep_for(boost::chrono::seconds(3600));
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

	bool startView()
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
		mControl->nvrView(true);
		mControl->step();
		return true;
	}

	bool stopView()
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
		mControl->nvrView(false);
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
		//inp.join();
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
	boost::shared_ptr<RTSPClientSrc>mLive;
	boost::shared_ptr<H264Decoder>decoder_1;
	boost::shared_ptr<H264Decoder>decoder_2;
	boost::shared_ptr<EglRenderer>renderer_1;
	boost::shared_ptr<EglRenderer>renderer_2;
	boost::shared_ptr<MultimediaQueueXform>multiQue;
	boost::shared_ptr<Mp4WriterSink>mp4writer_1;
	boost::shared_ptr<Mp4WriterSink>mp4writer_2;
	boost::shared_ptr<Mp4ReaderSource>mp4Reader_1;
	boost::shared_ptr<Mp4ReaderSource>mp4Reader_2;
	boost::shared_ptr<NVRControlModule>mControl;
	boost::thread inp;
	GtkWidget *playButton;
	GtkWidget *pauseButton;
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

bool NVRPipeline::startView()
{
	return mDetail->startView();
}

bool NVRPipeline::stopView()
{
	return mDetail->stopView();
}

bool NVRPipeline::xport(uint64_t TS, uint64_t TE)
{
	return mDetail->xport(TS, TE);
}