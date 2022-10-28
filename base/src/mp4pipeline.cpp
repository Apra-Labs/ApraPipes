#include "mp4pipeline.h"
#include "PipeLine.h"
#include "Mp4ReaderSource.h"
#include "Mp4VideoMetadata.h"
#include "EncodedImageMetadata.h"
#include "JPEGDecoderIM.h"
#include "FrameContainerQueue.h"
#include "ProtoDeserializer.h"
#include "FramesMuxer.h"
#include "SaturationResult.h"
#include "DefectsInfo.h"

namespace Mp4PlayerInternal {

	// ExternalSink class
	class ExternalSinkProps : public ModuleProps
	{
	public:
		ExternalSinkProps() : ModuleProps()
		{
		}
	};
	class ExternalSink : public Module
	{
	public:
		ExternalSink(ExternalSinkProps props, MessageHandler dataHandler) : Module(SINK, "ExternalSink", props), sinkQ(capacity), mDataHandler(dataHandler)
		{
		}

		~ExternalSink() 
		{
		}

		std::string setDefectsInfoString(std::vector<DefectsInfo> &results)
		{
			std::string resStr = "";
			for (auto i = 0; i < results.size(); ++i)
			{
				DefectsInfo &res = results[i];
				// defectFound
				if (res.defectFound)
				{
					resStr += "isDefectFound <" + std::to_string(res.defectFound) + "> \n";
				}
				// defects
				for (auto j = 0; j < res.centers.size(); ++j)
				{
					// centers
					resStr += "Center <" + std::to_string(res.centers[j].x) + "," + std::to_string(res.centers[j].y) + "> \n";
					// radius
					resStr += "Radius <" + std::to_string(res.radius[j]) + "> \n";
					// defectsLenght
					resStr += "DefectsLength <" + std::to_string(res.defectsLength[j]) + "> \n";
				}
			}
			return resStr;
		}

		std::tuple <uint8_t*, std::string> peek(uint64_t& frameSize)
		{
			auto frameContainer = sinkQ.peek();
			frame_sp image = Module::getFrameByType(frameContainer, FrameMetadata::RAW_IMAGE);
			frame_sp defectsFrame = Module::getFrameByType(frameContainer, FrameMetadata::DEFECTS_INFO);
			frame_sp saturationFrame = Module::getFrameByType(frameContainer, FrameMetadata::SATURATION_RESULT);
			
			std::string sectionSplit = "\n===========================\n";
			std::string frameTimestamp = "Timestamp <" + std::to_string(image->timestamp) + "> msecs" + sectionSplit;
			std::string defectsInfoResultStr = frameTimestamp + "<DefectsInfo> " + sectionSplit;
			std::string saturationResultStr= "<SaturationResult> " + sectionSplit;

			if (!image.get())
			{
				frameSize = 0;
				return std::make_tuple(nullptr, "");
				LOG_ERROR << "image frame not recieved";
			}
			else
			{
				frameSize = image->size();
			}

			if (defectsFrame.get())
			{
				DefectsInfo res;
				std::vector<DefectsInfo> results;
				res.deSerialize(defectsFrame, results);
				defectsInfoResultStr += setDefectsInfoString(results);
			}

			if (saturationFrame.get())
			{
				std::vector<SaturationResult> results;
				SaturationResult::deSerialize(saturationFrame, results);
				if (results.size() > 0)
				{
					saturationResultStr += std::to_string(results[0].avgValue);
				}
			}

			std::string finalRes = defectsInfoResultStr + "\n" + saturationResultStr;
			resultContainer = std::make_tuple(reinterpret_cast<uint8_t*>(image->data()), finalRes);
			return resultContainer;
		}

		void pop_if_full()
		{
			if (sinkQ.size() >= capacity)
			{
				sinkQ.pop();
			}
		}

		void flushQue() override
		{
			LOG_ERROR << "flushing sinkQ";
			sinkQ.flush();
			Module::flushQue();
		}

	protected:
		bool process(frame_container &frames)
		{			
			sinkQ.try_push(frames);

			// raise an event for the view
			(*mDataHandler)();
			return true;
		}

		bool validateInputPins() 
		{
			LOG_ERROR << "externalSink <validateInputPins>";
			return true;
		}

		bool validateInputOutputPins()
		{
			LOG_ERROR << "externalSink <validateInputPins>";
			return true;
		}

	private:
		FrameContainerQueue sinkQ;
		MessageHandler mDataHandler;
		int capacity = 2;
		std::tuple<uint8_t*, std::string> resultContainer = { nullptr, "" };
	}; // ExternalSink

	// class Mp4Pipeline_Detail
	class Mp4Pipeline_Detail {
	public:
		Mp4Pipeline_Detail()
		{
		}

		std::tuple<bool, std::string> open(std::string videoPath)
		{
			stopPipelineIfNewSignature(videoPath);
			std::tuple<bool, std::string> res = std::make_tuple(true, "");

			if (!bPipelineStarted)
			{
				res = try_startInternal(videoPath); // bPipelineStarted should be false here, IMPORTANT
				bPipelineStarted = std::get<0>(res);
				LOG_INFO << "Pipeline started<>" << bPipelineStarted;
			}
			
			return res;
		}
		
		bool close()
		{
			return stopPipeline();
		}
		
		bool pause()
		{
			if (!bPipelineStarted)
			{
				return true;
			}
			if (videoPaused)
			{
				return true;
			}
			try 
			{
				p->pause();
				videoPaused = true;
			}
			catch (AIPException &ex)
			{
				LOG_ERROR << "Error occured while pausing <" + ex.getError() + ">";
				return false;
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
			if (!bPipelineStarted)
			{
				return true;
			}
			if (!videoPaused)
			{
				return true;
			}
			else
			{
				try
				{
					p->play();
					videoPaused = false;
				}
				catch (AIPException &ex)
				{
					LOG_ERROR << "Error occured while pausing <" + ex.getError() + ">";
					return false;
				}
				catch (...)
				{
					LOG_ERROR << "Error occured while pausing <>";
					return false;
				}
				return true;
			}
		}
		
		bool seek(uint64_t ts, std::string skipDir="")
		{
			if (!bPipelineStarted)
			{
				LOG_ERROR << "Pipeline not started";
				return false;
			}
			if (!skipDir.empty())
			{
				auto tempProps = mp4Reader->getProps();
				tempProps.skipDir = skipDir;
				mp4Reader->setProps(tempProps);
			}
			mp4Reader->flushQue();
			return mp4Reader->randomSeek(ts);
		}

		bool RegisterMetadataListener()//boost::shared_ptr<std::function> &method)
		{
			// call the method as metadata is found.
			return true;
		}

		std::tuple <uint8_t*, std::string> peekSinkQ(uint64_t& frameSize)
		{
			if (bPipelineStarted)
			{
				return sink->peek(frameSize);
			}
			else 
			{
				frameSize = 0;
				return std::make_tuple(nullptr, "");

			}
		}

		void popFromSinkQIfFull()
		{
			sink->pop_if_full();
		}

		void setDataListener(MessageHandler dataHandler)
		{
			mDataHandler = dataHandler;
		}

		bool nextFrame()
		{
			if (!bPipelineStarted)
			{
				return false;
			}
			pause();
			p->step();
			return true;
		}

		void getResolution(uint32_t &width, uint32_t &height)
		{
			mp4Reader->getResolution(width, height);
		}

	private:
		
		std::string getPipelineSignature(std::string &videoPath)
		{
			return videoPath;
		}
		
		bool startInternal(std::string &startingVideoPath)
		{
			LOG_ERROR << "startInternal entry<" << bPipelineStarted << ">";
			if (bPipelineStarted == false)
			{	
				LoggerProps loggerProps;
				loggerProps.logLevel = boost::log::trivial::severity_level::info;
				Logger::setLogLevel(boost::log::trivial::severity_level::info);
				Logger::initLogger(loggerProps);

				auto mp4ReaderProps = Mp4ReaderSourceProps(startingVideoPath, true);
				mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
				auto encodedImageMetadata = framemetadata_sp(new EncodedImageMetadata());
				auto encodedImagePin = mp4Reader->addOutputPin(encodedImageMetadata);
				auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_2_0"));
				auto mp4MetadataPin = mp4Reader->addOutputPin(mp4Metadata);

				framemetadata_sp rawImageMetadata;
				decoder = boost::shared_ptr<JPEGDecoderIM>(new JPEGDecoderIM(new JPEGDecoderIMProps()));
				rawImageMetadata = framemetadata_sp(new RawImageMetadata());
				auto rawImagePin = decoder->addOutputPin(rawImageMetadata);
				mp4Reader->setNext(decoder);

				auto protoDesProps = ProtoDeserializerProps();
				protoDes = boost::shared_ptr<ProtoDeserializer>(new ProtoDeserializer(protoDesProps));
				// sieve disabled
				decoder->setNext(protoDes, true, false);

				auto sinkProps = ExternalSinkProps();
				sinkProps.logHealth = true;
				sinkProps.logHealthFrequency = 1000;
				sink = boost::shared_ptr<ExternalSink>(new ExternalSink(sinkProps, mDataHandler));
				// sieve disabled
				protoDes->setNext(sink, true, false);

				p = boost::shared_ptr<PipeLine>(new PipeLine("mp4reader"));
				p->appendModule(mp4Reader);

				if (!p->init())
				{
					throw AIPException(AIP_FATAL, "Engine Pipeline init failed. Check IPEngine Logs for more details.");
				}
				p->run_all_threaded();
			}
			return true;
		}
		
		std::tuple<bool,std::string> try_startInternal(std::string videoPath)
		{
			pipelineSignature = getPipelineSignature(videoPath);
			LOG_ERROR << "pipSig <" << pipelineSignature << ">";
			try
			{
				bool runflag = false;
				if (!videoPath.empty())
				{
					startInternal(videoPath);
					runflag = true;
				}
				else
				{
					return std::make_tuple(runflag, "empty videoPath");
				}
				return std::make_tuple(runflag, "");
			}
			catch (AIPException &ex)
			{
				auto msg = handleStartEngineFailed();
				if (msg.empty())
				{
					msg = ex.getError();
				}
				return std::make_tuple(false, msg);
			}
			catch (...)
			{
				LOG_ERROR << "try_startInternal failed";
				auto msg = handleStartEngineFailed();
				if (msg.empty())
				{
					msg = "Engine start failed. Check IPEngine Logs for more details.";
				}
				return std::make_tuple(false, msg);
			}
		}
		
		bool stopPipeline()
		{
			if (bPipelineStarted == false)
			{
				return true;
			}
			bPipelineStarted = false; // make false before calling stopInternal, IMPORTANT
			videoPaused = false; // reset pause status in case video is opened again
			try_stopInternal(true);
			LOG_INFO << "Pipeline Stopped<>" << bPipelineStarted << std::endl;
			return true;
		}
		
		bool try_stopInternal(bool stopPipeline)
		{
			try
			{
				return stopInternal(stopPipeline);
			}
			catch (...)
			{
				LOG_ERROR << "try_stopInternal failed";
				return false;
			}
		}
		
		bool stopInternal(bool stopPipeline)
		{
			// stops process only when pipeline is not running
			if (bPipelineStarted == false)
			{
				p->stop();
				p->term();
				p->wait_for_all();
				p.reset();
				mp4Reader.reset();
				sink.reset();
				return true;
			}
			else
			{
				LOG_ERROR << "Unexpected error while stopping pipeline";
			}
			return false;
		}
		
		void stopPipelineIfNewSignature(std::string &videoPath)
		{
			string newSignature = getPipelineSignature(videoPath);
			if (pipelineSignature == newSignature)
			{
				return;
			}

			LOG_ERROR << "stopping pipeline because new source has benn configured. old<" << pipelineSignature << "> new<" << newSignature << ">";
			stopPipeline();
		}
		
		std::string handleStartEngineFailed()
		{
			if (try_stopInternal(false))
			{
				return "";
			}
			return "handleStartEngineFailed failed";
		}
		
		bool bPipelineStarted = false;
		bool videoPaused = false;
		std::string pipelineSignature = "_";
		boost::shared_ptr<PipeLine> p;
		boost::shared_ptr<Mp4ReaderSource> mp4Reader;
		boost::shared_ptr<Module> decoder;
		boost::shared_ptr<ProtoDeserializer> protoDes;
		boost::shared_ptr<Module> resultMuxer;
		boost::shared_ptr<ExternalSink> sink;
		MessageHandler mDataHandler;

	}; // Mp4Pipeline_Detail


	// Mp4Pipeline methods
	Mp4Pipeline::Mp4Pipeline()
	{
		mDetail = new Mp4Pipeline_Detail();
	}

	Mp4Pipeline::~Mp4Pipeline()
	{
		delete mDetail;
	}

	std::tuple<bool, std::string> Mp4Pipeline::open(std::string videoPath)
	{
		return mDetail->open(videoPath);
	}

	bool Mp4Pipeline::close()
	{
		return mDetail->close();
	}

	bool Mp4Pipeline::pause()
	{
		return mDetail->pause();
	}

	bool Mp4Pipeline::resume()
	{
		return mDetail->resume();
	}

	bool Mp4Pipeline::seek(uint64_t skipTS, std::string skipDir)
	{
		return mDetail->seek(skipTS, skipDir);
	}

	bool Mp4Pipeline::RegisterMetadataListener()
	{
		return mDetail->RegisterMetadataListener();
	}

	std::tuple <uint8_t*, std::string> Mp4Pipeline::peekSinkQ(uint64_t& frameSize)
	{
		return mDetail->peekSinkQ(frameSize);
	}

	void Mp4Pipeline::popFromSinkQIfFull()
	{
		mDetail->popFromSinkQIfFull();
	}

	void Mp4Pipeline::SetDataListener(MessageHandler dataHandler)
	{
		mDataHandler = dataHandler;
		mDetail->setDataListener(mDataHandler);
	}

	void Mp4Pipeline::getResolution(uint32_t &width, uint32_t &height)
	{
		mDetail->getResolution(width, height);
	}

	bool Mp4Pipeline::nextFrame()
	{
		return mDetail->nextFrame();
	}
}
