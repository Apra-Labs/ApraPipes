#include "stdafx.h"
#include "FrameMetadata.h"
#include "PipeLine.h"
#include "H264Metadata.h"
#include "ImageViewerModule.h"
#include "Logger.h"
#include "RTSPClientSrc.h"
#include <H264Decoder.h>
#include <ColorConversionXForm.h>
#include "KeyboardListener.h"
#include <Mp4ReaderSource.h>
#include "Mp4VideoMetadata.h"
#include "relay_sample.h"

RelayPipeline::RelayPipeline() : pipeline("RelaySample") {}

bool RelayPipeline::setupPipeline() {
    // RTSP
    LOG_INFO << "Please provide the RTSP camera URL.." << endl;
    string input;
    getline(cin, input);
    auto url = std::string(input);
    auto rtspSource = boost::shared_ptr<Module>(new RTSPClientSrc(RTSPClientSrcProps(url, "", "")));
    auto rtspMetaData = framemetadata_sp(new H264Metadata(1280, 720));
    rtspSource->addOutputPin(rtspMetaData);

    auto rtspDecoder = boost::shared_ptr<Module>(new H264Decoder(H264DecoderProps()));
    rtspSource->setNext(rtspDecoder);
    rtspDecoder->setNext(this->rtspColorConversion);
    this->rtspColorConversion->setNext(this->sink);

    // MP4
    std::string videoPath = "../.././data/1714992199120.mp4";
    bool parseFS = false;
    auto h264ImageMetadata = framemetadata_sp(new H264Metadata(1280, 720));
    auto frameType = FrameMetadata::FrameType::H264_DATA;
    auto mp4ReaderProps = Mp4ReaderSourceProps(videoPath, parseFS, 0, true, true, false);
    mp4ReaderProps.fps = 9;
    auto mp4Reader = boost::shared_ptr<Mp4ReaderSource>(new Mp4ReaderSource(mp4ReaderProps));
    mp4Reader->addOutPutPin(h264ImageMetadata);
    auto mp4Metadata = framemetadata_sp(new Mp4VideoMetadata("v_1"));
    mp4Reader->addOutPutPin(mp4Metadata);
    std::vector<std::string> mImagePin;
    mImagePin = mp4Reader->getAllOutputPinsByType(frameType);
    auto mp4Decoder = boost::shared_ptr<H264Decoder>(new H264Decoder(H264DecoderProps()));
    mp4Reader->setNext(mp4Decoder, mImagePin);
    mp4Decoder->setNext(this->mp4ColorConversion);
    this->mp4ColorConversion->setNext(this->sink);
    pipeline.appendModule(rtspSource);
    pipeline.appendModule(mp4Reader);
    pipeline.init();

    return true;
}

bool RelayPipeline::startPipeline() {
    pipeline.run_all_threaded();
    return true;
}

bool RelayPipeline::stopPipeline() {
    pipeline.stop();
    pipeline.term();
    pipeline.wait_for_all();
    return true;
}

int main() {
    LoggerProps loggerProps;
    loggerProps.logLevel = boost::log::trivial::severity_level::info;
    Logger::setLogLevel(boost::log::trivial::severity_level::info);
    Logger::initLogger(loggerProps);

    RelayPipeline pipelineInstance;

    pipelineInstance.rtspColorConversion = boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)));
    pipelineInstance.mp4ColorConversion =boost::shared_ptr<ColorConversion>(new ColorConversion(ColorConversionProps(ColorConversionProps::YUV420PLANAR_TO_RGB)));
    pipelineInstance.sink = boost::shared_ptr<ImageViewerModuleExtended>(new ImageViewerModuleExtended(ImageViewerModuleProps("Relay Sample")));

    if (!pipelineInstance.setupPipeline()) {
        std::cerr << "Failed to setup pipeline." << std::endl;
        return 1;
    }

    if (!pipelineInstance.startPipeline()) {
        std::cerr << "Failed to start pipeline." << std::endl;
        return 1;
    }
    pipelineInstance.mp4ColorConversion->relay(pipelineInstance.sink, false);
	while (true) {
		int k = getchar();
          if (k == 114) {
			  pipelineInstance.mp4ColorConversion->relay(pipelineInstance.sink, true);
			  pipelineInstance.rtspColorConversion->relay(pipelineInstance.sink, false);
          }
          if (k == 108) {
			  pipelineInstance.mp4ColorConversion->relay(pipelineInstance.sink,false);
			  pipelineInstance.rtspColorConversion->relay(pipelineInstance.sink,true);
          }
          if (k == 115) {
			  pipelineInstance.stopPipeline();
			  break;

          }
    }

    return 0;
}
