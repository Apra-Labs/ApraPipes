#include "Mp4ReaderSource.h"
#include "TimelapsePipeline.h"
#include "H264Metadata.h"


TimelapsePipeline::TimelapsePipeline():pipeline("test") {
}


TimelapsePipeline::~TimelapsePipeline() {
}


bool TimelapsePipeline::setupPipeline() {
    bool overlayFrames = true;

	FileReaderModuleProps fileReaderProps("./data/h264_data/FVDO_Freeway_4cif_???.H264");
	fileReaderProps.fps = 30;
	fileReaderProps.readLoop = true;
	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(fileReaderProps));
	auto h264ImageMetadata = framemetadata_sp(new H264Metadata(0, 0));
	fileReader->addOutputPin(h264ImageMetadata);

	auto motionExtractor = boost::shared_ptr<MotionVectorExtractor>(new MotionVectorExtractor(MotionVectorExtractorProps(MotionVectorExtractorProps::OPENH264, overlayFrames)));
	fileReader->setNext(motionExtractor);

	auto overlay = boost::shared_ptr<OverlayModule>(new OverlayModule(OverlayModuleProps()));
	motionExtractor->setNext(overlay);

	auto sink = boost::shared_ptr<Module>(new ImageViewerModule(ImageViewerModuleProps("MotionVectorsOverlay")));
	overlay->setNext(sink);

    pipeline.appendModule(fileReader);
	pipeline.init();
    return true
}

bool TimelapsePipeline::startPipeline() {
    pipeline.run_all_threaded();
    return true;
}

bool TimelapsePipeline::stopPipeline() {
    m_cameraPipeline.stop();
	m_cameraPipeline.term();
	m_cameraPipeline.wait_for_all();
    return true;
}