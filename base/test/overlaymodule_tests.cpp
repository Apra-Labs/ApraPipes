#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "OverlayDataInfo.h"
#include "ExternalSourceModule.h"
#include "OverlayModule.h"
#include "ExternalSinkModule.h"
#include "FileReaderModule.h"
#include "FramesMuxer.h"

BOOST_AUTO_TEST_SUITE(overlaymodule_tests)

BOOST_AUTO_TEST_CASE(overlay_rectangle_test)
{
	RectangleOverlay recOverlay;

	recOverlay.x1 = 100;
	recOverlay.x2 = 150;
	recOverlay.y1 = 125;
	recOverlay.y2 = 175;


	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::OVERLAY_INFO_IMAGE));
	auto pinId = source->addOutputPin(metadata);

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
	auto metadata1 = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::RGB, CV_8UC3, 1280 * 3, CV_8U, FrameMetadata::HOST));
	auto rawImagePin = fileReader->addOutputPin(metadata1);

	auto muxer = boost::shared_ptr<Module>(new FramesMuxer());
	source->setNext(muxer);
	fileReader->setNext(muxer);

	auto overlay = boost::shared_ptr<OverlayModule>(new OverlayModule(OverlayModuleProps()));
	muxer->setNext(overlay);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	overlay->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(fileReader->init());
	BOOST_TEST(muxer->init());
	BOOST_TEST(overlay->init());
	BOOST_TEST(sink->init());


	frame_sp frame = source->makeFrame(recOverlay.getSerializeSize(),pinId);
	recOverlay.serialize(frame->data(), recOverlay.getSerializeSize());

	frame_container frames;
	frames.insert(make_pair(pinId, frame));
	
	source->send(frames);
	fileReader->step();
	muxer->step();
	muxer->step();
	overlay->step();
	frames = sink->try_pop();

}


BOOST_AUTO_TEST_SUITE_END()