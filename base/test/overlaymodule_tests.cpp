#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "OverlayDataInfo.h"
#include "ExternalSourceModule.h"
#include "OverlayModule.h"
#include "ExternalSinkModule.h"
#include "FileReaderModule.h"
#include "FramesMuxer.h"

BOOST_AUTO_TEST_SUITE(overlaymodule_tests)

BOOST_AUTO_TEST_CASE(composite_overlay_test)
{
	RectangleOverlay recOverlay;
	recOverlay.x1 = 100;
	recOverlay.x2 = 150;
	recOverlay.y1 = 125;
	recOverlay.y2 = 175;

	CircleOverlay circleOverlay;
	circleOverlay.x1 = 50;
	circleOverlay.y1 = 75;
	circleOverlay.radius = 1;

	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::OVERLAY_INFO_IMAGE));
	auto pinId = source->addOutputPin(metadata);

	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	source->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(sink->init());

	frame_sp frame = source->makeFrame(2048, pinId);

	CompositeOverlay compositeOverlay;
	compositeOverlay.add(&recOverlay);
	compositeOverlay.add(&circleOverlay);
	compositeOverlay.serialize(frame);

	
	CompositeOverlay compositeOverlayDes;
	compositeOverlayDes.deserialize(frame);
}

BOOST_AUTO_TEST_SUITE_END()