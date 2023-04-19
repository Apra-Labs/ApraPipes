#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "Overlay.h"
#include "ExternalSourceModule.h"
#include "OverlayModule.h"
#include "ExternalSinkModule.h"
#include "FileReaderModule.h"
#include "FramesMuxer.h"

BOOST_AUTO_TEST_SUITE(overlaymodule_tests)

BOOST_AUTO_TEST_CASE(composite_overlay_test)
{
	CircleOverlay circleOverlay;
	circleOverlay.x1 = 50;
	circleOverlay.y1 = 75;
	circleOverlay.radius = 1;

	RectangleOverlay recOverlay;
	recOverlay.x1 = 100;
	recOverlay.x2 = 150;
	recOverlay.y1 = 125;
	recOverlay.y2 = 175;

	LineOverlay lineOverlay;
	lineOverlay.x1 = 300;
	lineOverlay.x2 = 325;
	lineOverlay.y1 = 350;
	lineOverlay.y2 = 375;


	RectangleOverlay recOverlay1;
	recOverlay1.x1 = 200;
	recOverlay1.x2 = 250;
	recOverlay1.y1 = 225;
	recOverlay1.y2 = 275;

	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::OVERLAY_INFO_IMAGE));
	auto pinId = source->addOutputPin(metadata);

	BOOST_TEST(source->init());

	frame_sp frame = source->makeFrame(2048, pinId);

	CompositeOverlay compositeOverlay1;
	compositeOverlay1.add(&recOverlay);

	CompositeOverlay compositeOverlay2;
	compositeOverlay2.add(&recOverlay1);
	compositeOverlay2.add(&circleOverlay);

	compositeOverlay1.add(&compositeOverlay2);
	compositeOverlay1.add(&lineOverlay);

	DrawingOverlay drawingSerilaizer;
	drawingSerilaizer.add(&compositeOverlay1);
	drawingSerilaizer.serialize(frame);
	
	DrawingOverlay drawingDes;
	drawingDes.deserialize(frame);

	for (auto shape : drawingDes.gList)
	{
		if (shape->primitiveType == Primitive::RECTANGLE)
		{
			RectangleOverlay* rectangleOverlayDes = static_cast<RectangleOverlay*>(shape);
			BOOST_TEST(rectangleOverlayDes->x1 == recOverlay.x1);
			BOOST_TEST(rectangleOverlayDes->y1 == recOverlay.y1);
			BOOST_TEST(rectangleOverlayDes->x2 == recOverlay.x2);
			BOOST_TEST(rectangleOverlayDes->y2 == recOverlay.y2);
		}

		else if (shape->primitiveType == Primitive::CIRCLE)
		{
			CircleOverlay* circleOverlayDes = static_cast<CircleOverlay*>(shape);
			BOOST_TEST(circleOverlayDes->x1 == circleOverlay.x1);
			BOOST_TEST(circleOverlayDes->y1 == circleOverlay.y1);
			BOOST_TEST(circleOverlayDes->radius == circleOverlay.radius);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()