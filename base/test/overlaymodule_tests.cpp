#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "test_utils.h"
#include "Overlay.h"
#include "ExternalSourceModule.h"
#include "OverlayModule.h"
#include "ExternalSinkModule.h"
#include "FileReaderModule.h"
#include "FramesMuxer.h"
#include "FileWriterModule.h"

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

	CircleOverlay circleOverlay1;
	circleOverlay1.x1 = 300;
	circleOverlay1.y1 = 350;
	circleOverlay1.radius = 2;

	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::OVERLAY_INFO_IMAGE));
	auto pinId = source->addOutputPin(metadata);

	BOOST_TEST(source->init());

	CompositeOverlay compositeOverlay1;
	compositeOverlay1.add(&recOverlay);

	CompositeOverlay compositeOverlay2;
	compositeOverlay2.add(&recOverlay1);
	compositeOverlay2.add(&circleOverlay);

	compositeOverlay1.add(&compositeOverlay2);
	compositeOverlay1.add(&lineOverlay);

	DrawingOverlay drawingOverlay;
	drawingOverlay.add(&compositeOverlay1);
	drawingOverlay.add(&circleOverlay1);

	auto size = drawingOverlay.mGetSerializeSize();
	frame_sp frame = source->makeFrame(size, pinId);

	drawingOverlay.serialize(frame);

	DrawingOverlay drawingDes;
	drawingDes.deserialize(frame);

	auto list = drawingDes.getList();

	for (auto primitive1 : list)
	{
		if (primitive1->primitiveType == Primitive::COMPOSITE)
		{
			CompositeOverlay* mCompositeOverlay1 = static_cast<CompositeOverlay*>(primitive1);

			auto compositeList1 = mCompositeOverlay1->getList();

			for (auto primitive2 : compositeList1)
			{
				if (primitive2->primitiveType == Primitive::RECTANGLE)
				{
					RectangleOverlay* rectangleOverlayDes = static_cast<RectangleOverlay*>(primitive2);
					BOOST_TEST(rectangleOverlayDes->x1 == recOverlay.x1);
					BOOST_TEST(rectangleOverlayDes->y1 == recOverlay.y1);
					BOOST_TEST(rectangleOverlayDes->x2 == recOverlay.x2);
					BOOST_TEST(rectangleOverlayDes->y2 == recOverlay.y2);
				}

				if (primitive2->primitiveType == Primitive::COMPOSITE)
				{
					CompositeOverlay* mCompositeOverlay2 = static_cast<CompositeOverlay*>(primitive2);
					auto compositeList2 = mCompositeOverlay2->getList();

					for (auto primitive3 : compositeList2)
					{
						if (primitive3->primitiveType == Primitive::RECTANGLE)
						{
							RectangleOverlay* rectangleOverlayDes1 = static_cast<RectangleOverlay*>(primitive3);
							BOOST_TEST(rectangleOverlayDes1->x1 == recOverlay1.x1);
							BOOST_TEST(rectangleOverlayDes1->y1 == recOverlay1.y1);
							BOOST_TEST(rectangleOverlayDes1->x2 == recOverlay1.x2);
							BOOST_TEST(rectangleOverlayDes1->y2 == recOverlay1.y2);
						}

						if (primitive3->primitiveType == Primitive::CIRCLE)
						{
							CircleOverlay* circleOverlayDes1 = static_cast<CircleOverlay*>(primitive3);
							BOOST_TEST(circleOverlayDes1->x1 == circleOverlay.x1);
							BOOST_TEST(circleOverlayDes1->y1 == circleOverlay.y1);
							BOOST_TEST(circleOverlayDes1->radius == circleOverlay.radius);
						}
					}
				}

				if (primitive2->primitiveType == Primitive::LINE)
				{
					LineOverlay* lineOverlayDes = static_cast<LineOverlay*>(primitive2);
					BOOST_TEST(lineOverlayDes->x1 == lineOverlay.x1);
					BOOST_TEST(lineOverlayDes->y1 == lineOverlay.y1);
					BOOST_TEST(lineOverlayDes->x2 == lineOverlay.x2);
					BOOST_TEST(lineOverlayDes->y2 == lineOverlay.y2);
				}
			}
		}

		else if (primitive1->primitiveType == Primitive::CIRCLE)
		{
			CircleOverlay* circleOverlayDes2 = static_cast<CircleOverlay*>(primitive1);
			BOOST_TEST(circleOverlayDes2->x1 == circleOverlay1.x1);
			BOOST_TEST(circleOverlayDes2->y1 == circleOverlay1.y1);
			BOOST_TEST(circleOverlayDes2->radius == circleOverlay1.radius);
		}
	}
}

BOOST_AUTO_TEST_CASE(drawing_test)
{
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

	// it is a square
	RectangleOverlay compositeRec;
	compositeRec.x1 = 1000;
	compositeRec.x2 = 1050;
	compositeRec.y1 = 170;
	compositeRec.y2 = 220;
	BOOST_TEST((compositeRec.y2 - compositeRec.y1) == (compositeRec.x2 - compositeRec.x1));

	CircleOverlay compositeCircle;
	compositeCircle.x1 = compositeRec.x1 + (compositeRec.x2 - compositeRec.x1) / 2;
	compositeCircle.y1 = compositeRec.y1 + (compositeRec.y2 - compositeRec.y1) / 2;
	compositeCircle.radius = (compositeRec.y2 - compositeRec.y1) / 2;

	CircleOverlay circleOverlay1;
	circleOverlay1.x1 = 500;
	circleOverlay1.y1 = 600;
	circleOverlay1.radius = 12;

	CompositeOverlay compositeOverlay1;
	compositeOverlay1.add(&recOverlay);

	CompositeOverlay compositeOverlay2;
	compositeOverlay2.add(&compositeRec);
	compositeOverlay2.add(&compositeCircle);

	compositeOverlay1.add(&compositeOverlay2);
	compositeOverlay1.add(&lineOverlay);

	DrawingOverlay drawingOverlay;
	drawingOverlay.add(&compositeOverlay1);
	drawingOverlay.add(&circleOverlay1);

	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::OVERLAY_INFO_IMAGE));
	auto pinId = source->addOutputPin(metadata);

	auto fileReader = boost::shared_ptr<FileReaderModule>(new FileReaderModule(FileReaderModuleProps("./data/frame_1280x720_rgb.raw")));
	auto rawMetadata = framemetadata_sp(new RawImageMetadata(1280, 720, ImageMetadata::ImageType::RGB, CV_8UC3, 0, CV_8U, FrameMetadata::HOST, true));
	auto rawPinId = fileReader->addOutputPin(rawMetadata);

	auto muxer = boost::shared_ptr<Module>(new FramesMuxer());
	source->setNext(muxer);
	fileReader->setNext(muxer);

	auto overlay = boost::shared_ptr<OverlayModule>(new OverlayModule(OverlayModuleProps()));
	muxer->setNext(overlay);

	auto fileWriter = boost::shared_ptr<Module>(new FileWriterModule(FileWriterModuleProps("./data/testOutput/Overlay/OverlayImage.raw")));
	overlay->setNext(fileWriter);

	auto size = drawingOverlay.getSerializeSize();
	frame_sp frame = source->makeFrame(size, pinId);

	drawingOverlay.serialize(frame);

	frame_container frames;
	frames.insert(make_pair(pinId, frame));

	BOOST_TEST(source->init());
	BOOST_TEST(fileReader->init());
	BOOST_TEST(muxer->init());
	BOOST_TEST(overlay->init());
	BOOST_TEST(fileWriter->init());

	source->send(frames);
	source->step();
	fileReader->step();
	muxer->step();
	muxer->step();
	overlay->step();
	fileWriter->step();
}
BOOST_AUTO_TEST_SUITE_END()