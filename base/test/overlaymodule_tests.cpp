#include "stdafx.h"
#include <boost/test/unit_test.hpp>
#include "test_utils.h"
#include "Overlay.h"
#include "ExternalSourceModule.h"
#include "OverlayModule.h"
#include "ExternalSinkModule.h"
#include "FileReaderModule.h"
#include "FramesMuxer.h"

BOOST_AUTO_TEST_SUITE(overlaymodule_tests)

class ExternalSourceProps : public ModuleProps
{
public:
	ExternalSourceProps() : ModuleProps() {};

};

class ExternalSource : public Module
{

public:
	DrawingOverlay drawingOverlay;
	size_t msize;
	ExternalSource(ExternalSourceProps props) : Module(SOURCE, "ExternalSource", props)
	{
		CircleOverlay circleOverlay;
		circleOverlay.x1 = 1024;
		circleOverlay.y1 = 768;
		circleOverlay.radius = 100;

		RectangleOverlay recOverlay;
		recOverlay.x1 = 100;
		recOverlay.x2 = 500;
		recOverlay.y1 = 200;
		recOverlay.y2 = 400;

		LineOverlay lineOverlay;
		lineOverlay.x1 = 100;
		lineOverlay.x2 = 500;
		lineOverlay.y1 = 200;
		lineOverlay.y2 = 400;

		drawingOverlay.add(&circleOverlay);
		drawingOverlay.add(&recOverlay);
		drawingOverlay.add(&lineOverlay);

	    msize = drawingOverlay.mGetSerializeSize();  
	}

	boost::shared_ptr<FrameContainerQueue> getQue() { return Module::getQue(); }
	frame_sp makeFrame(size_t size, string& pinId) { return Module::makeFrame(size, pinId); }
	bool send(frame_container& frames) { return Module::send(frames); }
	
protected:
	bool process()
	{
		return false;
	}

	bool produce() override
	{
		auto metadata1 = framemetadata_sp(new FrameMetadata(FrameMetadata::OVERLAY_INFO_IMAGE));

		size_t fsize = msize;
		std::string fPinId1 = getOutputPinIdByType(FrameMetadata::FrameType::OVERLAY_INFO_IMAGE);
		auto dataframe = makeFrame(fsize, fPinId1);
		drawingOverlay.serialize(dataframe);
		
		const uint8_t* pReadData = nullptr;
		unsigned int readDataSize = 0U;
		BOOST_TEST(Test_Utils::readFile("./data/faces.raw", pReadData, readDataSize));
		std::string fPinId2 = getOutputPinIdByType(FrameMetadata::FrameType::RAW_IMAGE);
		auto imageframe = makeFrame(readDataSize, fPinId2);

		frame_container frames;
		frames.insert(std::make_pair(fPinId1, dataframe));
		frames.insert(std::make_pair(fPinId2, imageframe));

		send(frames);

		return true;
	}

	bool validateOutputPins()
	{
		return true;
	}
	bool validateInputPins()
	{
		return true;
	}

};

boost::shared_ptr<ExternalSource> source;

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
			CompositeOverlay *mCompositeOverlay1 = static_cast<CompositeOverlay *>(primitive1);

			auto compositeList1 = mCompositeOverlay1->getList();

			for (auto primitive2 : compositeList1)
			{
				if (primitive2->primitiveType == Primitive::RECTANGLE)
				{
					RectangleOverlay *rectangleOverlayDes = static_cast<RectangleOverlay *>(primitive2);
					BOOST_TEST(rectangleOverlayDes->x1 == recOverlay.x1);
					BOOST_TEST(rectangleOverlayDes->y1 == recOverlay.y1);
					BOOST_TEST(rectangleOverlayDes->x2 == recOverlay.x2);
					BOOST_TEST(rectangleOverlayDes->y2 == recOverlay.y2);
				}

				if (primitive2->primitiveType == Primitive::COMPOSITE)
				{
					CompositeOverlay *mCompositeOverlay2 = static_cast<CompositeOverlay *>(primitive2);
					auto compositeList2 = mCompositeOverlay2->getList();

					for (auto primitive3 : compositeList2)
					{
						if (primitive3->primitiveType == Primitive::RECTANGLE)
						{
							RectangleOverlay *rectangleOverlayDes1 = static_cast<RectangleOverlay *>(primitive3);
							BOOST_TEST(rectangleOverlayDes1->x1 == recOverlay1.x1);
							BOOST_TEST(rectangleOverlayDes1->y1 == recOverlay1.y1);
							BOOST_TEST(rectangleOverlayDes1->x2 == recOverlay1.x2);
							BOOST_TEST(rectangleOverlayDes1->y2 == recOverlay1.y2);
						}

						if (primitive3->primitiveType == Primitive::CIRCLE)
						{
							CircleOverlay *circleOverlayDes1 = static_cast<CircleOverlay *>(primitive3);
							BOOST_TEST(circleOverlayDes1->x1 == circleOverlay.x1);
							BOOST_TEST(circleOverlayDes1->y1 == circleOverlay.y1);
							BOOST_TEST(circleOverlayDes1->radius == circleOverlay.radius);
						}
					}
				}

				if (primitive2->primitiveType == Primitive::LINE)
				{
					LineOverlay *lineOverlayDes = static_cast<LineOverlay *>(primitive2);
					BOOST_TEST(lineOverlayDes->x1 == lineOverlay.x1);
					BOOST_TEST(lineOverlayDes->y1 == lineOverlay.y1);
					BOOST_TEST(lineOverlayDes->x2 == lineOverlay.x2);
					BOOST_TEST(lineOverlayDes->y2 == lineOverlay.y2);
				}
			}
		}

		else if (primitive1->primitiveType == Primitive::CIRCLE)
		{
			CircleOverlay *circleOverlayDes2 = static_cast<CircleOverlay *>(primitive1);
			BOOST_TEST(circleOverlayDes2->x1 == circleOverlay1.x1);
			BOOST_TEST(circleOverlayDes2->y1 == circleOverlay1.y1);
			BOOST_TEST(circleOverlayDes2->radius == circleOverlay1.radius);
		}
	}
}

BOOST_AUTO_TEST_CASE(drawing_test)
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

	const uint8_t* pReadData = nullptr;
	unsigned int readDataSize = 0U;
	BOOST_TEST(Test_Utils::readFile("./data/mono.jpg", pReadData, readDataSize));

	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::OVERLAY_INFO_IMAGE));
	auto pinId = source->addOutputPin(metadata);

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

	memcpy(frame->data(), pReadData, readDataSize);
	frame_container frames;
	frames.insert(make_pair(pinId, frame));

	auto overlay = boost::shared_ptr<OverlayModule>(new OverlayModule(OverlayModuleProps()));
	source->setNext(overlay);

	BOOST_TEST(source->init());
	BOOST_TEST(overlay->init());

	source->send(frames);
	source->step();
	overlay->step();

}
BOOST_AUTO_TEST_CASE(mdrawing_test)
{
	auto source = boost::shared_ptr<ExternalSource>(new ExternalSource(ExternalSourceProps()));


	//drawingOverlay.serialize(frame);

	/*auto overlay = boost::shared_ptr<OverlayModule>(new OverlayModule(OverlayModuleProps()));
	source->setNext(overlay);*/

	/*auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	overlay->setNext(sink);*/

	BOOST_TEST(source->init());
	/*BOOST_TEST(overlay->init());
	BOOST_TEST(sink->init());*/

	source->step();
    //overlay->step();

}
BOOST_AUTO_TEST_SUITE_END()