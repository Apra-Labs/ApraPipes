#include <boost/test/unit_test.hpp>

#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"

#include "FrameMetadata.h"
#include "Frame.h"
#include "Split.h"

BOOST_AUTO_TEST_SUITE(split_tests)

BOOST_AUTO_TEST_CASE(basic)
{	
	size_t readDataSize = 1024;
		
	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));

	auto source = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto source_pin = source->addOutputPin(metadata);
	
	SplitProps props;
	props.number = 3;
	auto split = boost::shared_ptr<Split>(new Split(props));
	source->setNext(split);
		
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule());
	split->setNext(sink);

	BOOST_TEST(source->init());
	BOOST_TEST(split->init());
	BOOST_TEST(sink->init());

	auto splitPinIds = split->getAllOutputPinsByType(FrameMetadata::GENERAL);
	BOOST_TEST(splitPinIds.size() == props.number);

	{		
		auto encodedImageFrame = source->makeFrame(readDataSize, metadata);		
		frame_container frames;
		frames.insert(make_pair(source_pin, encodedImageFrame));

		for (auto i = 0; i < 20; i++)
		{
			source->send(frames);
			split->step();
			BOOST_TEST(sink->pop().begin()->first == splitPinIds[i%props.number]);
		}
	}

	BOOST_TEST(source->term());
	BOOST_TEST(split->term());
	BOOST_TEST(sink->term());		
}

BOOST_AUTO_TEST_CASE(errors)
{
	size_t readDataSize = 1024;

	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));

	auto source1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	source1->addOutputPin(metadata);
	source1->addOutputPin(metadata);

	auto source2 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	source2->addOutputPin(metadata);

	auto source3 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	source3->addOutputPin(metadata);


	SplitProps props;
	props.number = 3;
	{
		auto split = boost::shared_ptr<Split>(new Split(props));
		source2->setNext(split);
	}

	try
	{
		auto split = boost::shared_ptr<Split>(new Split(props));
		source1->setNext(split); // only 1 input pins
		BOOST_TEST(false);
	}
	catch (...)
	{

	}

	try
	{
		auto split = boost::shared_ptr<Split>(new Split(props));
		source2->setNext(split);
		source3->setNext(split); // only 1 module and 1 input pin
		BOOST_TEST(false);
	}
	catch (...)
	{

	}
}

BOOST_AUTO_TEST_SUITE_END()