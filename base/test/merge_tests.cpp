#include <boost/test/unit_test.hpp>

#include "ExternalSourceModule.h"
#include "ExternalSinkModule.h"

#include "FrameMetadata.h"
#include "Frame.h"
#include "Merge.h"

BOOST_AUTO_TEST_SUITE(merge_tests)

BOOST_AUTO_TEST_CASE(basic)
{
	size_t readDataSize = 1024;

	auto metadata = framemetadata_sp(new FrameMetadata(FrameMetadata::GENERAL));

	auto source1 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto source_pin_1 = source1->addOutputPin(metadata);

	auto source2 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto source_pin_2 = source2->addOutputPin(metadata);

	auto source3 = boost::shared_ptr<ExternalSourceModule>(new ExternalSourceModule());
	auto source_pin_3 = source3->addOutputPin(metadata);
		
	auto merge = boost::shared_ptr<Merge>(new Merge());
	source1->setNext(merge);
	source2->setNext(merge);
	source3->setNext(merge);

	ExternalSinkModuleProps props;
	props.qlen = 50;
	auto sink = boost::shared_ptr<ExternalSinkModule>(new ExternalSinkModule(props));
	merge->setNext(sink);

	BOOST_TEST(source1->init());
	BOOST_TEST(source2->init());
	BOOST_TEST(source3->init());
	BOOST_TEST(merge->init());
	BOOST_TEST(sink->init());

	for (auto i = 0; i < 201; i++)
	{
		if (i % 3 == 0)
		{
			frame_container frames;
			auto encodedImageFrame_1 = source1->makeFrame(readDataSize, metadata);
			encodedImageFrame_1->fIndex2 = i;
			frames.insert(make_pair(source_pin_1, encodedImageFrame_1));
			source1->send(frames);
		}
		if (i % 3 == 1)
		{
			frame_container frames;
			auto encodedImageFrame_2 = source1->makeFrame(readDataSize, metadata);
			encodedImageFrame_2->fIndex2 = i;
			frames.insert(make_pair(source_pin_2, encodedImageFrame_2));
			source2->send(frames);
		}
		if (i % 3 == 2)
		{
			frame_container frames;
			auto encodedImageFrame_3 = source1->makeFrame(readDataSize, metadata);
			encodedImageFrame_3->fIndex2 = i;
			frames.insert(make_pair(source_pin_3, encodedImageFrame_3));
			source3->send(frames);
		}
		merge->step();
		BOOST_TEST(sink->pop().begin()->second->fIndex2 == i);
	}

	for (auto i = 201; i < 402; i++)
	{
		if (i % 3 == 0)
		{
			frame_container frames;
			auto encodedImageFrame_1 = source1->makeFrame(readDataSize, metadata);
			encodedImageFrame_1->fIndex2 = i+2;
			frames.insert(make_pair(source_pin_1, encodedImageFrame_1));
			source1->send(frames);
			merge->step();
			BOOST_TEST(sink->try_pop().size() == 0);
		}
		if (i % 3 == 1)
		{
			frame_container frames;
			auto encodedImageFrame_2 = source1->makeFrame(readDataSize, metadata);
			encodedImageFrame_2->fIndex2 = i;
			frames.insert(make_pair(source_pin_2, encodedImageFrame_2));
			source2->send(frames);
			merge->step();
			BOOST_TEST(sink->try_pop().size() == 0);
		}
		if (i % 3 == 2)
		{
			frame_container frames;
			auto encodedImageFrame_3 = source1->makeFrame(readDataSize, metadata);
			encodedImageFrame_3->fIndex2 = i-2;
			frames.insert(make_pair(source_pin_3, encodedImageFrame_3));
			source3->send(frames);
			merge->step();
			for (auto j = 2; j >= 0; j--)
			{
				BOOST_TEST(sink->pop().begin()->second->fIndex2 == i-j);
			}
		}		
	}

	// skipping 1 index - 402
	for (auto i = 403; i < 600; i++)
	{
		if (i % 3 == 0)
		{
			frame_container frames;
			auto encodedImageFrame_1 = source1->makeFrame(readDataSize, metadata);
			encodedImageFrame_1->fIndex2 = i;
			frames.insert(make_pair(source_pin_1, encodedImageFrame_1));
			source1->send(frames);
		}
		if (i % 3 == 1)
		{
			frame_container frames;
			auto encodedImageFrame_2 = source1->makeFrame(readDataSize, metadata);
			encodedImageFrame_2->fIndex2 = i;
			frames.insert(make_pair(source_pin_2, encodedImageFrame_2));
			source2->send(frames);
		}
		if (i % 3 == 2)
		{
			frame_container frames;
			auto encodedImageFrame_3 = source1->makeFrame(readDataSize, metadata);
			encodedImageFrame_3->fIndex2 = i;
			frames.insert(make_pair(source_pin_3, encodedImageFrame_3));
			source3->send(frames);
		}
		merge->step();
		if (i < 403 + 30)
		{
			BOOST_TEST(sink->try_pop().size() == 0);
		}
		else if (i == 403 + 30)
		{
			// everything will be flushed now
			for (auto j = 0; j < 31; j++)
			{
				BOOST_TEST(sink->pop().begin()->second->fIndex2 == 403+j);
			}
		}
		else
		{
			BOOST_TEST(sink->pop().begin()->second->fIndex2 == i);
		}
	}

	// skipping by 5
	for (auto i = 600; i < 780; i++)
	{
		auto fIndex2 = i + (i - 600) * 5;
		if (i % 3 == 0)
		{
			frame_container frames;
			auto encodedImageFrame_1 = source1->makeFrame(readDataSize, metadata);
			encodedImageFrame_1->fIndex2 = fIndex2;
			frames.insert(make_pair(source_pin_1, encodedImageFrame_1));
			source1->send(frames);
		}
		if (i % 3 == 1)
		{
			frame_container frames;
			auto encodedImageFrame_2 = source1->makeFrame(readDataSize, metadata);
			encodedImageFrame_2->fIndex2 = fIndex2;
			frames.insert(make_pair(source_pin_2, encodedImageFrame_2));
			source2->send(frames);
		}
		if (i % 3 == 2)
		{
			frame_container frames;
			auto encodedImageFrame_3 = source1->makeFrame(readDataSize, metadata);
			encodedImageFrame_3->fIndex2 = fIndex2;
			frames.insert(make_pair(source_pin_3, encodedImageFrame_3));
			source3->send(frames);
		}
		merge->step();
		if (i == 600)
		{
			BOOST_TEST(sink->pop().begin()->second->fIndex2 == i);
		}
		else if (i < 600 + 31)
		{
			BOOST_TEST(sink->try_pop().size() == 0);
		}
		else
		{
			BOOST_TEST(sink->pop().begin()->second->fIndex2 == (i - 30 + (i - 30 - 600) * 5));
		}
	}
	
	BOOST_TEST(source1->term());
	BOOST_TEST(source2->term());
	BOOST_TEST(source3->term());
	BOOST_TEST(merge->term());
	BOOST_TEST(sink->term());
}

BOOST_AUTO_TEST_CASE(errors)
{

}

BOOST_AUTO_TEST_SUITE_END()